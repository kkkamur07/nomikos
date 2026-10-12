"""DB postprocess for the PP-OCRv6 detection graph, poly and quad modes.

Mirrors PaddleX 3.7.0
``paddlex/inference/models/text_detection/processors.py::DBPostProcess``.
Quad mode (``box_type="quad"``) is ``boxes_from_bitmap``: binarise at
``thresh``, ``cv2.findContours`` over at most ``max_candidates`` contours,
``get_mini_boxes`` with a minimum side of 3, ``box_score_fast`` gated at
``box_thresh``, unclip expansion, a second ``get_mini_boxes`` with a minimum
side of 5, then scaling back to source coordinates with rounding and
clipping. Poly mode (``box_type="poly"``, passed explicitly by the served entry point) is
``polygons_from_bitmap``: the same contour also goes through
``approxPolyDP`` with the fixed ``contour_tolerance_px`` (skipped under 4
points), ``box_score_fast`` on the polygon gated at ``box_thresh``, unclip
expansion keeping the largest path, the same minimum side check of 5, then
the same scaling. Every poly detection carries both its quad (used for all
grouping decisions) and its polygon (used for the returned mask). When the
polygon step fails for a contour whose quad survived, the quad stands in as
the polygon and the detection is flagged as a fallback.
``score_mode`` is the default ``"fast"`` and ``use_dilation`` is
unset (False), so neither branch is reproduced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pyclipper

# PaddleX's ``min_size`` (3) and ``min_size + 2`` (5): the pre-expansion floor
# drops specks, the post-expansion floor drops boxes the expansion failed to
# grow (degenerate contours whose offset collapses).
PPOCR_MIN_SIDE = 3
PPOCR_MIN_EXPANDED_SIDE = PPOCR_MIN_SIDE + 2


@dataclass(frozen=True)
class DetectedQuad:
    """One detection in original image coordinates."""

    points: list[list[float]]
    score: float
    # Poly mode only: the simplified contour in original image coordinates,
    # paired one to one with the quad. ``None`` in quad mode. When the
    # polygon step fails for a surviving quad, the quad stands in and
    # ``polygon_fallback`` is True.
    polygon: list[list[float]] | None = None
    polygon_fallback: bool = False


def unclip(points: np.ndarray, ratio: float) -> np.ndarray | None:
    """Expand a contour by ``area * ratio / perimeter`` with round joins.

    This is PaddleX 3.7.0 ``DBPostProcess.unclip`` verbatim: the offset
    distance comes from OpenCV area and perimeter, and the offsetting
    itself is ``pyclipper`` with ``JT_ROUND``. When the offset splits into
    several paths the first one wins, exactly as PaddleX's ``Execute``
    fallback does. Returns ``None`` when the offset collapses, which the
    caller treats as a dropped candidate.
    """

    contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(contour) < 3:
        return None
    area = float(cv2.contourArea(contour))
    length = float(cv2.arcLength(contour, True))
    if length <= 0:
        return None
    distance = area * ratio / length
    if distance <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded = np.array(offset.Execute(distance))
    except ValueError:
        expanded = np.array(offset.Execute(distance)[0])
    if expanded.size == 0:
        return None
    coords = np.asarray(expanded.reshape(-1, 2), dtype=np.float64)
    if len(coords) < 3:
        return None
    return coords


def get_mini_boxes(contour: np.ndarray) -> tuple[list[list[float]], float]:
    """Minimum-area rectangle of a contour, ordered like PaddleX.

    Returns the four corners (left pair then right pair, top first within
    each pair) and the shorter side of the rectangle.
    """

    bounding = cv2.minAreaRect(np.asarray(contour, dtype=np.float32))
    corners = sorted((list(point) for point in cv2.boxPoints(bounding)), key=lambda point: point[0])
    if corners[1][1] > corners[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if corners[3][1] > corners[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [corners[index_1], corners[index_2], corners[index_3], corners[index_4]]
    return box, float(min(bounding[1]))


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """Mean probability inside a quad, via a filled mask over its bbox."""

    values = np.asarray(box, dtype=np.float64).reshape(-1, 2).copy()
    height, width = bitmap.shape[:2]
    xmin = max(0, min(math.floor(float(values[:, 0].min())), width - 1))
    xmax = max(0, min(math.ceil(float(values[:, 0].max())), width - 1))
    ymin = max(0, min(math.floor(float(values[:, 1].min())), height - 1))
    ymax = max(0, min(math.ceil(float(values[:, 1].max())), height - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    values[:, 0] -= xmin
    values[:, 1] -= ymin
    cv2.fillPoly(mask, [values.astype(np.int32)], (1,))
    return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def order_quad_clockwise(points: np.ndarray) -> np.ndarray:
    """Order four corners clockwise starting from the top left.

    The input is expected in ``get_mini_boxes`` order (left pair then right
    pair, top first within each pair), which is already a proper ring; this
    only pins the start corner and the orientation, by index permutation, so
    a corner can never be duplicated or lost however steep the quad is. The
    start corner is the top of the two leftmost corners, matching what
    ``get_mini_boxes`` emits first; a ring that runs counter-clockwise is
    reversed past the start corner. Clockwise is measured in image
    coordinates (y down), where it is a positive shoelace area.
    """

    corners = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(corners) != 4:
        raise ValueError("order_quad_clockwise needs exactly four corners")
    # Stable sort, so equal inputs always give equal outputs; the strict
    # y comparison mirrors ``get_mini_boxes``.
    by_x = sorted(range(4), key=lambda i: (corners[i][0], corners[i][1]))
    left_pair, right_pair = by_x[:2], by_x[2:]
    if corners[left_pair[1]][1] > corners[left_pair[0]][1]:
        left_top, left_bottom = left_pair
    else:
        left_top, left_bottom = left_pair[1], left_pair[0]
    if corners[right_pair[1]][1] > corners[right_pair[0]][1]:
        right_top, right_bottom = right_pair
    else:
        right_top, right_bottom = right_pair[1], right_pair[0]
    order = [left_top, right_top, right_bottom, left_bottom]
    ring = corners[order]
    area = float(
        np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    )
    if area < 0:
        order = [order[0], order[3], order[2], order[1]]
        ring = corners[order]
    return ring


#: Default contour tolerance of the poly branch, in detector pixels. Paddle's
#: ``polygons_from_bitmap`` uses 0.002 times the contour arc length (about
#: 2.7 px on a full line); the fixed 0.5 px keeps the wiggles that follow
#: the ink, which the serving simplifier used to flatten back out.
DEFAULT_CONTOUR_TOLERANCE_PX = 0.5


def _unclip_largest(points: np.ndarray, ratio: float) -> np.ndarray | None:
    """Expand a polygon like :func:`unclip` but keep the largest path.

    When the offset splits into several paths (a contour with holes or a
    pinch), Paddle's poly branch keeps the largest one instead of the first.
    Returns ``None`` when the offset collapses.
    """

    contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(contour) < 3:
        return None
    area = float(cv2.contourArea(contour))
    length = float(cv2.arcLength(contour, True))
    if length <= 0:
        return None
    distance = area * ratio / length
    if distance <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    paths = offset.Execute(distance)
    if not paths:
        return None
    areas = [
        abs(float(cv2.contourArea(np.asarray(path, dtype=np.float32).reshape(-1, 2))))
        for path in paths
        if len(np.asarray(path).reshape(-1, 2)) >= 3
    ]
    if not areas:
        return None
    best = int(np.argmax(np.asarray(areas, dtype=np.float64)))
    coords = np.asarray(paths[best], dtype=np.float64).reshape(-1, 2)
    if len(coords) < 3:
        return None
    return coords


def _polygon_for_contour(
    contour: np.ndarray,
    pred: np.ndarray,
    *,
    box_thresh: float,
    unclip_ratio: float,
    width_scale: float,
    height_scale: float,
    orig_width: int,
    orig_height: int,
    contour_tolerance_px: float = DEFAULT_CONTOUR_TOLERANCE_PX,
) -> list[list[float]] | None:
    """Build one Paddle ``polygons_from_bitmap`` polygon, or ``None``.

    The contour fit uses the fixed ``contour_tolerance_px`` instead of
    Paddle's ratio-scaled epsilon, so the polygon follows the ink. Any
    failure (fewer than 4 points after ``approxPolyDP``, a polygon score
    under ``box_thresh``, a collapsed unclip, a grown side under
    ``min_size + 2``) is ``None`` so the caller can fall back to the quad
    and lose no line.
    """

    contour_points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if float(cv2.arcLength(contour_points, True)) <= 0:
        return None
    approx = cv2.approxPolyDP(contour_points, max(float(contour_tolerance_px), 1e-9), True).reshape(
        -1, 2
    )
    if len(approx) < 4:
        return None
    score = box_score_fast(pred, np.asarray(approx, dtype=np.float64))
    if score < box_thresh:
        return None
    expanded = _unclip_largest(np.asarray(approx, dtype=np.float64), unclip_ratio)
    if expanded is None:
        return None
    _, grown_side = get_mini_boxes(expanded)
    if grown_side < PPOCR_MIN_EXPANDED_SIDE:
        return None
    return [
        [
            float(min(max(round(float(x) * width_scale), 0), orig_width)),
            float(min(max(round(float(y) * height_scale), 0), orig_height)),
        ]
        for x, y in expanded
    ]


def detect_lines(
    prob_map: np.ndarray,
    *,
    orig_width: int,
    orig_height: int,
    ratio_h: float,
    ratio_w: float,
    thresh: float = 0.2,
    box_thresh: float = 0.45,
    unclip_ratio: float = 1.4,
    max_candidates: int = 3000,
    box_type: str = "quad",
    contour_tolerance_px: float = DEFAULT_CONTOUR_TOLERANCE_PX,
) -> list[DetectedQuad]:
    """Run the DB postprocess over one probability map.

    ``prob_map`` is the ``(H, W)`` detector output in resized-image space;
    the returned quads are in original image coordinates, clockwise from the
    top left, each with its ``box_score_fast`` score. With
    ``box_type="poly"`` each detection also carries its polygon, built from
    the same contour that produced the quad; with ``box_type="quad"`` (the
    default) the loop below is exactly the old quad pipeline and the polygon
    stays ``None``. The served entry point passes ``"poly"`` explicitly.
    """

    if box_type not in ("quad", "poly"):
        raise ValueError('box_type must be "poly" or "quad"')
    pred = np.asarray(prob_map, dtype=np.float32)
    if pred.ndim != 2:
        raise ValueError("PP-OCRv6 det probability map must have shape (H, W)")
    height, width = pred.shape
    if height <= 0 or width <= 0:
        raise ValueError("PP-OCRv6 det probability map must not be empty")
    if ratio_h <= 0 or ratio_w <= 0:
        raise ValueError("PP-OCRv6 det resize ratios must be positive")

    bitmap = pred > thresh
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    width_scale = orig_width / width
    height_scale = orig_height / height

    quads: list[DetectedQuad] = []
    for contour in contours[:max_candidates]:
        box, short_side = get_mini_boxes(contour)
        if short_side < PPOCR_MIN_SIDE:
            continue
        score = box_score_fast(pred, np.asarray(box, dtype=np.float64))
        if score < box_thresh:
            continue
        expanded = unclip(np.asarray(box, dtype=np.float64), unclip_ratio)
        if expanded is None:
            continue
        grown, grown_side = get_mini_boxes(expanded)
        if grown_side < PPOCR_MIN_EXPANDED_SIDE:
            continue
        ordered = order_quad_clockwise(np.asarray(grown, dtype=np.float64))
        mapped = [
            [
                float(min(max(round(float(x) * width_scale), 0), orig_width)),
                float(min(max(round(float(y) * height_scale), 0), orig_height)),
            ]
            for x, y in ordered
        ]
        if box_type == "quad":
            quads.append(DetectedQuad(points=mapped, score=score))
            continue
        polygon = _polygon_for_contour(
            contour,
            pred,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            width_scale=width_scale,
            height_scale=height_scale,
            orig_width=orig_width,
            orig_height=orig_height,
            contour_tolerance_px=contour_tolerance_px,
        )
        if polygon is None:
            quads.append(
                DetectedQuad(
                    points=mapped,
                    score=score,
                    polygon=[list(point) for point in mapped],
                    polygon_fallback=True,
                )
            )
        else:
            quads.append(DetectedQuad(points=mapped, score=score, polygon=polygon))
    return quads


__all__ = [
    "DEFAULT_CONTOUR_TOLERANCE_PX",
    "DetectedQuad",
    "box_score_fast",
    "detect_lines",
    "get_mini_boxes",
    "order_quad_clockwise",
    "unclip",
]
