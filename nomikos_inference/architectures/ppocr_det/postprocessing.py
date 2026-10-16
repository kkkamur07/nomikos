"""DB postprocess for the PP-OCRv6 detection graph, quad mode.

Mirrors PaddleX 3.7.0
``paddlex/inference/models/text_detection/processors.py::DBPostProcess`` in
quad mode (``box_type="quad"``, the default this model's config leaves
unset), which is ``boxes_from_bitmap``: binarise at ``thresh``,
``cv2.findContours`` over at most ``max_candidates`` contours,
``get_mini_boxes`` with a minimum side of 3, ``box_score_fast`` gated at
``box_thresh``, unclip expansion, a second ``get_mini_boxes`` with a minimum
side of 5, then scaling back to source coordinates with rounding and
clipping. ``score_mode`` is the default ``"fast"`` and ``use_dilation`` is
unset (False), so neither branch is reproduced here.

The ``poly`` branch (``polygons_from_bitmap`` with ``approxPolyDP``) is
deliberately not implemented: the config selects quad mode and the segment
contract carries four corner quads.
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


def _contour_quad(
    contour: np.ndarray, pred: np.ndarray, *, box_thresh: float, unclip_ratio: float
) -> tuple[np.ndarray, float] | None:
    """The scored, unclipped, clockwise quad of one contour, or None if it fails a gate."""
    box, short_side = get_mini_boxes(contour)
    if short_side < PPOCR_MIN_SIDE:
        return None
    score = box_score_fast(pred, np.asarray(box, dtype=np.float64))
    if score < box_thresh:
        return None
    expanded = unclip(np.asarray(box, dtype=np.float64), unclip_ratio)
    if expanded is None:
        return None
    grown, grown_side = get_mini_boxes(expanded)
    if grown_side < PPOCR_MIN_EXPANDED_SIDE:
        return None
    return order_quad_clockwise(np.asarray(grown, dtype=np.float64)), score


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
) -> list[DetectedQuad]:
    """Run the DB quad postprocess over one probability map.

    ``prob_map`` is the ``(H, W)`` detector output in resized-image space;
    the returned quads are in original image coordinates, clockwise from the
    top left, each with its ``box_score_fast`` score.
    """

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
        found = _contour_quad(contour, pred, box_thresh=box_thresh, unclip_ratio=unclip_ratio)
        if found is None:
            continue
        ordered, score = found
        mapped = [
            [
                float(min(max(round(float(x) * width_scale), 0), orig_width)),
                float(min(max(round(float(y) * height_scale), 0), orig_height)),
            ]
            for x, y in ordered
        ]
        quads.append(DetectedQuad(points=mapped, score=score))
    return quads


__all__ = [
    "DetectedQuad",
    "box_score_fast",
    "detect_lines",
    "get_mini_boxes",
    "order_quad_clockwise",
    "unclip",
]
