"""SegmentRunResponse construction for PP-OCRv6 detection quads and polygons."""

from __future__ import annotations

from typing import Any

from nomikos_inference.architectures.ppocr_det.polygons import (
    DEFAULT_OUTLINE_TOLERANCE_PX,
    DEFAULT_VERTICAL_GROWTH,
    grow_outlines_vertical,
    polyline_baseline,
    simplify_outline,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import (
    PageLayout,
    layout_lines,
    order_lines,
)
from nomikos_inference.contracts.common import MAX_SEGMENT_LINES
from nomikos_inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse

#: Where across the quad the synthetic baseline sits, from the top edge (0)
#: to the bottom edge (1). PP-OCRv6 emits boxes, not baselines, and kraken
#: baselines sit near the bottom of the letter bodies, so the default puts
#: the polyline three quarters down the quad sides.
DEFAULT_BASELINE_FRACTION = 0.75


def synthetic_baseline_points(quad: list[list[float]], fraction: float) -> list[list[float]]:
    """Two-point baseline across a clockwise quad at ``fraction`` down."""
    top_left, top_right, bottom_right, bottom_left = quad
    left = [
        top_left[0] + fraction * (bottom_left[0] - top_left[0]),
        top_left[1] + fraction * (bottom_left[1] - top_left[1]),
    ]
    right = [
        top_right[0] + fraction * (bottom_right[0] - top_right[0]),
        top_right[1] + fraction * (bottom_right[1] - top_right[1]),
    ]
    return [left, right]


def _baseline_points(quad: list[list[float]], fraction: float) -> list[list[float]]:
    return synthetic_baseline_points(quad, fraction)


def _bounds_box(polys: list[list[list[float]]]) -> dict[str, Any]:
    """Axis-aligned bounds of several polygons as a box dict."""
    xs = [float(point[0]) for poly in polys for point in poly]
    ys = [float(point[1]) for poly in polys for point in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _regions(
    line_columns: list[int | None], line_points: list[list[list[float]]]
) -> tuple[list[SegmentBlock], list[str]]:
    """Blocks for lines in reading order, and each line's block.

    One block per column, and one per spanning band: a run of consecutive lines that
    belong to no column (a title, a heading across the columns, a footer). Blocks are
    numbered by the reading position of their first line, so a title stays first when
    the page is numbered region by region (ADR 0011).
    """
    keys: list[tuple[str, int]] = []
    band = -1
    previous: int | None = 0
    for column in line_columns:
        if column is None:
            if previous is not None:
                band += 1
            keys.append(("band", band))
        else:
            keys.append(("column", column))
        previous = column
    external_of: dict[tuple[str, int], str] = {}
    polys_of: dict[tuple[str, int], list[list[list[float]]]] = {}
    for key, points in zip(keys, line_points, strict=True):
        if key not in external_of:
            external_of[key] = f"ppocr-det-region-{len(external_of) + 1}"
        polys_of.setdefault(key, []).append(points)
    blocks = [
        SegmentBlock(external_id=external_id, order=order, box=_bounds_box(polys_of[key]))
        for order, (key, external_id) in enumerate(external_of.items())
    ]
    return blocks, [external_of[key] for key in keys]


def _poly_points_for(
    quad: DetectedQuad,
    baseline_fraction: float,
    image_width: int,
    image_height: int,
    outline_tolerance_px: float,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], bool]:
    """Ungrown polygon, quad ring and baseline polyline for one detection.

    Returns the simplified outline, the quad ring and baseline the growth
    step needs for direction, the baseline polyline sampled from the ungrown
    outline, and whether the quad stood in as the polygon.
    """
    quad_points = [[float(x), float(y)] for x, y in quad.points]
    outline = quad.polygon if quad.polygon is not None else quad_points
    fallback = bool(quad.polygon_fallback) or quad.polygon is None
    quad_baseline = _baseline_points(quad_points, baseline_fraction)
    simplified = simplify_outline(
        outline,
        outline_tolerance_px=outline_tolerance_px,
        page_width=image_width,
        page_height=image_height,
    )
    if simplified is None:
        return quad_points, quad_points, quad_baseline, True
    baseline = polyline_baseline(simplified, quad_points, baseline_fraction, baseline=quad_baseline)
    if baseline is None:
        baseline = quad_baseline
    return simplified, quad_points, baseline, fallback


def build_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
    box_type: str = "quad",
    outline_tolerance_px: float = DEFAULT_OUTLINE_TOLERANCE_PX,
    vertical_growth: float = DEFAULT_VERTICAL_GROWTH,
) -> SegmentRunResponse:
    """Number detection quads in reading order, one block per column.

    Past ``MAX_SEGMENT_LINES`` the highest scoring quads survive (score ties
    keep input order), then the survivors are ordered; numbering always
    follows reading order from 1. Blocks follow ``column_order``: region n
    is the nth column in reading order, with the bounds of its member
    lines as its box. Spanning lines read as their own band, in a block of
    their own; blocks are numbered by their first line's reading position.
    With ``box_type="poly"`` each line serves its polygon grown across the
    line and a baseline polyline sampled from the ungrown outline; the quad
    path is unchanged.
    """

    if box_type not in ("quad", "poly"):
        raise ValueError('box_type must be "poly" or "quad"')
    ranked = sorted(range(len(quads)), key=lambda i: (-quads[i].score, i))
    survivors = [quads[i] for i in ranked[:MAX_SEGMENT_LINES]]
    reading = order_lines(survivors, direction=reading_direction)
    layout = layout_lines(survivors, direction=reading_direction)
    per_survivor: list[tuple] = []
    grown_points: list[list[list[float]]] = []
    if box_type == "poly":
        for quad in survivors:
            per_survivor.append(
                _poly_points_for(
                    quad, baseline_fraction, image_width, image_height, outline_tolerance_px
                )
            )
        grown_points = grow_outlines_vertical(
            [simplified for simplified, _, _, _ in per_survivor],
            [ring for _, ring, _, _ in per_survivor],
            factor=float(vertical_growth),
            baselines=[_baseline_points(quad.points, baseline_fraction) for quad in survivors],
            page_width=image_width,
            page_height=image_height,
            outline_tolerance_px=outline_tolerance_px,
        )

    spanning = set(layout.spanning)
    column_of: dict[int, int] = {}
    for position, column in enumerate(layout.columns):
        for member in column.members:
            column_of[member] = position
    blocks, block_of_line = _regions(
        [None if index in spanning else column_of[index] for index in reading],
        [
            grown_points[index] if box_type == "poly" else survivors[index].points
            for index in reading
        ],
    )

    lines = []
    for position, quad_index in enumerate(reading):
        quad = survivors[quad_index]
        block_external_id = block_of_line[position]
        source_metadata: dict[str, Any] = {
            "detector": "pp-ocrv6-det",
            "score": float(quad.score),
            "baseline_source": "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
        if box_type == "poly":
            _, _, served_baseline, fallback = per_survivor[quad_index]
            if fallback:
                source_metadata["polygon_fallback"] = True
            lines.append(
                SegmentLine(
                    external_id=f"ppocr-det-line-{position + 1}",
                    order=position,
                    block_external_id=block_external_id,
                    baseline={"points": served_baseline},
                    mask=None,
                    points=grown_points[quad_index],
                    kraken_ceiling=None,
                    source_metadata=source_metadata,
                )
            )
            continue
        lines.append(
            SegmentLine(
                external_id=f"ppocr-det-line-{position + 1}",
                order=position,
                block_external_id=block_external_id,
                baseline={"points": _baseline_points(quad.points, baseline_fraction)},
                mask=None,
                points=[[float(x), float(y)] for x, y in quad.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=blocks if lines else [], lines=lines)


def build_refined_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    items: list,
    layout: PageLayout,
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
    noise_policy: str = "flag",
    box_type: str = "quad",
) -> SegmentRunResponse:
    """Number refined lines in reading order, one block per column.

    ``items`` are refinement output whose ``members`` name the original quad
    indices, so the body order follows ``order_lines`` over the raw quads:
    each refined line is emitted at its first member's position, initials
    first within their row group, and suspects (unless dropped) after every
    body line, top to bottom. Past ``MAX_SEGMENT_LINES`` the highest
    scoring items survive, ties broken geometrically. Blocks follow
    reading position over the refined lines that landed in each column;
    spanning lines read in a block of their own per band.
    """

    if box_type not in ("quad", "poly"):
        raise ValueError('box_type must be "poly" or "quad"')
    scoped = [item for item in items if not (noise_policy == "drop" and item.suspect)]

    def _mid(item: Any) -> tuple[float, float]:
        first, second = item.baseline[0], item.baseline[-1]
        return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)

    ranked = sorted(
        range(len(scoped)),
        key=lambda i: (-scoped[i].score, _mid(scoped[i])[1], _mid(scoped[i])[0]),
    )
    survivors = [scoped[i] for i in ranked[:MAX_SEGMENT_LINES]]
    owner: dict[int, Any] = {}
    for item in survivors:
        for member in item.members:
            owner.setdefault(member, item)
    emission: list[Any] = []
    seen: set[int] = set()
    for quad_index in order_lines(quads, direction=reading_direction):
        item = owner.get(quad_index)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        emission.append(item)

    group_of: dict[int, tuple[int, int]] = {}
    for column_index, column in enumerate(layout.columns):
        for row_index, row in enumerate(column.rows):
            for member in row:
                group_of[member] = (column_index, row_index)
    slots: dict[tuple[int, int] | None, list[int]] = {}
    for position, item in enumerate(emission):
        groups = {group_of.get(member) for member in item.members} - {None}
        key = next(iter(groups)) if len(groups) == 1 else None
        slots.setdefault(key, []).append(position)
    for key, positions in slots.items():
        if key is None:
            continue
        first = [position for position in positions if emission[position].role == "initial"]
        rest = [position for position in positions if emission[position].role != "initial"]
        ordered = [emission[position] for position in first + rest]
        for position, item in zip(positions, ordered, strict=True):
            emission[position] = item

    if noise_policy == "flag":
        body = [item for item in emission if not item.suspect]
        tail = sorted(
            (item for item in emission if item.suspect),
            key=lambda item: (_mid(item)[1], _mid(item)[0]),
        )
        emission = body + tail

    spanning = set(layout.spanning)
    column_of_member: dict[int, int] = {}
    for column_index, column in enumerate(layout.columns):
        for member in column.members:
            column_of_member[member] = column_index

    def _item_column(item: Any) -> int | None:
        columns = set()
        for member in item.members:
            if member in spanning:
                return None
            column = column_of_member.get(member)
            if column is None:
                return None
            columns.add(column)
        if len(columns) == 1:
            return next(iter(columns))
        return None

    item_columns = [_item_column(item) for item in emission]
    blocks, block_of_line = _regions(item_columns, [item.points for item in emission])

    lines = []
    for position, item in enumerate(emission):
        block_external_id = block_of_line[position]
        source_metadata: dict[str, Any] = {
            "detector": "pp-ocrv6-det",
            "score": float(item.score),
            "baseline_source": "merged_axis" if item.merged_from > 1 else "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
        if item.role == "initial":
            source_metadata["role"] = "initial"
        if item.merged_from > 1:
            source_metadata["merged_from"] = item.merged_from
        if item.suspect:
            source_metadata["suspect"] = True
            source_metadata["suspect_reason"] = item.suspect_reason
        if item.overlap_unresolved:
            source_metadata["overlap_unresolved"] = True
        if box_type == "poly" and getattr(item, "polygon_fallback", False):
            source_metadata["polygon_fallback"] = True
        lines.append(
            SegmentLine(
                external_id=f"ppocr-det-line-{position + 1}",
                order=position,
                block_external_id=block_external_id,
                baseline={"points": [[float(x), float(y)] for x, y in item.baseline]},
                mask=None,
                points=[[float(x), float(y)] for x, y in item.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=blocks if lines else [], lines=lines)


__all__ = [
    "build_ppocr_det_response",
    "build_refined_ppocr_det_response",
    "synthetic_baseline_points",
]
