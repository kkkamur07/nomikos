"""SegmentRunResponse construction for PP-OCRv6 detection quads and polygons."""

from __future__ import annotations

from typing import Any

from nomikos_inference.architectures.ppocr_det.geometry import (
    RefinedLine,
    synthetic_baseline_points,
)
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

Metadata = dict[str, object]


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
    quad_baseline = synthetic_baseline_points(quad_points, baseline_fraction)
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
            baselines=[
                synthetic_baseline_points(quad.points, baseline_fraction) for quad in survivors
            ],
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
        source_metadata: Metadata = {
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
                baseline={"points": synthetic_baseline_points(quad.points, baseline_fraction)},
                mask=None,
                points=[[float(x), float(y)] for x, y in quad.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=blocks if lines else [], lines=lines)


def _mid(item: RefinedLine) -> tuple[float, float]:
    first, second = item.baseline[0], item.baseline[-1]
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _survivors(items: list[RefinedLine], noise_policy: str) -> list[RefinedLine]:
    """Drop suspects under ``"drop"``, then keep the best ``MAX_SEGMENT_LINES``."""
    scoped = [item for item in items if not (noise_policy == "drop" and item.suspect)]
    ranked = sorted(
        range(len(scoped)),
        key=lambda i: (-scoped[i].score, _mid(scoped[i])[1], _mid(scoped[i])[0]),
    )
    return [scoped[i] for i in ranked[:MAX_SEGMENT_LINES]]


def _emission_order(
    survivors: list[RefinedLine], quads: list[DetectedQuad], direction: str
) -> list[RefinedLine]:
    """Each line once, at its first member's place in the raw quads' order."""
    owner: dict[int, RefinedLine] = {}
    for item in survivors:
        for member in item.members:
            owner.setdefault(member, item)
    emission: list[RefinedLine] = []
    seen: set[int] = set()
    for quad_index in order_lines(quads, direction=direction):
        owned = owner.get(quad_index)
        if owned is None or id(owned) in seen:
            continue
        seen.add(id(owned))
        emission.append(owned)
    return emission


def _row_group_of(layout: PageLayout) -> dict[int, tuple[int, int]]:
    """Quad index to its (column, row) group."""
    return {
        member: (column_index, row_index)
        for column_index, column in enumerate(layout.columns)
        for row_index, row in enumerate(column.rows)
        for member in row
    }


def _initials_first(emission: list[RefinedLine], layout: PageLayout) -> None:
    """Within each row group, move initials ahead of the row's other lines.

    A line whose members span more than one row group keeps its place.
    """
    group_of = _row_group_of(layout)
    slots: dict[tuple[int, int], list[int]] = {}
    for position, item in enumerate(emission):
        groups = {group_of[member] for member in item.members if member in group_of}
        if len(groups) == 1:
            slots.setdefault(groups.pop(), []).append(position)
    for positions in slots.values():
        ordered = sorted(
            (emission[position] for position in positions), key=lambda item: item.role != "initial"
        )
        for position, item in zip(positions, ordered, strict=True):
            emission[position] = item


def _suspects_last(emission: list[RefinedLine]) -> list[RefinedLine]:
    body = [item for item in emission if not item.suspect]
    tail = sorted(
        (item for item in emission if item.suspect),
        key=lambda item: (_mid(item)[1], _mid(item)[0]),
    )
    return body + tail


def _item_columns(emission: list[RefinedLine], layout: PageLayout) -> list[int | None]:
    """Each line's column, or None when it spans columns or has a spanning member."""
    spanning = set(layout.spanning)
    column_of_member: dict[int, int] = {}
    for column_index, column in enumerate(layout.columns):
        for member in column.members:
            column_of_member[member] = column_index

    def _item_column(item: RefinedLine) -> int | None:
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

    return [_item_column(item) for item in emission]


def _refined_metadata(item: RefinedLine, baseline_fraction: float, box_type: str) -> Metadata:
    metadata: Metadata = {
        "detector": "pp-ocrv6-det",
        "score": float(item.score),
        "baseline_source": "merged_axis" if item.merged_from > 1 else "quad_axis",
        "baseline_fraction": float(baseline_fraction),
    }
    if item.role == "initial":
        metadata["role"] = "initial"
    if item.merged_from > 1:
        metadata["merged_from"] = item.merged_from
    if item.suspect:
        metadata["suspect"] = True
        metadata["suspect_reason"] = item.suspect_reason
    if item.overlap_unresolved:
        metadata["overlap_unresolved"] = True
    if box_type == "poly" and item.polygon_fallback:
        metadata["polygon_fallback"] = True
    return metadata


def build_refined_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    items: list[RefinedLine],
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
    emission = _emission_order(_survivors(items, noise_policy), quads, reading_direction)
    _initials_first(emission, layout)
    if noise_policy == "flag":
        emission = _suspects_last(emission)

    blocks, block_of_line = _regions(
        _item_columns(emission, layout), [item.points for item in emission]
    )
    lines = [
        SegmentLine(
            external_id=f"ppocr-det-line-{position + 1}",
            order=position,
            block_external_id=block_of_line[position],
            baseline={"points": [[float(x), float(y)] for x, y in item.baseline]},
            mask=None,
            points=[[float(x), float(y)] for x, y in item.points],
            kraken_ceiling=None,
            source_metadata=_refined_metadata(item, baseline_fraction, box_type),
        )
        for position, item in enumerate(emission)
    ]
    return SegmentRunResponse(blocks=blocks if lines else [], lines=lines)


__all__ = [
    "DEFAULT_BASELINE_FRACTION",
    "build_ppocr_det_response",
    "build_refined_ppocr_det_response",
    "synthetic_baseline_points",
]
