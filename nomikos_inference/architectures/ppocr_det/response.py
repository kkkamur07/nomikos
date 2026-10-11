"""SegmentRunResponse construction for PP-OCRv6 detection quads."""

from __future__ import annotations

from typing import Any

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


def build_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
) -> SegmentRunResponse:
    """Number detection quads in reading order, one block per column.

    Past ``MAX_SEGMENT_LINES`` the highest scoring quads survive (score ties
    keep input order), then the survivors are ordered; numbering always
    follows reading order from 1. Blocks follow ``column_order``: region n
    is the nth column in reading order, with the bounds of its member
    quads as its box. Spanning lines read as their own band and carry no
    block.
    """

    ranked = sorted(range(len(quads)), key=lambda i: (-quads[i].score, i))
    survivors = [quads[i] for i in ranked[:MAX_SEGMENT_LINES]]
    reading = order_lines(survivors, direction=reading_direction)
    layout = layout_lines(survivors, direction=reading_direction)

    spanning = set(layout.spanning)
    column_of: dict[int, int] = {}
    for position, column in enumerate(layout.columns):
        for member in column.members:
            column_of[member] = position
    blocks: list[SegmentBlock] = []
    block_of_column: dict[int, str] = {}
    for column_position in layout.column_order:
        members = [survivors[member] for member in layout.columns[column_position].members]
        if not members:
            continue
        external_id = f"ppocr-det-region-{len(blocks) + 1}"
        block_of_column[column_position] = external_id
        blocks.append(
            SegmentBlock(
                external_id=external_id,
                order=len(blocks),
                box=_bounds_box([quad.points for quad in members]),
            )
        )

    lines = []
    for position, quad_index in enumerate(reading):
        quad = survivors[quad_index]
        if quad_index in spanning:
            block_external_id = None
        else:
            block_external_id = block_of_column[column_of[quad_index]]
        source_metadata: dict[str, Any] = {
            "detector": "pp-ocrv6-det",
            "score": float(quad.score),
            "baseline_source": "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
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
) -> SegmentRunResponse:
    """Number refined lines in reading order, one block per column.

    ``items`` are refinement output whose ``members`` name the original quad
    indices, so the body order follows ``order_lines`` over the raw quads:
    each refined line is emitted at its first member's position, initials
    first within their row group, and suspects (unless dropped) after every
    body line, top to bottom. Past ``MAX_SEGMENT_LINES`` the highest
    scoring items survive, ties broken geometrically. Blocks follow
    ``column_order`` over the refined lines that landed in each column;
    spanning lines carry no block.
    """

    scoped = [item for item in items if not (noise_policy == "drop" and item.suspect)]

    def _mid(item: Any) -> tuple[float, float]:
        first, second = item.baseline
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
    points_by_column: dict[int, list[list[list[float]]]] = {}
    for item, column in zip(emission, item_columns, strict=True):
        if column is not None:
            points_by_column.setdefault(column, []).append(item.points)
    blocks: list[SegmentBlock] = []
    block_of_column: dict[int, str] = {}
    for column_position in layout.column_order:
        polys = points_by_column.get(column_position)
        if not polys:
            continue
        external_id = f"ppocr-det-region-{len(blocks) + 1}"
        block_of_column[column_position] = external_id
        blocks.append(
            SegmentBlock(
                external_id=external_id,
                order=len(blocks),
                box=_bounds_box(polys),
            )
        )

    lines = []
    for position, item in enumerate(emission):
        column = item_columns[position]
        block_external_id = block_of_column.get(column) if column is not None else None
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
