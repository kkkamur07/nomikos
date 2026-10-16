"""SegmentRunResponse construction for PP-OCRv6 detection quads."""

from __future__ import annotations

from nomikos_inference.architectures.ppocr_det.geometry import (
    RefinedLine,
    synthetic_baseline_points,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout, order_lines
from nomikos_inference.contracts.common import MAX_SEGMENT_LINES
from nomikos_inference.contracts.segment import SegmentBlock, SegmentLine, SegmentRunResponse

#: Where across the quad the synthetic baseline sits, from the top edge (0)
#: to the bottom edge (1). PP-OCRv6 emits boxes, not baselines, and kraken
#: baselines sit near the bottom of the letter bodies, so the default puts
#: the polyline three quarters down the quad sides.
DEFAULT_BASELINE_FRACTION = 0.75

Metadata = dict[str, str | float | int | bool]


def _page_block(image_width: int, image_height: int) -> SegmentBlock:
    return SegmentBlock(
        external_id="ppocr-det-block-1",
        order=0,
        box={
            "points": [
                [0.0, 0.0],
                [float(image_width), 0.0],
                [float(image_width), float(image_height)],
                [0.0, float(image_height)],
            ]
        },
    )


def build_ppocr_det_response(
    image_width: int,
    image_height: int,
    quads: list[DetectedQuad],
    *,
    baseline_fraction: float = DEFAULT_BASELINE_FRACTION,
    reading_direction: str = "ltr",
) -> SegmentRunResponse:
    """Number detection quads in reading order under one full-page block.

    Past ``MAX_SEGMENT_LINES`` the highest scoring quads survive (score ties
    keep input order), then the survivors are ordered; numbering always
    follows reading order from 1.
    """

    ranked = sorted(range(len(quads)), key=lambda i: (-quads[i].score, i))
    survivors = [quads[i] for i in ranked[:MAX_SEGMENT_LINES]]
    reading = order_lines(survivors, direction=reading_direction)

    block = _page_block(image_width, image_height)

    lines = []
    for position, quad_index in enumerate(reading):
        quad = survivors[quad_index]
        source_metadata: Metadata = {
            "detector": "pp-ocrv6-det",
            "score": float(quad.score),
            "baseline_source": "quad_axis",
            "baseline_fraction": float(baseline_fraction),
        }
        lines.append(
            SegmentLine(
                external_id=f"ppocr-det-line-{position + 1}",
                order=position,
                block_external_id=block.external_id,
                baseline={"points": synthetic_baseline_points(quad.points, baseline_fraction)},
                mask=None,
                points=[[float(x), float(y)] for x, y in quad.points],
                kraken_ceiling=None,
                source_metadata=source_metadata,
            )
        )
    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)


def _mid(item: RefinedLine) -> tuple[float, float]:
    first, second = item.baseline
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


def _refined_metadata(item: RefinedLine, baseline_fraction: float) -> Metadata:
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
) -> SegmentRunResponse:
    """Number refined lines in reading order under one full-page block.

    ``items`` are refinement output whose ``members`` name the original quad
    indices, so the body order follows ``order_lines`` over the raw quads:
    each refined line is emitted at its first member's position, initials
    first within their row group, and suspects (unless dropped) after every
    body line, top to bottom. Past ``MAX_SEGMENT_LINES`` the highest
    scoring items survive, ties broken geometrically.
    """

    emission = _emission_order(_survivors(items, noise_policy), quads, reading_direction)
    _initials_first(emission, layout)
    if noise_policy == "flag":
        emission = _suspects_last(emission)

    block = _page_block(image_width, image_height)
    lines = [
        SegmentLine(
            external_id=f"ppocr-det-line-{position + 1}",
            order=position,
            block_external_id=block.external_id,
            baseline={"points": [[float(x), float(y)] for x, y in item.baseline]},
            mask=None,
            points=[[float(x), float(y)] for x, y in item.points],
            kraken_ceiling=None,
            source_metadata=_refined_metadata(item, baseline_fraction),
        )
        for position, item in enumerate(emission)
    ]
    return SegmentRunResponse(blocks=[block] if lines else [], lines=lines)


__all__ = [
    "DEFAULT_BASELINE_FRACTION",
    "build_ppocr_det_response",
    "build_refined_ppocr_det_response",
]
