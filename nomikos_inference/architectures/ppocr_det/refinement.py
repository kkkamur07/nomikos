"""Refine raw PP-OCRv6 detection quads into clean manuscript lines.

Pipeline order inside the adapter: detect quads, merge row fragments
(``merge``), resolve overlaps (``overlaps``), classify suspects (here), order,
build the response. Everything here is pure geometry relative to the page's
own medians and the column bands that ``reading_order`` builds; no absolute
pixel threshold transfers between manuscripts. Overlap is shared area
divided by the smaller polygon's area: 5 to 7 percent is normal (ascenders,
descenders), 20 percent or more is treated as a defect.
"""

from __future__ import annotations

import numpy as np

from nomikos_inference.architectures.ppocr_det.geometry import (
    RefinedLine,
    Work,
    median,
    quad_height,
    work_of_quad,
)
from nomikos_inference.architectures.ppocr_det.merge import (
    DEFAULT_MERGE_GAP_RATIO,
    DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    mark_initials,
    merge_row_fragments,
)
from nomikos_inference.architectures.ppocr_det.overlaps import (
    DEFAULT_OVERLAP_CUT_THRESHOLD,
    resolve_overlaps,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout

# A detection counts as inside a column band when it stays within this
# tolerance of the band edges. Measured at 1 px: tighter clips real merged
# lines that stick out past their builders, looser admits margin noise.
BAND_TOLERANCE_PX = 1.0
# Bounding-box angle off the page dominant angle that marks a non-text
# shape. Real text lines sit within a few degrees; initials are exempt by
# role rather than by angle.
SUSPECT_ANGLE_DEGREES = 20.0

Band = tuple[float, float]


def _builder_bands(layout: PageLayout, quads: list[DetectedQuad]) -> list[Band]:
    """Text bands: x ranges over each column's founding builders."""
    bands: list[Band] = []
    for column in layout.columns:
        if not column.builders:
            continue
        xs = [point[0] for i in column.builders for point in quads[i].points]
        bands.append((min(xs), max(xs)))
    return sorted(bands)


def _text_block(layout: PageLayout, quads: list[DetectedQuad]) -> Band | None:
    """Top and bottom of the builders' overall y range."""
    ys = [
        point[1] for column in layout.columns for i in column.builders for point in quads[i].points
    ]
    if not ys:
        return None
    return min(ys), max(ys)


def _bbox_angle(points: list[list[float]]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width <= 0:
        return 90.0
    return float(np.degrees(np.arctan2(height, width)))


def _in_gutter(work: Work, block: Band | None, gaps: list[tuple[Band, Band]]) -> bool:
    """Strictly between two neighbouring bands and inside the text block."""
    xmin, ymin, xmax, ymax = work.poly.bounds
    centre_y = (ymin + ymax) / 2.0
    return (
        block is not None
        and block[0] <= centre_y <= block[1]
        and any(xmin > left_high and xmax < right_low for (_, left_high), (right_low, _) in gaps)
    )


def _outside_bands(work: Work, bands: list[Band]) -> bool:
    xmin, _, xmax, _ = work.poly.bounds
    return not any(
        xmin >= low - BAND_TOLERANCE_PX and xmax <= high + BAND_TOLERANCE_PX for low, high in bands
    )


def classify_suspects(
    works: list[Work],
    layout: PageLayout,
    quads: list[DetectedQuad],
) -> None:
    """Flag non-text detections in place, never deleting.

    A singleton that is not an initial is a suspect when it lies outside
    every builder column band, unless it sits strictly between two bands
    inside the text block top to bottom (a gutter numeral between two
    columns keeps its place and is never a suspect by angle either). Every
    other non-initial whose bounding-box angle is more than
    ``SUSPECT_ANGLE_DEGREES`` off the page dominant angle is a suspect
    whatever its position. Merged lines always have a column home.
    """
    bands = _builder_bands(layout, quads)
    block = _text_block(layout, quads)
    angles = [
        _bbox_angle([[float(x), float(y)] for x, y in work.poly.exterior.coords]) for work in works
    ]
    dominant = median(angles, 0.0)
    gaps = list(zip(bands, bands[1:], strict=False))
    for position, work in enumerate(works):
        if work.role == "initial" or _in_gutter(work, block, gaps):
            continue
        if work.merged_from == 1 and _outside_bands(work, bands):
            work.suspect = True
            work.suspect_reason = "outside_bands"
        elif abs(angles[position] - dominant) > SUSPECT_ANGLE_DEGREES:
            work.suspect = True
            work.suspect_reason = "angle"


def _refined_line(work: Work) -> RefinedLine:
    ring = [list(point) for point in work.poly.exterior.coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return RefinedLine(
        points=[[float(x), float(y)] for x, y in ring],
        baseline=[
            [float(work.baseline[0][0]), float(work.baseline[0][1])],
            [float(work.baseline[1][0]), float(work.baseline[1][1])],
        ],
        score=work.score,
        members=work.members,
        role=work.role,
        suspect=work.suspect,
        suspect_reason=work.suspect_reason,
        merged_from=work.merged_from,
        overlap_unresolved=work.overlap_unresolved,
    )


def refine_to_lines(
    quads: list[DetectedQuad],
    layout: PageLayout,
    *,
    baseline_fraction: float = 0.75,
    merge: bool = True,
    resolve: bool = True,
    classify: bool = True,
    merge_gap_ratio: float = DEFAULT_MERGE_GAP_RATIO,
    merge_max_overlap_ratio: float = DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    merge_max_height_ratio: float = DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    overlap_cut_threshold: float = DEFAULT_OVERLAP_CUT_THRESHOLD,
) -> list[RefinedLine]:
    """Run the refinement stage and return unordered manuscript lines."""
    if merge:
        works = merge_row_fragments(
            quads,
            layout,
            gap_ratio=merge_gap_ratio,
            max_overlap_ratio=merge_max_overlap_ratio,
            height_ratio=merge_max_height_ratio,
            baseline_fraction=baseline_fraction,
        )
    else:
        works = [work_of_quad(i, quad, baseline_fraction) for i, quad in enumerate(quads)]
        mark_initials(works, quads, layout, merge_max_height_ratio)
    if resolve:
        median_height = median([quad_height(quad) for quad in quads], 1.0)
        works = resolve_overlaps(
            works, layout, cut_threshold=overlap_cut_threshold, median_height=median_height
        )
    if classify:
        classify_suspects(works, layout, quads)
    return [_refined_line(work) for work in works]


__all__ = [
    "BAND_TOLERANCE_PX",
    "DEFAULT_MERGE_GAP_RATIO",
    "DEFAULT_MERGE_MAX_OVERLAP_RATIO",
    "DEFAULT_MERGE_MAX_HEIGHT_RATIO",
    "DEFAULT_OVERLAP_CUT_THRESHOLD",
    "SUSPECT_ANGLE_DEGREES",
    "RefinedLine",
    "classify_suspects",
    "merge_row_fragments",
    "refine_to_lines",
    "resolve_overlaps",
]
