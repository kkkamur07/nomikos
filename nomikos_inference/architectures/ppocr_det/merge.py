"""Merge row fragments of one visual line into a single polygon.

The first refinement stage. Inside one column, quads that the row grouping
puts in the same row and that sit side by side along x are one visual line
split by the detector; they merge into one valid polygon with a baseline
running between the outer members' outer ends.
"""

from __future__ import annotations

from collections.abc import Iterator

from shapely.geometry import Polygon
from shapely.ops import unary_union

from nomikos_inference.architectures.ppocr_det.geometry import (
    Baseline,
    Work,
    bbox,
    is_initial,
    median,
    quad_height,
    synthetic_baseline_points,
    work_of_quad,
)
from nomikos_inference.architectures.ppocr_det.polygons import (
    convex_hull_of,
    intersect_outline,
    union_outlines,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout

DEFAULT_MERGE_GAP_RATIO = 0.25
DEFAULT_MERGE_MAX_OVERLAP_RATIO = 0.1
DEFAULT_MERGE_MAX_HEIGHT_RATIO = 2.0


def _bridge(prev: Work, nxt: Work) -> Polygon | None:
    """Rectangle over the x gap inside the pair's shared y range."""
    _, prev_ymin, prev_xmax, prev_ymax = prev.poly.bounds
    _, nxt_ymin, nxt_xmax, nxt_ymax = nxt.poly.bounds
    low, high = max(prev_ymin, nxt_ymin), min(prev_ymax, nxt_ymax)
    if high <= low:
        return None
    return Polygon([(prev_xmax, low), (nxt_xmax, low), (nxt_xmax, high), (prev_xmax, high)])


def _merged_polygon(parts: list[Polygon]) -> Polygon:
    """Union member polygons and gap bridges into one valid simple polygon."""
    merged = unary_union(parts)
    if isinstance(merged, Polygon) and merged.is_valid and not merged.is_empty:
        return merged
    hull = merged.convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        raise ValueError("merge produced no polygon")
    return hull


def _merged_baseline(
    works: list[Work], baseline_fraction: float, quad_points: list[list[list[float]]]
) -> Baseline:
    """Baseline from the outer ends of the outer members' baselines."""
    left = min(works, key=lambda work: (work.poly.bounds[0], work.poly.bounds[1]))
    right = max(works, key=lambda work: (work.poly.bounds[2], work.poly.bounds[1]))
    left_base = synthetic_baseline_points(quad_points[left.members[0]], baseline_fraction)
    right_base = synthetic_baseline_points(quad_points[right.members[0]], baseline_fraction)
    start = min(left_base, key=lambda point: point[0])
    end = max(right_base, key=lambda point: point[0])
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def _merge_run(
    run: list[int],
    quads: list[DetectedQuad],
    quad_points: list[list[list[float]]],
    baseline_fraction: float,
) -> Work:
    """Union one x-run of row mates plus gap bridges into a single line."""
    ordered = sorted(run, key=lambda i: bbox(quads[i].points)[0])
    parts: list[Polygon] = [Polygon(quad_points[i]) for i in ordered]
    members = [work_of_quad(i, quads[i], baseline_fraction) for i in ordered]
    for prev, nxt in zip(members, members[1:], strict=False):
        bridge = _bridge(prev, nxt)
        if bridge is not None:
            parts.append(bridge)
    merged = _merged_polygon(parts)
    total_area = sum(work.area for work in members)
    score = sum(work.score * work.area for work in members) / total_area if total_area > 0 else 0.0
    member_outlines = [work.outline for work in members if work.outline is not None]
    merged_outline = union_outlines(member_outlines) if member_outlines else None
    if merged_outline is None and member_outlines:
        hull = convex_hull_of([point for outline in member_outlines for point in outline])
        ring = [list(point) for point in merged.exterior.coords]
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        merged_outline = intersect_outline(hull, ring) or hull
    return Work(
        poly=merged,
        baseline=_merged_baseline(members, baseline_fraction, quad_points),
        score=score,
        members=tuple(ordered),
        merged_from=len(ordered),
        outline=merged_outline,
        polygon_fallback=any(work.polygon_fallback for work in members),
    )


def _row_runs(
    row: tuple[int, ...],
    quads: list[DetectedQuad],
    *,
    gap_min: float,
    gap_max: float,
    height_ratio: float,
) -> Iterator[list[int]]:
    """Split one row, left to right, into runs of side-by-side fragments.

    A quad extends the current run when its gap to the run's right edge lies
    in ``[gap_min, gap_max]`` and the run stays within ``height_ratio`` from
    its shortest to its tallest member.
    """
    ordered = sorted(
        row,
        key=lambda i: (
            bbox(quads[i].points)[0],
            bbox(quads[i].points)[1],
            tuple(round(c, 6) for point in quads[i].points for c in point),
        ),
    )
    run: list[int] = []
    run_max_h = run_min_h = run_xmax = 0.0
    for index in ordered:
        xmin, ymin, xmax, ymax = bbox(quads[index].points)
        height = ymax - ymin
        if run:
            gap = xmin - run_xmax
            new_max, new_min = max(run_max_h, height), min(run_min_h, height)
            if gap_min <= gap <= gap_max and new_max <= height_ratio * new_min:
                run.append(index)
                run_max_h, run_min_h, run_xmax = new_max, new_min, max(run_xmax, xmax)
                continue
            yield run
        run, run_max_h, run_min_h, run_xmax = [index], height, height, xmax
    if run:
        yield run


def _merge_row(
    row: tuple[int, ...],
    quads: list[DetectedQuad],
    quad_points: list[list[list[float]]],
    heights: list[float],
    *,
    gap_min: float,
    gap_max: float,
    height_ratio: float,
    initial_fallback: float,
    baseline_fraction: float,
) -> list[Work]:
    """One work per run of the row: merged runs, or lone members (maybe initials)."""
    works: list[Work] = []
    for run in _row_runs(row, quads, gap_min=gap_min, gap_max=gap_max, height_ratio=height_ratio):
        if len(run) > 1:
            works.append(_merge_run(run, quads, quad_points, baseline_fraction))
            continue
        work = work_of_quad(run[0], quads[run[0]], baseline_fraction)
        if is_initial(run[0], row, heights, initial_fallback, height_ratio):
            work.role = "initial"
        works.append(work)
    return works


def merge_row_fragments(
    quads: list[DetectedQuad],
    layout: PageLayout,
    *,
    gap_ratio: float = DEFAULT_MERGE_GAP_RATIO,
    max_overlap_ratio: float = DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    height_ratio: float = DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    baseline_fraction: float = 0.75,
) -> list[Work]:
    """Merge row mates split into fragments, one visual line per run.

    Inside one column, quads that the row grouping puts in the same row are
    one visual line: runs along x merge when the gap is at most
    ``gap_ratio`` times the page median quad height and the members overlap
    by at most ``max_overlap_ratio`` times the median height. Fragments of
    one visual line sit side by side; row mates that overlap belong to the
    overlap stage, not the merge stage. A member taller than
    ``height_ratio`` times a run mate never merges: a tall multi-line
    initial stays its own line with ``role`` ``"initial"``. Merging never
    crosses columns; spanning quads pass through untouched.
    """
    heights = [quad_height(quad) for quad in quads]
    median_height = median(heights, 1.0)
    quad_points = [[[float(x), float(y)] for x, y in quad.points] for quad in quads]
    builder_median = median(
        [heights[i] for column in layout.columns for i in column.builders], median_height
    )

    works: list[Work] = []
    for column in layout.columns:
        for row in column.rows:
            works.extend(
                _merge_row(
                    row,
                    quads,
                    quad_points,
                    heights,
                    gap_min=-max_overlap_ratio * median_height,
                    gap_max=gap_ratio * median_height,
                    height_ratio=height_ratio,
                    initial_fallback=builder_median,
                    baseline_fraction=baseline_fraction,
                )
            )
    in_column = {i for column in layout.columns for i in column.members}
    for index in sorted(set(range(len(quads))) - in_column):
        works.append(work_of_quad(index, quads[index], baseline_fraction))
    return works


def mark_initials(
    works: list[Work],
    quads: list[DetectedQuad],
    layout: PageLayout,
    height_ratio: float,
) -> None:
    """Flag tall singletons as initials when merging is switched off.

    The same rule merging applies to a lone run member, with the builders'
    median height (or 1 for a page without builders) as the fallback.
    """
    heights = [quad_height(quad) for quad in quads]
    reference_all = median([heights[i] for column in layout.columns for i in column.builders], 1.0)
    for column in layout.columns:
        for row in column.rows:
            for index in row:
                if is_initial(index, row, heights, reference_all, height_ratio):
                    works[index].role = "initial"


__all__ = [
    "DEFAULT_MERGE_GAP_RATIO",
    "DEFAULT_MERGE_MAX_HEIGHT_RATIO",
    "DEFAULT_MERGE_MAX_OVERLAP_RATIO",
    "mark_initials",
    "merge_row_fragments",
]
