"""Refine raw PP-OCRv6 detection quads into clean manuscript lines.

Pipeline order inside the adapter: detect quads, merge row fragments,
resolve overlaps, classify suspects, order, build the response. Everything
here is pure geometry relative to the page's own medians and the column
bands that ``reading_order`` builds; no absolute pixel threshold transfers
between manuscripts. Overlap is shared area divided by the smaller
polygon's area: 5 to 7 percent is normal (ascenders, descenders), 20
percent or more is treated as a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout
from nomikos_inference.architectures.ppocr_det.response import synthetic_baseline_points

DEFAULT_MERGE_GAP_RATIO = 0.25
DEFAULT_MERGE_MAX_OVERLAP_RATIO = 0.1
DEFAULT_MERGE_MAX_HEIGHT_RATIO = 2.0
DEFAULT_OVERLAP_CUT_THRESHOLD = 0.20
DUPLICATE_SHARED_RATIO = 0.50
# A detection counts as inside a column band when it stays within this
# tolerance of the band edges. Measured at 1 px: tighter clips real merged
# lines that stick out past their builders, looser admits margin noise.
BAND_TOLERANCE_PX = 1.0
# Bounding-box angle off the page dominant angle that marks a non-text
# shape. Real text lines sit within a few degrees; initials are exempt by
# role rather than by angle.
SUSPECT_ANGLE_DEGREES = 20.0
_MAX_OVERLAP_PASSES = 100


@dataclass
class RefinedLine:
    """One manuscript line after refinement, before ordering."""

    points: list[list[float]]
    baseline: list[list[float]]
    score: float
    members: tuple[int, ...] = ()
    role: str = "line"
    suspect: bool = False
    suspect_reason: str = ""
    merged_from: int = 1
    overlap_unresolved: bool = False


@dataclass
class _Work:
    """Mutable geometry while merging and cutting."""

    poly: Polygon
    baseline: tuple[tuple[float, float], tuple[float, float]]
    score: float
    members: tuple[int, ...]
    role: str = "line"
    merged_from: int = 1
    overlap_unresolved: bool = False
    dropped: bool = False
    suspect: bool = False
    suspect_reason: str = ""
    area: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = self.poly.area


def _bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _median(values: list[float], fallback: float) -> float:
    if not values:
        return fallback
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _work_of_quad(index: int, quad: DetectedQuad, baseline_fraction: float) -> _Work:
    points = [[float(x), float(y)] for x, y in quad.points]
    baseline = synthetic_baseline_points(points, baseline_fraction)
    return _Work(
        poly=Polygon(points),
        baseline=((baseline[0][0], baseline[0][1]), (baseline[1][0], baseline[1][1])),
        score=float(quad.score),
        members=(index,),
    )


def _bridge(prev: _Work, nxt: _Work) -> Polygon | None:
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
    hull = unary_union(parts).convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty:
        raise ValueError("merge produced no polygon")
    return hull


def _merged_baseline(
    works: list[_Work], baseline_fraction: float, quad_points: list[list[list[float]]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Baseline from the outer ends of the outer members' baselines."""
    left = min(works, key=lambda work: (work.poly.bounds[0], work.poly.bounds[1]))
    right = max(works, key=lambda work: (work.poly.bounds[2], work.poly.bounds[1]))
    left_base = synthetic_baseline_points(quad_points[left.members[0]], baseline_fraction)
    right_base = synthetic_baseline_points(quad_points[right.members[0]], baseline_fraction)
    start = min(left_base, key=lambda point: point[0])
    end = max(right_base, key=lambda point: point[0])
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def merge_row_fragments(
    quads: list[DetectedQuad],
    layout: PageLayout,
    *,
    gap_ratio: float = DEFAULT_MERGE_GAP_RATIO,
    max_overlap_ratio: float = DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    height_ratio: float = DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    baseline_fraction: float = 0.75,
) -> list[_Work]:
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
    heights = [_bbox(quad.points)[3] - _bbox(quad.points)[1] for quad in quads]
    median_height = _median(heights, 1.0) if heights else 1.0
    quad_points = [[[float(x), float(y)] for x, y in quad.points] for quad in quads]
    builder_heights = [
        _bbox(quads[i].points)[3] - _bbox(quads[i].points)[1]
        for column in layout.columns
        for i in column.builders
    ]
    builder_median = _median(builder_heights, median_height)

    in_column = {i for column in layout.columns for i in column.members}
    works: list[_Work] = []
    for column in layout.columns:
        for row in column.rows:
            row_heights = {i: heights[i] for i in row}
            ordered = sorted(
                row,
                key=lambda i: (
                    _bbox(quads[i].points)[0],
                    _bbox(quads[i].points)[1],
                    tuple(round(c, 6) for point in quads[i].points for c in point),
                ),
            )
            run: list[int] = []
            run_max_h = 0.0
            run_min_h = 0.0
            run_xmax = 0.0

            def close_run(run: list[int], row_heights: dict[int, float]) -> None:
                if len(run) > 1:
                    works.append(_merge_run(run, quads, quad_points, baseline_fraction))
                elif run:
                    index = run[0]
                    work = _work_of_quad(index, quads[index], baseline_fraction)
                    mates = [h for j, h in row_heights.items() if j != index]
                    reference = _median(mates, builder_median)
                    if reference > 0 and heights[index] > height_ratio * reference:
                        work.role = "initial"
                    works.append(work)

            for index in ordered:
                xmin, ymin, xmax, ymax = _bbox(quads[index].points)
                height = ymax - ymin
                if not run:
                    run, run_max_h, run_min_h, run_xmax = [index], height, height, xmax
                    continue
                gap = xmin - run_xmax
                new_max = max(run_max_h, height)
                new_min = min(run_min_h, height)
                if (
                    gap <= gap_ratio * median_height
                    and gap >= -max_overlap_ratio * median_height
                    and new_max <= height_ratio * new_min
                ):
                    run.append(index)
                    run_max_h, run_min_h, run_xmax = new_max, new_min, max(run_xmax, xmax)
                else:
                    close_run(run, row_heights)
                    run, run_max_h, run_min_h, run_xmax = [index], height, height, xmax
            close_run(run, row_heights)
    spanning = sorted(set(range(len(quads))) - in_column)
    for index in spanning:
        works.append(_work_of_quad(index, quads[index], baseline_fraction))
    return works


def _merge_run(
    run: list[int],
    quads: list[DetectedQuad],
    quad_points: list[list[list[float]]],
    baseline_fraction: float,
) -> _Work:
    """Union one x-run of row mates plus gap bridges into a single line."""
    ordered = sorted(run, key=lambda i: _bbox(quads[i].points)[0])
    parts: list[Polygon] = [Polygon(quad_points[i]) for i in ordered]
    members = [_work_of_quad(i, quads[i], baseline_fraction) for i in ordered]
    for prev, nxt in zip(members, members[1:], strict=False):
        bridge = _bridge(prev, nxt)
        if bridge is not None:
            parts.append(bridge)
    merged = _merged_polygon(parts)
    ring = [list(point) for point in merged.exterior.coords]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    total_area = sum(work.area for work in members)
    score = sum(work.score * work.area for work in members) / total_area if total_area > 0 else 0.0
    baseline = _merged_baseline(members, baseline_fraction, quad_points)
    return _Work(
        poly=merged,
        baseline=baseline,
        score=score,
        members=tuple(ordered),
        merged_from=len(ordered),
    )


def _shared_ratio(left: _Work, right: _Work) -> float:
    smaller = min(left.area, right.area)
    if smaller <= 0:
        return 0.0
    return left.poly.intersection(right.poly).area / smaller


def _shared_over_larger(left: _Work, right: _Work) -> float:
    larger = max(left.area, right.area)
    if larger <= 0:
        return 0.0
    return left.poly.intersection(right.poly).area / larger


def _baseline_distance(left: _Work, right: _Work) -> float:
    """Perpendicular distance between the two baseline midpoints."""
    first = np.asarray(left.baseline, dtype=float)
    second = np.asarray(right.baseline, dtype=float)
    first_dir = first[1] - first[0]
    second_dir = second[1] - second[0]
    if float(np.dot(first_dir, second_dir)) < 0:
        second_dir = -second_dir
    mean = first_dir + second_dir
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return float(np.linalg.norm((first.mean(axis=0) + second.mean(axis=0)) / 2))
    normal = np.array([-mean[1], mean[0]]) / norm
    return float(abs(np.dot(second.mean(axis=0) - first.mean(axis=0), normal)))


def _median_line_spacing(layout: PageLayout, fallback: float) -> float:
    """Median vertical pitch between consecutive row groups in a column."""
    gaps: list[float] = []
    for column in layout.columns:
        centres = sorted(
            sum(layout.centre_y[i] for i in row) / len(row) for row in column.rows if row
        )
        gaps.extend(b - a for a, b in zip(centres, centres[1:], strict=False) if b > a)
    return _median(gaps, fallback)


def _clip_baseline(
    baseline: tuple[tuple[float, float], tuple[float, float]], kept: Polygon
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a baseline segment to the kept half after a cut."""
    line = LineString([baseline[0], baseline[1]])
    clipped = line.intersection(kept)
    if clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        coords = list(clipped.coords)
        return (tuple(coords[0]), tuple(coords[-1]))
    longest = None
    longest_length = 0.0
    for part in getattr(clipped, "geoms", []):
        if isinstance(part, LineString) and part.length > longest_length:
            longest, longest_length = part, part.length
    if longest is None:
        return None
    coords = list(longest.coords)
    return (tuple(coords[0]), tuple(coords[-1]))


def _cut_pair(left: _Work, right: _Work) -> bool:
    """Clip a stacked pair apart at the mid-baseline line. True when cut."""
    first = np.asarray(left.baseline, dtype=float)
    second = np.asarray(right.baseline, dtype=float)
    first_dir = first[1] - first[0]
    second_dir = second[1] - second[0]
    if float(np.dot(first_dir, second_dir)) < 0:
        second_dir = -second_dir
    mean = first_dir + second_dir
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return False
    direction = mean / norm
    normal = np.array([-direction[1], direction[0]])
    mid = (first.mean(axis=0) + second.mean(axis=0)) / 2
    arm, reach = direction * 1e5, normal * 1e5
    centre = mid.tolist()
    left_sign = float(np.dot(np.asarray(left.poly.centroid.coords[0]) - mid, normal))
    left_sign = 1.0 if left_sign >= 0 else -1.0
    kept: list[Polygon] = []
    for work, sign in ((left, left_sign), (right, -left_sign)):
        corners = [
            (centre - arm + sign * reach).tolist(),
            (centre + arm + sign * reach).tolist(),
            (centre + arm).tolist(),
            (centre - arm).tolist(),
        ]
        half = Polygon(corners)
        piece = work.poly.intersection(half)
        if piece.is_empty or piece.area <= 0.5 * work.area:
            return False
        if not isinstance(piece, Polygon):
            return False
        baseline = _clip_baseline(work.baseline, piece)
        if baseline is None:
            return False
        kept.append((piece, baseline))
    old_areas = (left.area, right.area)
    for work, (piece, baseline) in zip((left, right), kept, strict=True):
        work.poly = piece
        work.baseline = baseline
        work.area = piece.area
    return abs(left.area - old_areas[0]) > 1e-9 or abs(right.area - old_areas[1]) > 1e-9


def _candidate_pairs(
    works: list[_Work], live: list[int], cut_threshold: float
) -> list[tuple[int, int]]:
    """Index pairs ``a < b`` of ``live`` whose bounding boxes touch, sorted.

    Shared area is positive only when the bounding boxes intersect, so a
    pair outside this list has a shared ratio of 0 and can never reach a
    positive ``cut_threshold``. The STRtree turns the all-pairs scan into a
    query per polygon; a non-positive threshold admits every pair, so it
    keeps the full scan.
    """
    if cut_threshold <= 0:
        return [(a, b) for pos, a in enumerate(live) for b in live[pos + 1 :]]
    if len(live) < 2:
        return []
    polys = [works[i].poly for i in live]
    query_index, tree_index = STRtree(polys).query(polys)
    return sorted(
        {
            (live[first], live[second])
            for first, second in zip(query_index.tolist(), tree_index.tolist(), strict=True)
            if first < second
        }
    )


def resolve_overlaps(
    works: list[_Work],
    layout: PageLayout,
    *,
    cut_threshold: float = DEFAULT_OVERLAP_CUT_THRESHOLD,
    median_height: float = 1.0,
) -> list[_Work]:
    """Cut stacked neighbours apart, drop duplicates, flag the rest.

    Pairs sharing at least ``cut_threshold`` of the smaller polygon: stacked
    pairs (baseline distance at least half the page median line spacing) are
    clipped apart at the mid-baseline line until they share zero area, unless
    the cut would remove more than half of either polygon. A pair is a
    duplicate only when the two polygons are nearly the same region: shared
    area divided by the larger polygon's area is above
    ``DUPLICATE_SHARED_RATIO`` (0.50) and the baseline distance is below half
    the line spacing; only then is the lower score dropped. A containment
    pair (shared over the smaller above the cut threshold, shared over the
    larger at or below 0.50, baselines coinciding) is never dropped: both are
    marked ``overlap_unresolved`` and left. A work with ``role == "initial"``
    is never dropped, whatever the ratios: the pair is marked unresolved
    instead. Anything else is marked ``overlap_unresolved`` and left.
    """
    spacing = _median_line_spacing(layout, median_height)
    for _ in range(_MAX_OVERLAP_PASSES):
        pairs: list[tuple[float, int, int]] = []
        live = [i for i, work in enumerate(works) if not work.dropped]
        for a, b in _candidate_pairs(works, live, cut_threshold):
            ratio = _shared_ratio(works[a], works[b])
            if ratio >= cut_threshold:
                pairs.append((-ratio, a, b))
        pairs.sort()
        changed = False
        for _, a, b in pairs:
            left, right = works[a], works[b]
            if left.dropped or right.dropped:
                continue
            ratio = _shared_ratio(left, right)
            if ratio < cut_threshold:
                continue
            distance = _baseline_distance(left, right)
            if distance < spacing / 2:
                if left.role == "initial" or right.role == "initial":
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
                elif _shared_over_larger(left, right) > DUPLICATE_SHARED_RATIO:
                    if (right.score, right.members) > (left.score, left.members):
                        left.dropped = True
                    else:
                        right.dropped = True
                    changed = True
                else:
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
            elif distance >= spacing / 2:
                if _cut_pair(left, right):
                    changed = True
                else:
                    left.overlap_unresolved = True
                    right.overlap_unresolved = True
            else:
                left.overlap_unresolved = True
                right.overlap_unresolved = True
        if not changed:
            break
    live = [i for i, work in enumerate(works) if not work.dropped]
    for a, b in _candidate_pairs(works, live, cut_threshold):
        left, right = works[a], works[b]
        if _shared_ratio(left, right) >= cut_threshold:
            left.overlap_unresolved = True
            right.overlap_unresolved = True
    return [work for work in works if not work.dropped]


def _builder_bands(layout: PageLayout, quads: list[DetectedQuad]) -> list[tuple[float, float]]:
    """Text bands: x ranges over each column's founding builders."""
    bands: list[tuple[float, float]] = []
    for column in layout.columns:
        if not column.builders:
            continue
        xs = [point[0] for i in column.builders for point in quads[i].points]
        bands.append((min(xs), max(xs)))
    return sorted(bands)


def _text_block(layout: PageLayout, quads: list[DetectedQuad]) -> tuple[float, float] | None:
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


def classify_suspects(
    works: list[_Work],
    layout: PageLayout,
    quads: list[DetectedQuad],
) -> None:
    """Flag non-text detections in place, never deleting.

    A singleton that is not an initial is a suspect when it lies outside
    every builder column band, unless it sits strictly between two bands
    inside the text block top to bottom (a gutter numeral between two
    columns keeps its place and is never a suspect by angle either). Any
    other non-initial whose bounding-box angle is more than
    ``SUSPECT_ANGLE_DEGREES`` off the page dominant angle is a suspect
    whatever its position. Merged lines always have a column home.
    """
    bands = _builder_bands(layout, quads)
    block = _text_block(layout, quads)
    angles = [
        _bbox_angle([[float(x), float(y)] for x, y in work.poly.exterior.coords]) for work in works
    ]
    dominant = _median(angles, 0.0)
    pairs = list(zip(bands, bands[1:], strict=False))
    for position, work in enumerate(works):
        if work.role == "initial":
            continue
        xmin, ymin, xmax, ymax = work.poly.bounds
        centre_y = (ymin + ymax) / 2.0
        in_gutter = (
            block is not None
            and block[0] <= centre_y <= block[1]
            and any(
                xmin > left_high and xmax < right_low for (_, left_high), (right_low, _) in pairs
            )
        )
        if in_gutter:
            continue
        outside = not any(
            xmin >= low - BAND_TOLERANCE_PX and xmax <= high + BAND_TOLERANCE_PX
            for low, high in bands
        )
        if outside and work.merged_from == 1:
            work.suspect = True
            work.suspect_reason = "outside_bands"
            continue
        if abs(angles[position] - dominant) > SUSPECT_ANGLE_DEGREES:
            work.suspect = True
            work.suspect_reason = "angle"


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
    heights = [_bbox(quad.points)[3] - _bbox(quad.points)[1] for quad in quads]
    median_height = _median(heights, 1.0)
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
        works = [_work_of_quad(i, quad, baseline_fraction) for i, quad in enumerate(quads)]
        _mark_initials(works, quads, layout, merge_max_height_ratio)
    if resolve:
        works = resolve_overlaps(
            works, layout, cut_threshold=overlap_cut_threshold, median_height=median_height
        )
    if classify:
        classify_suspects(works, layout, quads)
    lines = []
    for work in works:
        ring = [list(point) for point in work.poly.exterior.coords]
        if len(ring) >= 2 and ring[0] == ring[-1]:
            ring = ring[:-1]
        lines.append(
            RefinedLine(
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
        )
    return lines


def _mark_initials(
    works: list[_Work],
    quads: list[DetectedQuad],
    layout: PageLayout,
    height_ratio: float,
) -> None:
    """Flag tall singletons as initials when merging is switched off."""
    heights = {i: _bbox(quad.points)[3] - _bbox(quad.points)[1] for i, quad in enumerate(quads)}
    builder_heights = [heights[i] for column in layout.columns for i in column.builders]
    reference_all = _median(builder_heights, 1.0)
    for column in layout.columns:
        for row in column.rows:
            for index in row:
                mates = [heights[j] for j in row if j != index]
                reference = _median(mates, reference_all)
                if reference > 0 and heights[index] > height_ratio * reference:
                    works[index].role = "initial"


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
