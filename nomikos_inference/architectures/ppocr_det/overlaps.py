"""Resolve overlapping refined lines: cut stacked pairs, drop duplicates.

The second refinement stage. Overlap is shared area divided by the smaller
polygon's area: 5 to 7 percent is normal (ascenders, descenders), 20
percent or more is treated as a defect. Candidate pairs come from an
STRtree over the polygons, so a page costs a query per line rather than an
intersection per pair.
"""

from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Polygon

from nomikos_inference.architectures.ppocr_det.geometry import (
    Baseline,
    Work,
    baseline_distance,
    median,
)
from nomikos_inference.architectures.ppocr_det.reading_order import PageLayout

DEFAULT_OVERLAP_CUT_THRESHOLD = 0.20
DUPLICATE_SHARED_RATIO = 0.50
_MAX_OVERLAP_PASSES = 100


def _shared_ratio(left: Work, right: Work) -> float:
    smaller = min(left.area, right.area)
    if smaller <= 0:
        return 0.0
    return left.poly.intersection(right.poly).area / smaller


def _shared_over_larger(left: Work, right: Work) -> float:
    larger = max(left.area, right.area)
    if larger <= 0:
        return 0.0
    return left.poly.intersection(right.poly).area / larger


def _median_line_spacing(layout: PageLayout, fallback: float) -> float:
    """Median vertical pitch between consecutive row groups in a column."""
    gaps: list[float] = []
    for column in layout.columns:
        centres = sorted(
            sum(layout.centre_y[i] for i in row) / len(row) for row in column.rows if row
        )
        gaps.extend(b - a for a, b in zip(centres, centres[1:], strict=False) if b > a)
    return median(gaps, fallback)


def _ends(line: LineString) -> Baseline:
    coords = list(line.coords)
    return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])


def _clip_baseline(baseline: Baseline, kept: Polygon) -> Baseline | None:
    """Clip a baseline segment to the kept half after a cut."""
    clipped = LineString([baseline[0], baseline[1]]).intersection(kept)
    if clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        return _ends(clipped)
    parts = [part for part in getattr(clipped, "geoms", []) if isinstance(part, LineString)]
    longest = None
    for part in parts:
        if part.length > (longest.length if longest is not None else 0.0):
            longest = part
    return None if longest is None else _ends(longest)


def _half_plane_piece(
    work: Work, centre: np.ndarray, arm: np.ndarray, reach: np.ndarray
) -> tuple[Polygon, Baseline] | None:
    """The part of ``work`` on the ``reach`` side of the cut line, if it keeps over half."""
    half = Polygon(
        [
            (centre - arm + reach).tolist(),
            (centre + arm + reach).tolist(),
            (centre + arm).tolist(),
            (centre - arm).tolist(),
        ]
    )
    piece = work.poly.intersection(half)
    if piece.is_empty or piece.area <= 0.5 * work.area or not isinstance(piece, Polygon):
        return None
    baseline = _clip_baseline(work.baseline, piece)
    if baseline is None:
        return None
    return piece, baseline


def _cut_pair(left: Work, right: Work) -> bool:
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
    centre = np.asarray(mid.tolist())
    left_sign = float(np.dot(np.asarray(left.poly.centroid.coords[0]) - mid, normal))
    left_sign = 1.0 if left_sign >= 0 else -1.0
    kept: list[tuple[Polygon, Baseline]] = []
    for work, sign in ((left, left_sign), (right, -left_sign)):
        piece = _half_plane_piece(work, centre, arm, sign * reach)
        if piece is None:
            return False
        kept.append(piece)
    old_areas = (left.area, right.area)
    for work, (poly, baseline) in zip((left, right), kept, strict=True):
        work.poly = poly
        work.baseline = baseline
        work.area = poly.area
    return abs(left.area - old_areas[0]) > 1e-9 or abs(right.area - old_areas[1]) > 1e-9


def _candidate_pairs(
    works: list[Work], live: list[int], cut_threshold: float
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


def _mark_unresolved(left: Work, right: Work) -> None:
    left.overlap_unresolved = True
    right.overlap_unresolved = True


def _resolve_close_pair(left: Work, right: Work) -> bool:
    """Baselines coincide: drop a duplicate, else mark both. True when dropped."""
    if left.role == "initial" or right.role == "initial":
        _mark_unresolved(left, right)
        return False
    if _shared_over_larger(left, right) <= DUPLICATE_SHARED_RATIO:
        _mark_unresolved(left, right)
        return False
    if (right.score, right.members) > (left.score, left.members):
        left.dropped = True
    else:
        right.dropped = True
    return True


def _resolve_pair(left: Work, right: Work, spacing: float, cut_threshold: float) -> bool:
    """Act on one candidate pair; True when the geometry or the live set changed."""
    if left.dropped or right.dropped or _shared_ratio(left, right) < cut_threshold:
        return False
    if baseline_distance(left, right) < spacing / 2:
        return _resolve_close_pair(left, right)
    if _cut_pair(left, right):
        return True
    _mark_unresolved(left, right)
    return False


def _overlap_pass(works: list[Work], spacing: float, cut_threshold: float) -> bool:
    """One pass over the pairs above the threshold, largest share first."""
    live = [i for i, work in enumerate(works) if not work.dropped]
    pairs: list[tuple[float, int, int]] = []
    for a, b in _candidate_pairs(works, live, cut_threshold):
        ratio = _shared_ratio(works[a], works[b])
        if ratio >= cut_threshold:
            pairs.append((-ratio, a, b))
    pairs.sort()
    changed = False
    for _, a, b in pairs:
        if _resolve_pair(works[a], works[b], spacing, cut_threshold):
            changed = True
    return changed


def resolve_overlaps(
    works: list[Work],
    layout: PageLayout,
    *,
    cut_threshold: float = DEFAULT_OVERLAP_CUT_THRESHOLD,
    median_height: float = 1.0,
) -> list[Work]:
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
        if not _overlap_pass(works, spacing, cut_threshold):
            break
    live = [i for i, work in enumerate(works) if not work.dropped]
    for a, b in _candidate_pairs(works, live, cut_threshold):
        if _shared_ratio(works[a], works[b]) >= cut_threshold:
            _mark_unresolved(works[a], works[b])
    return [work for work in works if not work.dropped]


__all__ = [
    "DEFAULT_OVERLAP_CUT_THRESHOLD",
    "DUPLICATE_SHARED_RATIO",
    "resolve_overlaps",
]
