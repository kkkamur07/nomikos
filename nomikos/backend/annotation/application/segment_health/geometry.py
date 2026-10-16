"""Polygon arithmetic for line segments: boxes, clips, joins and overlaps.

Everything here is pure geometry over point lists: no database, no shapely, no
numpy. That is a deliberate constraint rather than an accident. The platform
API ships to Vercel from the `platform-prod` dependency group, which carries
neither shapely nor numpy, and adding GEOS to a serverless bundle to cut a
polygon at a vertical line would be a poor trade. The two operations actually
needed are a half-plane clip and a join of two pieces of one text line, and
both are exact in a hundred lines of arithmetic.

This module also holds the line model the other segment-health modules share:
:class:`Segment`, :class:`PageStats` and the three finding types.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

Point = list[float]
Polygon = list[Point]
# (min x, max x, min y, max y), the order the reference scripts used.
Box = tuple[float, float, float, float]


def bbox(points: Polygon) -> Box:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), max(xs), min(ys), max(ys))


def width_of(points: Polygon) -> float:
    box = bbox(points)
    return box[1] - box[0]


def height_of(points: Polygon) -> float:
    box = bbox(points)
    return box[3] - box[2]


@dataclass(frozen=True)
class Segment:
    """One line as this module needs it, detached from the ORM."""

    id: str
    points: Polygon
    baseline: Polygon
    manual_geometry: bool = False
    has_text: bool = False
    is_paired: bool = False


@dataclass(frozen=True)
class PageStats:
    """What a page says about itself, before any segment is judged."""

    width: float
    height: float
    bands: list[tuple[float, float]]
    median_width: float
    median_height: float
    line_spacing: float


@dataclass(frozen=True)
class Suspect:
    line_id: str
    reasons: list[str]


@dataclass(frozen=True)
class SpanningSplit:
    line_id: str
    cuts: list[float]
    pieces: list[tuple[Polygon, Polygon]] = field(default_factory=list)


@dataclass(frozen=True)
class FragmentMerge:
    primary_id: str
    fragment_id: str
    points: Polygon
    baseline: Polygon


@dataclass(frozen=True)
class OverlapTrim:
    """Two line masks sharing enough area that one of them is wrong."""

    upper_id: str
    lower_id: str
    ratio: float
    cut: float
    upper_points: Polygon = field(default_factory=list)
    lower_points: Polygon = field(default_factory=list)
    upper_loss: float = 0.0
    lower_loss: float = 0.0
    duplicate: bool = False


def y_at(polyline: Polygon, x: float) -> float:
    """Height of a polyline at an x, clamped to its ends."""
    return _y_at_ordered(sorted(polyline), x)


def _y_at_ordered(ordered: Polygon, x: float) -> float:
    if x <= ordered[0][0]:
        return ordered[0][1]
    if x >= ordered[-1][0]:
        return ordered[-1][1]
    for (ax, ay), (bx, by) in zip(ordered, ordered[1:], strict=False):
        if ax <= x <= bx:
            return ay if bx == ax else ay + (by - ay) * (x - ax) / (bx - ax)
    return ordered[-1][1]


def _intersect_at(first: Point, second: Point, cut: float, axis: int) -> Point:
    other = 1 - axis
    span = second[axis] - first[axis]
    t = (cut - first[axis]) / span if span else 0.0
    crossing = first[other] + t * (second[other] - first[other])
    return [round(cut, 1), round(crossing, 1)] if axis == 0 else [round(crossing, 1), round(cut, 1)]


def _clip_half(polygon: Polygon, cut: float, *, axis: int, keep_lower: bool) -> Polygon:
    """Sutherland-Hodgman against one axis-aligned line.

    A half-plane is convex, which is what this algorithm needs of its clip
    region; the polygon being clipped may be as concave as kraken likes.
    """
    inside = (
        (lambda p: p[axis] <= cut) if keep_lower else (lambda p: p[axis] >= cut)  # noqa: E731
    )
    out: Polygon = []
    for index in range(len(polygon)):
        current, previous = polygon[index], polygon[index - 1]
        if inside(current):
            if not inside(previous):
                out.append(_intersect_at(previous, current, cut, axis))
            out.append([float(current[0]), float(current[1])])
        elif inside(previous):
            out.append(_intersect_at(previous, current, cut, axis))
    return out


def clip_half(polygon: Polygon, x_cut: float, *, keep_left: bool) -> Polygon:
    """Everything to one side of the vertical line x = x_cut."""
    return _clip_half(polygon, x_cut, axis=0, keep_lower=keep_left)


def clip_across(polygon: Polygon, y_cut: float, *, keep_above: bool) -> Polygon:
    """Everything to one side of the horizontal line y = y_cut."""
    return _clip_half(polygon, y_cut, axis=1, keep_lower=keep_above)


def clip_polygon(polygon: Polygon, low: float | None, high: float | None) -> Polygon:
    clipped = [[float(p[0]), float(p[1])] for p in polygon]
    if low is not None:
        clipped = clip_half(clipped, low, keep_left=False)
    if high is not None and clipped:
        clipped = clip_half(clipped, high, keep_left=True)
    return clipped


def clip_baseline(baseline: Polygon, low: float | None, high: float | None) -> Polygon:
    if len(baseline) < 2:
        return []
    ordered = sorted([[float(p[0]), float(p[1])] for p in baseline])
    low = ordered[0][0] if low is None else max(low, ordered[0][0])
    high = ordered[-1][0] if high is None else min(high, ordered[-1][0])
    if high - low < 10:
        return []
    inner = [point for point in ordered if low < point[0] < high]
    return (
        [[round(low, 1), round(y_at(ordered, low), 1)]]
        + inner
        + [[round(high, 1), round(y_at(ordered, high), 1)]]
    )


def _baseline_mean_y(baseline: Polygon) -> float:
    return statistics.mean(point[1] for point in baseline)


def baseline_distance(first: Polygon, second: Polygon) -> float:
    """How far apart two baselines run vertically, across the x they share.

    Vertical offset at a common x, not distance between the polylines. The
    difference matters: two halves of one text line are separated mostly
    horizontally, by the gap where the missing text sits, so a straight
    point-to-polyline distance measures that gap and reports two pieces of the
    same line as far apart. What actually distinguishes one line from the next
    is how much higher or lower it runs, which is what this measures.

    Where the two do not overlap in x, each baseline is extended flat from its
    nearest end, so a fragment beyond the end of its line is compared against
    where that line was still going.
    """
    if len(first) < 2 or len(second) < 2:
        return math.inf
    # Each baseline is sorted once here rather than inside every sample: the
    # order is the same eleven times, and sorting it eleven times is what made
    # the fragment finder the slowest of the four.
    ordered_first = sorted(first)
    ordered_second = sorted(second)
    low = min(ordered_first[0][0], ordered_second[0][0])
    high = max(ordered_first[-1][0], ordered_second[-1][0])
    if high <= low:
        return abs(_baseline_mean_y(first) - _baseline_mean_y(second))
    samples = [low + (high - low) * index / 10 for index in range(11)]
    return statistics.mean(
        abs(_y_at_ordered(ordered_first, x) - _y_at_ordered(ordered_second, x)) for x in samples
    )


def _extent_at(polygon: Polygon, x: float) -> tuple[float, float] | None:
    """The polygon's top and bottom at a given x, or None if it is not there.

    Found by intersecting the vertical line at x with every edge, which is the
    only way that works on a real segment polygon. Bucketing vertices by their x
    and taking the highest and lowest at each one looks equivalent and is not:
    kraken puts its top and bottom vertices at different x, so almost every x
    carries exactly one vertex, the top and bottom collapse onto the same
    polyline, and the result is a zigzag rather than an outline. That version
    passed a test built from rectangles, where the top and bottom vertices do
    share their x, and produced 51 self-intersecting rings out of 51 on real
    pages.
    """
    ys: list[float] = []
    count = len(polygon)
    for index in range(count):
        ax, ay = polygon[index]
        bx, by = polygon[(index + 1) % count]
        if ax == bx:
            if abs(ax - x) <= 1e-9:
                ys.extend((ay, by))
            continue
        low, high = (ax, bx) if ax < bx else (bx, ax)
        if low - 1e-9 <= x <= high + 1e-9:
            ys.append(ay + (by - ay) * (x - ax) / (bx - ax))
    if not ys:
        return None
    return (min(ys), max(ys))


def _envelope(pieces: list[Polygon]) -> tuple[Polygon, Polygon]:
    """The top and bottom edges of everything at once, at each vertex column.

    Sampled at the x of every vertex in every piece, taking the highest top and
    the lowest bottom of whichever pieces reach that column. Two pieces side by
    side, overlapping, or one sitting inside the other all come out the same
    way, and no column falls in the gap between two pieces, so the chains cross
    it as a straight run.
    """
    xs = sorted({float(point[0]) for piece in pieces for point in piece})
    upper: Polygon = []
    lower: Polygon = []
    for x in xs:
        extents = [extent for extent in (_extent_at(piece, x) for piece in pieces) if extent]
        if not extents:
            continue
        upper.append([x, min(extent[0] for extent in extents)])
        lower.append([x, max(extent[1] for extent in extents)])
    return upper, lower


def merge_polygons(primary: Polygon, fragment: Polygon) -> Polygon:
    """One outline around two pieces of the same text line.

    Not a general polygon union, and it does not need to be. The two inputs are
    already known to sit on one baseline, so the merged outline is the top edge
    of both pieces followed by the bottom edge of both, walked back. Joining the
    chains bridges the gap by construction, which is the part that matters: the
    reference implementation had to add an explicit bridging band because a
    buffer round trip on its own silently dropped the fragment.
    """
    upper, lower = _envelope([primary, fragment])
    if len(upper) < 2:
        return _dedupe([[round(p[0], 1), round(p[1], 1)] for p in primary + fragment])
    # Where the outline is a single point tall the two chains meet at one
    # vertex. At the two outer ends that is the ring's turning point and is
    # walked once, on the way out. Anywhere in between it belongs to neither
    # chain: kept in both it is visited twice and pinches the ring shut against
    # itself, and kept in one the return path starts a column further in and
    # cuts back across the outward one. Dropping it lets the chains run
    # parallel, which is what keeps the ring simple.
    last = len(upper) - 1
    keep = [i for i in range(len(upper)) if i in (0, last) or upper[i] != lower[i]]
    upper = [upper[i] for i in keep]
    lower = [lower[i] for i in keep]
    merged = list(upper)
    merged.extend(
        lower[index] for index in range(len(lower) - 1, -1, -1) if lower[index] != upper[index]
    )
    return _dedupe([[round(p[0], 1), round(p[1], 1)] for p in merged])


def _dedupe(points: Polygon) -> Polygon:
    out: Polygon = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def merge_baselines(primary: Polygon, fragment: Polygon) -> Polygon:
    """Both baselines as one, left to right, without near-duplicate points."""
    points = sorted(
        [[float(p[0]), float(p[1])] for p in primary]
        + [[float(p[0]), float(p[1])] for p in fragment]
    )
    if not points:
        return []
    out = [points[0]]
    for point in points[1:]:
        if abs(point[0] - out[-1][0]) >= 4:
            out.append(point)
    if len(out) < 2:
        out = [points[0], points[-1]]
    return [[round(p[0], 1), round(p[1], 1)] for p in out]


def polygon_area(polygon: Polygon) -> float:
    """Shoelace, unsigned."""
    total = 0.0
    for index in range(len(polygon)):
        ax, ay = polygon[index]
        bx, by = polygon[(index + 1) % len(polygon)]
        total += ax * by - bx * ay
    return abs(total) / 2


def overlap_area(first: Polygon, second: Polygon) -> float:
    """Area the two outlines share.

    Integrated column by column rather than by intersecting the rings: the
    height they share at an x is the overlap of their two vertical extents, and
    that is the same view of a polygon the rest of this module takes. It is
    exact for outlines with one top edge and one bottom edge, which is what
    kraken draws and what everything here already assumes.
    """
    first_box, second_box = bbox(first), bbox(second)
    low = max(first_box[0], second_box[0])
    high = min(first_box[1], second_box[1])
    if high <= low:
        return 0.0
    columns = sorted(
        {low, high}
        | {float(p[0]) for p in first if low < p[0] < high}
        | {float(p[0]) for p in second if low < p[0] < high}
    )
    # The shared height is piecewise linear except where the two swap which one
    # is on top, so each gap is walked in steps rather than trusted as one.
    xs: list[float] = []
    for left, right in zip(columns, columns[1:], strict=False):
        xs.extend(left + (right - left) * step / 4 for step in range(4))
    xs.append(columns[-1])

    def shared(x: float) -> float:
        a, b = _extent_at(first, x), _extent_at(second, x)
        if a is None or b is None:
            return 0.0
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    heights = [shared(x) for x in xs]
    return sum(
        (left_height + right_height) / 2 * (right_x - left_x)
        for left_x, right_x, left_height, right_height in zip(
            xs, xs[1:], heights, heights[1:], strict=False
        )
    )
