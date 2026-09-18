"""Find the three mistakes kraken makes on a page, and build the fix for each.

Everything here is pure geometry: no database, no shapely, no numpy. That is a
deliberate constraint rather than an accident. The platform API ships to Vercel
from the `platform-prod` dependency group, which carries neither shapely nor
numpy, and adding GEOS to a serverless bundle to cut a polygon at a vertical
line would be a poor trade. The two operations actually needed are a half-plane
clip and a join of two pieces of one text line, and both are exact in a hundred
lines of arithmetic.

The other constraint is that every threshold is relative to the page it is
measured on. The same rule in absolute pixels was tried across Syriac, Greek
and two Armenian manuscripts and did not transfer: what counts as a narrow
segment on a 2479px scan is not what counts on a photographed folio. So widths
and heights are compared against that page's own medians, and horizontal
position is compared against column bands derived from that page's own layout.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

Point = list[float]
Polygon = list[Point]
# (min x, max x, min y, max y), the order the reference scripts used.
Box = tuple[float, float, float, float]

# A segment has to be this much wider than the page before it is taken as
# evidence of where a column runs. Narrow things are what we are trying to
# classify, so they cannot also be what defines the layout.
BAND_MIN_WIDTH_FRACTION = 0.15
# How finely the page is sliced when reading columns off the coverage profile.
# 200 slices is about 12px on a 2479px scan, comfortably narrower than a gutter.
COVERAGE_SLICES = 200
# A slice belongs to a column when this many of the page's text-width segments
# cover it, as a fraction of the busiest slice. The reference gutters sit near
# 25% of peak and the columns at 100%, so a floor between the two separates
# them without needing to know how many columns the page has.
COLUMN_COVERAGE_RATIO = 0.35
# Narrow enough to be a candidate for noise. On its own this catches real
# gutter numerals too, which is exactly why it is never the whole rule.
NARROW_WIDTH_FRACTION = 0.06
# A segment with no more corners than this, and squat, is a sliver: a dot shaken
# off an initial, a speck of dirt read as ink.
TINY_MAX_POINTS = 6
TINY_HEIGHT_RATIO = 0.6
# Two baselines belong to one text line if they sit this close, measured
# against the spacing between lines on that page rather than in pixels.
COLLINEAR_SPACING_RATIO = 0.4
# ... and if the horizontal gap between them is smaller than this, measured
# against the height of the line itself.
ADJACENT_GAP_HEIGHT_RATIO = 1.5
# A fragment is the smaller piece by a wide margin. Two halves of similar size
# are more likely two real lines that kraken split correctly.
FRAGMENT_MAX_WIDTH_RATIO = 0.6
# A spanning segment has to reach this far into each band before the band counts
# as covered, so that a polygon merely leaning into its neighbour is left alone.
BAND_COVERAGE_RATIO = 0.2

# Two line masks that share this much of the smaller one are a defect, not a
# pair of neighbours whose descenders reach. Measured across 1617 overlapping
# pairs in two manuscripts: 1388 share under 5%, which is ordinary, and the
# distribution thins to nothing between 10% and 20%. Nineteen pairs sit above.
OVERLAP_DEFECT_RATIO = 0.2
# Past this the two are not neighbours bleeding into each other, they are one
# line drawn twice, and cutting between their baselines halves a duplicate
# rather than separating two lines. Those are reported without a trim.
OVERLAP_DUPLICATE_RATIO = 0.5


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
    # True when something downstream depends on this row surviving: a human
    # typed text against it, or the pairing points at it.
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
    # Plain-language reasons, shown to whoever decides. A flag nobody can check
    # is a flag nobody should act on.
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
    #: Shared area as a fraction of the smaller of the two.
    ratio: float
    #: Midway between the two baselines, where the trim would cut.
    cut: float
    #: The trimmed outlines, empty when no trim is offered.
    upper_points: Polygon = field(default_factory=list)
    lower_points: Polygon = field(default_factory=list)
    #: What each would lose, so the cost is on screen before anyone agrees.
    upper_loss: float = 0.0
    lower_loss: float = 0.0
    #: Set when the two are one line drawn twice rather than two that bleed.
    duplicate: bool = False


def _median(values: list[float], fallback: float) -> float:
    return statistics.median(values) if values else fallback


def column_bands(segments: list[Segment], page_width: float) -> list[tuple[float, float]]:
    """The x-ranges the page's columns actually occupy.

    Derived from the page rather than configured, because the same document can
    hold one-column and two-column pages and nobody wants to declare which is
    which. Only segments wide enough to be running text vote, so the numerals
    and specks we are about to classify do not get to define the layout that
    classifies them.

    The columns are read off a coverage profile: how many text-width segments
    cover each slice of the page. A gutter is a deep, narrow dip in that
    profile, and on the reference pages it is unmissable, coverage falling from
    24 segments to 6 in a single slice.

    Merging overlapping x-ranges instead would be simpler and wrong. The
    segments that span two columns are precisely the ones this module exists to
    find, and each of them bridges the gutter; a few of those are enough to
    weld the two columns into one band, at which point the spanning segments
    are inside a single column and invisible. Counting coverage rather than
    merging extents keeps a handful of bad segments from outvoting a hundred
    good ones.
    """
    spans = [
        (bbox(segment.points)[0], bbox(segment.points)[1])
        for segment in segments
        if segment.points and width_of(segment.points) >= BAND_MIN_WIDTH_FRACTION * page_width
    ]
    if not spans or page_width <= 0:
        return []
    slice_width = page_width / COVERAGE_SLICES
    coverage = [0] * COVERAGE_SLICES
    for low, high in spans:
        first = max(0, int(low / slice_width))
        last = min(COVERAGE_SLICES - 1, int(high / slice_width))
        for index in range(first, last + 1):
            coverage[index] += 1
    peak = max(coverage)
    if peak == 0:
        return []
    floor = COLUMN_COVERAGE_RATIO * peak
    bands: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, value in enumerate(coverage):
        if value >= floor and run_start is None:
            run_start = index
        elif value < floor and run_start is not None:
            bands.append(_band_extent(spans, run_start, index - 1, slice_width))
            run_start = None
    if run_start is not None:
        bands.append(_band_extent(spans, run_start, COVERAGE_SLICES - 1, slice_width))
    return [band for band in bands if band[1] - band[0] >= BAND_MIN_WIDTH_FRACTION * page_width]


def _band_extent(
    spans: list[tuple[float, float]], first_slice: int, last_slice: int, slice_width: float
) -> tuple[float, float]:
    """Snap a run of covered slices to the segments that actually fill it.

    The slice boundaries are only as fine as the profile, so the band is
    tightened onto the segments lying wholly inside it. Segments that overrun
    the run are ignored here for the same reason they are downweighted above:
    a segment crossing the gutter would drag the band across it.
    """
    low, high = first_slice * slice_width, (last_slice + 1) * slice_width
    inside = [
        span for span in spans if span[0] >= low - slice_width and span[1] <= high + slice_width
    ]
    if not inside:
        return (low, high)
    return (min(span[0] for span in inside), max(span[1] for span in inside))


def _baseline_mean_y(baseline: Polygon) -> float:
    return statistics.mean(point[1] for point in baseline)


def line_spacing(segments: list[Segment], bands: list[tuple[float, float]]) -> float:
    """Median gap between one text line and the next, measured within a column.

    Measured per band, because on a two-column page the baselines of the left
    and right columns interleave in y, and mixing them would report half the
    real spacing and make every threshold built on it too tight.
    """
    gaps: list[float] = []
    for low, high in bands:
        ys = sorted(
            _baseline_mean_y(segment.baseline)
            for segment in segments
            if len(segment.baseline) >= 2 and _band_index(segment, [(low, high)]) == 0
        )
        gaps.extend(
            second - first for first, second in zip(ys, ys[1:], strict=False) if second - first > 0
        )
    if not gaps:
        return 0.0
    return statistics.median(gaps)


def page_stats(segments: list[Segment], page_width: float, page_height: float) -> PageStats:
    usable = [segment for segment in segments if len(segment.points) >= 3]
    bands = column_bands(usable, page_width)
    widths = [width_of(segment.points) for segment in usable]
    heights = [height_of(segment.points) for segment in usable]
    return PageStats(
        width=page_width,
        height=page_height,
        bands=bands,
        median_width=_median(widths, 0.0),
        median_height=_median(heights, 0.0),
        line_spacing=line_spacing(usable, bands),
    )


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _band_index(segment: Segment, bands: list[tuple[float, float]]) -> int | None:
    """Which band this segment sits in, or None when it sits between them."""
    if not segment.points:
        return None
    box = bbox(segment.points)
    span = (box[0], box[1])
    best: int | None = None
    best_overlap = 0.0
    for index, band in enumerate(bands):
        overlap = _overlap(span, band)
        if overlap > best_overlap:
            best, best_overlap = index, overlap
    if best is None:
        return None
    # More than half of the segment has to be inside before it counts as living
    # there; a numeral clipping the edge of a column is still in the gutter.
    return best if best_overlap >= 0.5 * (span[1] - span[0]) else None


def find_suspects(segments: list[Segment], stats: PageStats) -> list[Suspect]:
    """Segments that look like noise, flagged and never deleted.

    The rule is a conjunction on purpose. Width alone was measured on four
    manuscripts and fails on all of them: on one it deletes 46 real gutter
    numerals a page, on another the noise is not narrow at all but squat and
    four-cornered, sitting inside the columns. What survived every page was
    narrow AND (living outside every column OR being barely a polygon).

    Nothing here decides to delete. Some of what this flags is real ink that
    nobody has transcribed yet, and no amount of geometry can tell that from
    a smudge. A human can, in one look, which is what the flag is for.
    """
    found: list[Suspect] = []
    for segment in segments:
        if segment.manual_geometry or len(segment.points) < 3:
            continue
        if segment.has_text or segment.is_paired:
            continue
        width = width_of(segment.points)
        if width >= NARROW_WIDTH_FRACTION * stats.width:
            continue
        reasons = [f"narrower than {NARROW_WIDTH_FRACTION:.0%} of the page"]
        outside = stats.bands and _band_index(segment, stats.bands) is None
        tiny = len(segment.points) <= TINY_MAX_POINTS and (
            stats.median_height > 0
            and height_of(segment.points) < TINY_HEIGHT_RATIO * stats.median_height
        )
        if outside:
            reasons.append("outside every column")
        if tiny:
            reasons.append(
                f"only {len(segment.points)} corners and shorter than "
                f"{TINY_HEIGHT_RATIO:.0%} of a typical line"
            )
        if outside or tiny:
            found.append(Suspect(line_id=segment.id, reasons=reasons))
    return found


def y_at(polyline: Polygon, x: float) -> float:
    """Height of a polyline at an x, clamped to its ends."""
    ordered = sorted(polyline)
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


def find_spanning(segments: list[Segment], stats: PageStats) -> list[SpanningSplit]:
    """Segments that swallowed two columns and the gutter between them.

    Kraken emits one block for the whole page, so nothing stops a line box from
    running left column, gutter, right column in a single polygon. Every one of
    these found in the reference data crossed the gutter, which is what makes
    the cut well defined: the midpoint of the gap between the two bands.
    """
    if len(stats.bands) < 2:
        return []
    found: list[SpanningSplit] = []
    for segment in segments:
        if segment.manual_geometry or len(segment.points) < 3:
            continue
        box = bbox(segment.points)
        span = (box[0], box[1])
        covered = [
            index
            for index, band in enumerate(stats.bands)
            if _overlap(span, band)
            >= BAND_COVERAGE_RATIO * min(band[1] - band[0], span[1] - span[0])
        ]
        if len(covered) < 2:
            continue
        cuts = [
            (stats.bands[first][1] + stats.bands[second][0]) / 2
            for first, second in zip(covered, covered[1:], strict=False)
        ]
        bounds: list[float | None] = [None, *cuts, None]
        pieces: list[tuple[Polygon, Polygon]] = []
        degenerate = False
        for index in range(len(covered)):
            piece = clip_polygon(segment.points, bounds[index], bounds[index + 1])
            baseline = clip_baseline(segment.baseline, bounds[index], bounds[index + 1])
            if len(piece) < 3 or len(baseline) < 2:
                # A cut that produces a sliver is a cut in the wrong place.
                # Better to leave the segment whole and say nothing than to
                # offer a fix that loses part of the line.
                degenerate = True
                break
            pieces.append((piece, baseline))
        if degenerate:
            continue
        found.append(SpanningSplit(line_id=segment.id, cuts=cuts, pieces=pieces))
    return found


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
    low = min(point[0] for point in first + second)
    high = max(point[0] for point in first + second)
    if high <= low:
        return abs(_baseline_mean_y(first) - _baseline_mean_y(second))
    samples = [low + (high - low) * index / 10 for index in range(11)]
    return statistics.mean(abs(y_at(first, x) - y_at(second, x)) for x in samples)


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


def convex_hull(points: Polygon) -> Polygon:
    """The smallest convex ring around a cloud of points, counter-clockwise.

    Andrew's monotone chain, which needs no dependency and no tolerance. This is
    only ever the fallback: an envelope hugs a text line far more tightly, and a
    hull over a curved line swallows whatever sits inside its bulge. It is here
    because the fallback has to stay a *simple* polygon once more than two pieces
    can be handed in, and concatenating the pieces does not.
    """
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return [[round(x, 1), round(y, 1)] for x, y in unique]

    def turn(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    def chain(ordered: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for point in ordered:
            while len(out) >= 2 and turn(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
        return out

    hull = chain(unique)[:-1] + chain(unique[::-1])[:-1]
    return [[round(x, 1), round(y, 1)] for x, y in hull]


def merge_polygon_group(pieces: list[Polygon]) -> Polygon:
    """One outline around any number of pieces of one text line.

    Not a general polygon union, and it does not need to be. The pieces are
    already known to belong together, so the merged outline is the top edge of
    all of them followed by the bottom edge of all of them, walked back. Joining
    the chains bridges the gaps by construction, which is the part that matters:
    the reference implementation had to add an explicit bridging band because a
    buffer round trip on its own silently dropped the fragment.

    Where the pieces touch or overlap this is their union; where they do not it
    is the union plus the columns between them, which is tighter than a hull and
    is what someone merging a broken line is asking for. The hull is the
    fallback for the degenerate case where every point shares one x and there is
    no envelope to build.
    """
    usable = [piece for piece in pieces if len(piece) >= 3]
    if not usable:
        return _dedupe([[round(p[0], 1), round(p[1], 1)] for piece in pieces for p in piece])
    if len(usable) == 1:
        return _dedupe([[round(p[0], 1), round(p[1], 1)] for p in usable[0]])
    upper, lower = _envelope(usable)
    if len(upper) < 2:
        return convex_hull([point for piece in usable for point in piece])
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


def merge_polygons(primary: Polygon, fragment: Polygon) -> Polygon:
    """The two-piece case, which is all the fragment finder ever offers."""
    return merge_polygon_group([primary, fragment])


def _dedupe(points: Polygon) -> Polygon:
    out: Polygon = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def merge_baseline_group(baselines: list[Polygon]) -> Polygon:
    """Every baseline as one, left to right, without near-duplicate points.

    A piece with no baseline of its own contributes nothing rather than a
    placeholder, so merging a line with a stray blot that kraken gave no
    baseline leaves the line's own baseline intact.
    """
    points = sorted(
        [[float(p[0]), float(p[1])] for baseline in baselines for p in baseline if len(p) == 2]
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


def merge_baselines(primary: Polygon, fragment: Polygon) -> Polygon:
    """The two-piece case, which is all the fragment finder ever offers."""
    return merge_baseline_group([primary, fragment])


def find_fragments(segments: list[Segment], stats: PageStats) -> list[FragmentMerge]:
    """Pairs where kraken cut one text line into two.

    A line end, a piece shaken off a decorated initial, a stray dot. The pair
    has to share a baseline, sit side by side with a small gap, and be lopsided:
    two halves of similar width are more likely two real lines that were split
    correctly than one line that was split wrongly.

    The larger piece is the primary and keeps its id, so whatever is already
    attached to it, a transcription or a pairing, survives the merge. A
    fragment that carries text or a pairing of its own is never offered for
    merging, because merging it would destroy that.
    """
    if stats.line_spacing <= 0:
        return []
    usable = [
        segment
        for segment in segments
        if not segment.manual_geometry and len(segment.points) >= 3 and len(segment.baseline) >= 2
    ]
    collinear_limit = COLLINEAR_SPACING_RATIO * stats.line_spacing
    found: list[FragmentMerge] = []
    claimed: set[str] = set()
    for index, first in enumerate(usable):
        for second in usable[index + 1 :]:
            if first.id in claimed or second.id in claimed:
                continue
            if _band_index(first, stats.bands) != _band_index(second, stats.bands):
                continue
            if baseline_distance(first.baseline, second.baseline) >= collinear_limit:
                continue
            first_box, second_box = bbox(first.points), bbox(second.points)
            gap = max(first_box[0], second_box[0]) - min(first_box[1], second_box[1])
            line_height = max(height_of(first.points), height_of(second.points))
            if gap > ADJACENT_GAP_HEIGHT_RATIO * line_height:
                continue
            wide, narrow = sorted(
                (first, second), key=lambda seg: width_of(seg.points), reverse=True
            )
            if width_of(narrow.points) > FRAGMENT_MAX_WIDTH_RATIO * width_of(wide.points):
                continue
            if narrow.has_text or narrow.is_paired:
                continue
            found.append(
                FragmentMerge(
                    primary_id=wide.id,
                    fragment_id=narrow.id,
                    points=merge_polygons(wide.points, narrow.points),
                    baseline=merge_baselines(wide.baseline, narrow.baseline),
                )
            )
            claimed.add(narrow.id)
    return found


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


def _baseline_midline(upper: Segment, lower: Segment) -> float:
    """The y midway between two baselines, across the span they share."""
    upper_box, lower_box = bbox(upper.points), bbox(lower.points)
    low = max(upper_box[0], lower_box[0])
    high = min(upper_box[1], lower_box[1])
    samples = [low, (low + high) / 2, high] if high > low else [(low + high) / 2]
    return statistics.mean((y_at(upper.baseline, x) + y_at(lower.baseline, x)) / 2 for x in samples)


def find_overlaps(segments: list[Segment], stats: PageStats) -> list[OverlapTrim]:
    """Pairs of lines whose masks have grown into each other.

    Neighbouring lines share a little: a descender reaches, an ascender reaches
    back, and across two manuscripts 1388 of 1617 overlapping pairs share under
    5% of the smaller mask. Sharing a fifth of it is something else, and the
    fix is a cut midway between the two baselines, which on all nineteen of the
    heavy pairs separated them completely.

    Offered, never applied. The cut removes a sixth of a mask on average and
    half of one at worst, and the worst case is a pair that is really one line
    drawn twice, where cutting between the baselines is the wrong answer
    entirely. Those are reported as duplicates with no trim attached, and the
    rest carry what each side would lose so that whoever decides can see the
    cost first.

    Known limit. Of the nineteen, fourteen are two stacked lines, three are
    duplicates, and two are a text line that has swallowed a narrow mark
    sitting beside it rather than under it. A cut between the baselines does
    separate those two, at a cost of 13% and 42% of the wider mask, but the
    shape of the mistake is sideways and so the right fix is a sideways clip.
    They are offered the cut with their cost shown, like the rest, because a
    person can see what it would do; they are not silently right.
    """
    del stats  # Read from the pair itself; nothing here is a page-wide rule.
    candidates = [
        segment
        for segment in segments
        if not segment.manual_geometry and len(segment.points) >= 3 and len(segment.baseline) >= 2
    ]
    found: list[OverlapTrim] = []
    for index, first in enumerate(candidates):
        first_box = bbox(first.points)
        first_area = polygon_area(first.points)
        if first_area <= 0:
            continue
        for second in candidates[index + 1 :]:
            second_box = bbox(second.points)
            if (
                first_box[1] <= second_box[0]
                or second_box[1] <= first_box[0]
                or first_box[3] <= second_box[2]
                or second_box[3] <= first_box[2]
            ):
                continue
            second_area = polygon_area(second.points)
            if second_area <= 0:
                continue
            shared = overlap_area(first.points, second.points)
            ratio = shared / min(first_area, second_area)
            if ratio < OVERLAP_DEFECT_RATIO:
                continue
            upper, lower = sorted(
                (first, second), key=lambda s: statistics.mean(p[1] for p in s.baseline)
            )
            cut = _baseline_midline(upper, lower)
            if ratio >= OVERLAP_DUPLICATE_RATIO:
                found.append(
                    OverlapTrim(
                        upper_id=upper.id, lower_id=lower.id, ratio=ratio, cut=cut, duplicate=True
                    )
                )
                continue
            kept_upper = clip_across(upper.points, cut, keep_above=True)
            kept_lower = clip_across(lower.points, cut, keep_above=False)
            if len(kept_upper) < 3 or len(kept_lower) < 3:
                continue
            upper_area = polygon_area(upper.points)
            lower_area = polygon_area(lower.points)
            found.append(
                OverlapTrim(
                    upper_id=upper.id,
                    lower_id=lower.id,
                    ratio=ratio,
                    cut=round(cut, 1),
                    upper_points=_dedupe(kept_upper),
                    lower_points=_dedupe(kept_lower),
                    upper_loss=1 - polygon_area(kept_upper) / upper_area,
                    lower_loss=1 - polygon_area(kept_lower) / lower_area,
                )
            )
    return found
