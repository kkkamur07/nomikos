"""The four finders: suspects, spanning splits, fragment merges, overlap trims.

Each finder is a thin loop over a per-pair (or per-segment) predicate, so the
rule that decides one case can be read without the loop that tries every case.
"""

from __future__ import annotations

import statistics

from backend.annotation.application.segment_health.geometry import (
    Box,
    FragmentMerge,
    OverlapTrim,
    PageStats,
    Segment,
    SpanningSplit,
    Suspect,
    _dedupe,
    baseline_distance,
    bbox,
    clip_across,
    clip_baseline,
    clip_polygon,
    height_of,
    merge_baselines,
    merge_polygons,
    overlap_area,
    polygon_area,
    width_of,
    y_at,
)
from backend.annotation.application.segment_health.page_stats import _band_index, _overlap

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
        reasons = _suspect_reasons(segment, stats)
        if reasons is not None:
            found.append(Suspect(line_id=segment.id, reasons=reasons))
    return found


def _suspect_reasons(segment: Segment, stats: PageStats) -> list[str] | None:
    if segment.manual_geometry or len(segment.points) < 3:
        return None
    if segment.has_text or segment.is_paired:
        return None
    if width_of(segment.points) >= NARROW_WIDTH_FRACTION * stats.width:
        return None
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
    return reasons if outside or tiny else None


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
        split = _spanning_split(segment, stats)
        if split is not None:
            found.append(split)
    return found


def _spanning_split(segment: Segment, stats: PageStats) -> SpanningSplit | None:
    if segment.manual_geometry or len(segment.points) < 3:
        return None
    box = bbox(segment.points)
    span = (box[0], box[1])
    covered = [
        index
        for index, band in enumerate(stats.bands)
        if _overlap(span, band) >= BAND_COVERAGE_RATIO * min(band[1] - band[0], span[1] - span[0])
    ]
    if len(covered) < 2:
        return None
    cuts = [
        (stats.bands[first][1] + stats.bands[second][0]) / 2
        for first, second in zip(covered, covered[1:], strict=False)
    ]
    bounds: list[float | None] = [None, *cuts, None]
    pieces: list[tuple[list[list[float]], list[list[float]]]] = []
    for index in range(len(covered)):
        piece = clip_polygon(segment.points, bounds[index], bounds[index + 1])
        baseline = clip_baseline(segment.baseline, bounds[index], bounds[index + 1])
        if len(piece) < 3 or len(baseline) < 2:
            # A cut that produces a sliver is a cut in the wrong place.
            # Better to leave the segment whole and say nothing than to
            # offer a fix that loses part of the line.
            return None
        pieces.append((piece, baseline))
    return SpanningSplit(line_id=segment.id, cuts=cuts, pieces=pieces)


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
    boxes = {segment.id: bbox(segment.points) for segment in usable}
    collinear_limit = COLLINEAR_SPACING_RATIO * stats.line_spacing
    found: list[FragmentMerge] = []
    claimed: set[str] = set()
    for index, first in enumerate(usable):
        for second in usable[index + 1 :]:
            merge = _fragment_pair(first, second, stats, boxes, collinear_limit, claimed)
            if merge is not None:
                found.append(merge)
                claimed.add(merge.fragment_id)
    return found


def _fragment_pair(
    first: Segment,
    second: Segment,
    stats: PageStats,
    boxes: dict[str, Box],
    collinear_limit: float,
    claimed: set[str],
) -> FragmentMerge | None:
    if first.id in claimed or second.id in claimed:
        return None
    if _band_index(first, stats.bands) != _band_index(second, stats.bands):
        return None
    # The bbox gap is nearly free and the baseline distance samples eleven
    # columns through two sorted polylines, so the gap runs first.
    first_box, second_box = boxes[first.id], boxes[second.id]
    gap = max(first_box[0], second_box[0]) - min(first_box[1], second_box[1])
    line_height = max(first_box[3] - first_box[2], second_box[3] - second_box[2])
    if gap > ADJACENT_GAP_HEIGHT_RATIO * line_height:
        return None
    first_width = first_box[1] - first_box[0]
    second_width = second_box[1] - second_box[0]
    wide, narrow = (first, second) if first_width >= second_width else (second, first)
    wide_width = max(first_width, second_width)
    narrow_width = min(first_width, second_width)
    if narrow_width > FRAGMENT_MAX_WIDTH_RATIO * wide_width:
        return None
    if narrow.has_text or narrow.is_paired:
        return None
    if baseline_distance(first.baseline, second.baseline) >= collinear_limit:
        return None
    return FragmentMerge(
        primary_id=wide.id,
        fragment_id=narrow.id,
        points=merge_polygons(wide.points, narrow.points),
        baseline=merge_baselines(wide.baseline, narrow.baseline),
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
    boxes = {segment.id: bbox(segment.points) for segment in candidates}
    areas = {segment.id: polygon_area(segment.points) for segment in candidates}
    found: list[OverlapTrim] = []
    for index, first in enumerate(candidates):
        if areas[first.id] <= 0:
            continue
        for second in candidates[index + 1 :]:
            trim = _overlap_pair(first, second, boxes, areas)
            if trim is not None:
                found.append(trim)
    return found


def _overlap_pair(
    first: Segment,
    second: Segment,
    boxes: dict[str, Box],
    areas: dict[str, float],
) -> OverlapTrim | None:
    first_box, second_box = boxes[first.id], boxes[second.id]
    if (
        first_box[1] <= second_box[0]
        or second_box[1] <= first_box[0]
        or first_box[3] <= second_box[2]
        or second_box[3] <= first_box[2]
    ):
        return None
    if areas[second.id] <= 0:
        return None
    shared = overlap_area(first.points, second.points)
    ratio = shared / min(areas[first.id], areas[second.id])
    if ratio < OVERLAP_DEFECT_RATIO:
        return None
    upper, lower = sorted((first, second), key=lambda s: statistics.mean(p[1] for p in s.baseline))
    cut = _baseline_midline(upper, lower)
    if ratio >= OVERLAP_DUPLICATE_RATIO:
        return OverlapTrim(
            upper_id=upper.id, lower_id=lower.id, ratio=ratio, cut=round(cut, 1), duplicate=True
        )
    kept_upper = clip_across(upper.points, cut, keep_above=True)
    kept_lower = clip_across(lower.points, cut, keep_above=False)
    if len(kept_upper) < 3 or len(kept_lower) < 3:
        return None
    upper_area = polygon_area(upper.points)
    lower_area = polygon_area(lower.points)
    return OverlapTrim(
        upper_id=upper.id,
        lower_id=lower.id,
        ratio=ratio,
        cut=round(cut, 1),
        upper_points=_dedupe(kept_upper),
        lower_points=_dedupe(kept_lower),
        upper_loss=1 - polygon_area(kept_upper) / upper_area,
        lower_loss=1 - polygon_area(kept_lower) / lower_area,
    )
