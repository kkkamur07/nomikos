"""Page-level measurements: column bands, medians and line spacing.

The columns are read off a coverage profile rather than configured, because
the same document can hold one-column and two-column pages and nobody wants to
declare which is which.
"""

from __future__ import annotations

import statistics

from backend.annotation.application.segment_health.geometry import (
    PageStats,
    Segment,
    bbox,
    height_of,
    width_of,
)

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
    coverage = _coverage_profile(spans, slice_width)
    peak = max(coverage)
    if peak == 0:
        return []
    bands = _covered_runs(spans, coverage, slice_width, COLUMN_COVERAGE_RATIO * peak)
    return [band for band in bands if band[1] - band[0] >= BAND_MIN_WIDTH_FRACTION * page_width]


def _coverage_profile(spans: list[tuple[float, float]], slice_width: float) -> list[int]:
    """How many text-width segments cover each slice of the page."""
    coverage = [0] * COVERAGE_SLICES
    for low, high in spans:
        first = max(0, int(low / slice_width))
        last = min(COVERAGE_SLICES - 1, int(high / slice_width))
        for index in range(first, last + 1):
            coverage[index] += 1
    return coverage


def _covered_runs(
    spans: list[tuple[float, float]],
    coverage: list[int],
    slice_width: float,
    floor: float,
) -> list[tuple[float, float]]:
    """Runs of slices at or above the coverage floor, snapped to segments."""
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
    return bands


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


def _baseline_mean_y(baseline: list[list[float]]) -> float:
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
