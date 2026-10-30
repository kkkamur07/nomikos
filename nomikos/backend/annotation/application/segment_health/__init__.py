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

This package keeps the historic ``segment_health`` import path: every public
name lives in :mod:`geometry`, :mod:`page_stats` or :mod:`finders` and is
re-exported here.
"""

from backend.annotation.application.segment_health.finders import (
    ADJACENT_GAP_HEIGHT_RATIO,
    BAND_COVERAGE_RATIO,
    COLLINEAR_SPACING_RATIO,
    FRAGMENT_MAX_WIDTH_RATIO,
    NARROW_WIDTH_FRACTION,
    OVERLAP_DEFECT_RATIO,
    OVERLAP_DUPLICATE_RATIO,
    TINY_HEIGHT_RATIO,
    TINY_MAX_POINTS,
    find_fragments,
    find_overlaps,
    find_spanning,
    find_suspects,
)
from backend.annotation.application.segment_health.geometry import (
    Box,
    FragmentMerge,
    OverlapTrim,
    PageStats,
    Point,
    Polygon,
    Segment,
    SpanningSplit,
    Suspect,
    baseline_distance,
    bbox,
    clip_across,
    clip_baseline,
    clip_half,
    clip_polygon,
    convex_hull,
    height_of,
    merge_baseline_group,
    merge_baselines,
    merge_polygon_group,
    merge_polygons,
    overlap_area,
    polygon_area,
    width_of,
    y_at,
)
from backend.annotation.application.segment_health.page_stats import (
    BAND_MIN_WIDTH_FRACTION,
    COLUMN_COVERAGE_RATIO,
    COVERAGE_SLICES,
    column_bands,
    line_spacing,
    page_stats,
)

__all__ = [
    "ADJACENT_GAP_HEIGHT_RATIO",
    "BAND_COVERAGE_RATIO",
    "BAND_MIN_WIDTH_FRACTION",
    "Box",
    "COLLINEAR_SPACING_RATIO",
    "COLUMN_COVERAGE_RATIO",
    "COVERAGE_SLICES",
    "FRAGMENT_MAX_WIDTH_RATIO",
    "FragmentMerge",
    "NARROW_WIDTH_FRACTION",
    "OVERLAP_DEFECT_RATIO",
    "OVERLAP_DUPLICATE_RATIO",
    "OverlapTrim",
    "PageStats",
    "Point",
    "Polygon",
    "Segment",
    "SpanningSplit",
    "Suspect",
    "TINY_HEIGHT_RATIO",
    "TINY_MAX_POINTS",
    "baseline_distance",
    "bbox",
    "clip_across",
    "clip_baseline",
    "clip_half",
    "clip_polygon",
    "column_bands",
    "convex_hull",
    "find_fragments",
    "find_overlaps",
    "find_spanning",
    "find_suspects",
    "height_of",
    "line_spacing",
    "merge_baseline_group",
    "merge_baselines",
    "merge_polygon_group",
    "merge_polygons",
    "overlap_area",
    "page_stats",
    "polygon_area",
    "width_of",
    "y_at",
]
