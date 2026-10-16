"""Boundary pins for every segment-health threshold and exact geometry outputs.

Each threshold exists as a comparison in the finders; a test here holds an
input on each side of the boundary, so moving the constant or flipping the
comparison changes an asserted verdict.
"""

from __future__ import annotations

import pytest

from backend.annotation.application.segment_health import (
    PageStats,
    Segment,
    baseline_distance,
    bbox,
    clip_across,
    clip_half,
    column_bands,
    find_fragments,
    find_overlaps,
    find_spanning,
    find_suspects,
    height_of,
    line_spacing,
    merge_baselines,
    merge_polygons,
    overlap_area,
    page_stats,
    polygon_area,
    width_of,
    y_at,
)
from backend.annotation.application.segment_health.finders import (
    ADJACENT_GAP_HEIGHT_RATIO,
    BAND_COVERAGE_RATIO,
    COLLINEAR_SPACING_RATIO,
    NARROW_WIDTH_FRACTION,
    OVERLAP_DUPLICATE_RATIO,
    TINY_HEIGHT_RATIO,
)
from backend.annotation.application.segment_health.geometry import (
    FragmentMerge,
    OverlapTrim,
    SpanningSplit,
    Suspect,
)
from backend.annotation.application.segment_health.page_stats import (
    BAND_MIN_WIDTH_FRACTION,
    COLUMN_COVERAGE_RATIO,
)

W = 2000.0
H = 3000.0


def seg(
    identifier: str,
    points: list[list[float]],
    baseline: list[list[float]] | None = None,
    *,
    manual: bool = False,
    has_text: bool = False,
    is_paired: bool = False,
) -> Segment:
    base = baseline if baseline is not None else [points[0], points[-1]]
    return Segment(
        id=identifier,
        points=points,
        baseline=base,
        manual_geometry=manual,
        has_text=has_text,
        is_paired=is_paired,
    )


def rect(x0: float, x1: float, top: float, bottom: float) -> list[list[float]]:
    return [[x0, top], [x1, top], [x1, bottom], [x0, bottom]]


def stats(bands: list[tuple[float, float]] | None = None, *, spacing: float = 100.0) -> PageStats:
    return PageStats(
        width=W,
        height=H,
        bands=bands if bands is not None else [(200.0, 1200.0)],
        median_width=800.0,
        median_height=100.0,
        line_spacing=spacing,
    )


class TestModelContract:
    @pytest.mark.parametrize(
        "cls", [Segment, PageStats, Suspect, SpanningSplit, FragmentMerge, OverlapTrim]
    )
    def test_findings_are_frozen(self, cls) -> None:
        assert cls.__dataclass_params__.frozen is True

    def test_segment_defaults_claim_no_human_work(self) -> None:
        segment = Segment(id="s", points=[[0.0, 0.0]], baseline=[])
        assert segment.manual_geometry is False
        assert segment.has_text is False
        assert segment.is_paired is False

    def test_finding_defaults_are_empty(self) -> None:
        assert SpanningSplit(line_id="s", cuts=[]).pieces == []
        trim = OverlapTrim(upper_id="u", lower_id="l", ratio=0.3, cut=1.0)
        assert trim.upper_points == []
        assert trim.lower_points == []
        assert trim.upper_loss == 0.0
        assert trim.lower_loss == 0.0
        assert trim.duplicate is False


class TestExactGeometry:
    def test_bbox_width_and_height(self) -> None:
        points = [[1.0, 2.0], [5.0, 2.0], [5.0, 9.0], [1.0, 9.0]]
        assert bbox(points) == (1.0, 5.0, 2.0, 9.0)
        assert width_of(points) == 4.0
        assert height_of(points) == 7.0

    def test_y_at_interpolates_and_clamps(self) -> None:
        line = [[0.0, 0.0], [10.0, 20.0]]
        assert y_at(line, 5.0) == 10.0
        assert y_at(line, -3.0) == 0.0
        assert y_at(line, 99.0) == 20.0

    def test_clip_half_keeps_one_side(self) -> None:
        square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        assert clip_half(square, 2.0, keep_left=True) == [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ]

    def test_clip_across_keeps_above(self) -> None:
        square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        kept = clip_across(square, 2.0, keep_above=True)
        assert max(point[1] for point in kept) <= 2.0

    def test_polygon_area_and_overlap(self) -> None:
        first = rect(0.0, 10.0, 0.0, 10.0)
        second = rect(5.0, 15.0, 0.0, 10.0)
        assert polygon_area(first) == 100.0
        assert overlap_area(first, second) == 50.0
        assert overlap_area(first, rect(20.0, 30.0, 0.0, 10.0)) == 0.0

    def test_merge_polygons_bridges_the_gap(self) -> None:
        merged = merge_polygons(rect(0.0, 10.0, 0.0, 10.0), rect(14.0, 20.0, 0.0, 10.0))
        assert bbox(merged) == (0.0, 20.0, 0.0, 10.0)
        assert polygon_area(merged) == pytest.approx(200.0)

    def test_merge_baselines_runs_left_to_right(self) -> None:
        merged = merge_baselines([[14.0, 5.0], [20.0, 5.0]], [[0.0, 5.0], [10.0, 5.0]])
        assert [point[0] for point in merged] == sorted(point[0] for point in merged)
        assert merged[0] == [0.0, 5.0]
        assert merged[-1] == [20.0, 5.0]

    def test_baseline_distance_is_vertical_not_pointwise(self) -> None:
        first = [[0.0, 0.0], [10.0, 0.0]]
        second = [[100.0, 5.0], [110.0, 5.0]]
        assert baseline_distance(first, second) == 5.0
        assert baseline_distance([[0.0, 0.0]], second) == float("inf")


class TestSuspectBoundaries:
    def _page(self, extra: Segment, voter_x0: float = 1500.0) -> list[Segment]:
        voter = seg("v", rect(voter_x0, voter_x0 + 400.0, 400.0, 500.0))
        return [voter, extra]

    def test_exactly_at_the_narrow_limit_is_not_noise(self) -> None:
        limit = NARROW_WIDTH_FRACTION * W
        candidate = seg("c", rect(0.0, limit, 400.0, 500.0))
        assert find_suspects(self._page(candidate), stats()) == []

    def test_just_under_the_narrow_limit_outside_is_noise(self) -> None:
        candidate = seg("c", rect(0.0, NARROW_WIDTH_FRACTION * W - 0.1, 400.0, 500.0))
        (suspect,) = find_suspects(self._page(candidate), stats())
        assert suspect.line_id == "c"
        assert suspect.reasons == [
            "narrower than 6% of the page",
            "outside every column",
        ]

    def test_medium_width_outside_is_still_running_text(self) -> None:
        candidate = seg("c", rect(0.0, 0.5 * W, 400.0, 500.0))
        assert find_suspects(self._page(candidate), stats()) == []

    def test_seven_corners_is_not_a_sliver(self) -> None:
        points = [
            [300.0, 450.0],
            [340.0, 450.0],
            [380.0, 460.0],
            [380.0, 490.0],
            [340.0, 500.0],
            [300.0, 500.0],
            [290.0, 475.0],
        ]
        candidate = seg("c", points)
        assert find_suspects(self._page(candidate), stats()) == []

    def test_six_squat_corners_is_a_sliver(self) -> None:
        points = [
            [300.0, 460.0],
            [340.0, 460.0],
            [380.0, 470.0],
            [380.0, 490.0],
            [340.0, 500.0],
            [300.0, 500.0],
        ]
        candidate = seg("c", points)
        (suspect,) = find_suspects(self._page(candidate), stats())
        assert suspect.line_id == "c"
        assert any("corners" in reason for reason in suspect.reasons)

    def test_taller_than_the_sliver_ratio_is_not_tiny(self) -> None:
        candidate = seg("c", rect(300.0, 350.0, 430.0, 500.0))
        assert TINY_HEIGHT_RATIO * 100.0 == 60.0
        assert find_suspects(self._page(candidate), stats()) == []

    def test_text_and_pairings_are_never_flagged(self) -> None:
        narrow = rect(0.0, 10.0, 400.0, 500.0)
        page = self._page(seg("t", narrow, has_text=True)) + [seg("p", narrow, is_paired=True)]
        assert find_suspects(page, stats()) == []


class TestColumnBandBoundaries:
    def test_segment_exactly_at_the_vote_width_counts(self) -> None:
        width = BAND_MIN_WIDTH_FRACTION * W
        (band,) = column_bands([seg("v", rect(0.0, width, 0.0, 100.0))], W)
        assert band == (0.0, width)

    def test_single_voter_at_the_vote_width_still_bands(self) -> None:
        # A lone 300px voter on a 2000px page spans exactly the 0.15 floor, so
        # the band survives; widening the floor to 0.16 would drop it.
        assert column_bands([seg("v", rect(300.0, 600.0, 0.0, 100.0))], W) == [(300.0, 600.0)]

    def test_band_exactly_at_the_floor_width_survives(self) -> None:
        width = BAND_MIN_WIDTH_FRACTION * W
        (band,) = column_bands([seg("v", rect(0.0, width, 0.0, 100.0))], W)
        assert band[1] - band[0] == pytest.approx(width)

    def test_two_separated_columns_stay_two_bands(self) -> None:
        left = seg("l", rect(0.0, 300.0, 0.0, 100.0))
        right = seg("r", rect(1700.0, 2000.0, 0.0, 100.0))
        assert column_bands([left, right], W) == [(0.0, 300.0), (1700.0, 2000.0)]

    def test_floor_boundary_slice_decides_the_band(self) -> None:
        voters = [seg(f"a{i}", rect(0.0, 1000.0, 0.0, 100.0)) for i in range(7)]
        voters += [seg(f"b{i}", rect(500.0, 1000.0, 200.0, 300.0)) for i in range(13)]
        assert pytest.approx(7.0) == COLUMN_COVERAGE_RATIO * 20
        assert column_bands(voters, W) == [(0.0, 1000.0)]

    def test_negative_edge_starts_the_profile_at_zero(self) -> None:
        # The overrunner is ignored when snapping, so the band starts at the
        # first covered slice; skipping slice zero would move it to 10.
        (band,) = column_bands([seg("v", rect(-50.0, 500.0, 0.0, 100.0))], W)
        assert band == (0.0, 510.0)

    def test_overrunning_edge_is_clamped_to_the_page(self) -> None:
        (band,) = column_bands([seg("v", rect(1700.0, 2500.0, 0.0, 100.0))], W)
        assert band == (1700.0, 2000.0)

    def test_empty_page_and_empty_width_have_no_bands(self) -> None:
        assert column_bands([], W) == []
        assert column_bands([seg("v", rect(0.0, 300.0, 0.0, 100.0))], 0.0) == []

    def test_line_spacing_is_median_within_a_band(self) -> None:
        segments = [
            seg(f"l{i}", rect(300.0, 900.0, y, y + 100.0), [[300.0, y], [900.0, y]])
            for i, y in enumerate((400.0, 500.0, 600.0, 700.0))
        ]
        assert line_spacing(segments, [(200.0, 1200.0)]) == 100.0

    def test_line_spacing_needs_two_point_baselines(self) -> None:
        segments = [
            seg("a", rect(300.0, 900.0, 400.0, 500.0), [[300.0, 400.0]]),
            seg("b", rect(300.0, 900.0, 500.0, 600.0), [[300.0, 500.0], [900.0, 500.0]]),
        ]
        assert line_spacing(segments, [(200.0, 1200.0)]) == 0.0

    def test_page_stats_reports_medians_and_spacing(self) -> None:
        segments = [
            seg(f"l{i}", rect(300.0, 900.0, y, y + 100.0), [[300.0, y], [900.0, y]])
            for i, y in enumerate((400.0, 500.0, 600.0))
        ]
        result = page_stats(segments, W, H)
        assert result.median_width == 600.0
        assert result.median_height == 100.0
        assert result.line_spacing == 100.0
        assert result.bands == [(300.0, 900.0)]


class TestFragmentBoundaries:
    def _pair(
        self,
        first_x: tuple[float, float],
        second_x: tuple[float, float],
        second_dy: float = 0.0,
    ) -> list[Segment]:
        first = seg(
            "a",
            rect(first_x[0], first_x[1], 450.0, 550.0),
            [[first_x[0], 550.0], [first_x[1], 550.0]],
        )
        second = seg(
            "b",
            rect(second_x[0], second_x[1], 450.0 + second_dy, 550.0 + second_dy),
            [[second_x[0], 550.0 + second_dy], [second_x[1], 550.0 + second_dy]],
        )
        return [first, second]

    def test_collinear_boundary_rejects(self) -> None:
        limit = COLLINEAR_SPACING_RATIO * 100.0
        assert limit == 40.0
        pair = self._pair((300.0, 700.0), (760.0, 900.0), second_dy=limit)
        assert find_fragments(pair, stats()) == []

    def test_gap_boundary_offers_but_beyond_does_not(self) -> None:
        limit = ADJACENT_GAP_HEIGHT_RATIO * 100.0
        at = self._pair((300.0, 700.0), (700.0 + limit, 840.0 + limit))
        assert len(find_fragments(at, stats())) == 1
        beyond = self._pair((300.0, 700.0), (700.0 + limit + 0.1, 840.0 + limit + 0.1))
        assert find_fragments(beyond, stats()) == []

    def test_equal_halves_are_two_lines_not_a_fragment(self) -> None:
        pair = self._pair((300.0, 700.0), (700.0, 1100.0))
        assert find_fragments(pair, stats()) == []

    def test_lopsided_close_pair_merges_and_reports_ids(self) -> None:
        pair = self._pair((300.0, 900.0), (960.0, 1060.0))
        (merge,) = find_fragments(pair, stats())
        assert merge.primary_id == "a"
        assert merge.fragment_id == "b"
        assert merge.baseline[0][0] < merge.baseline[-1][0]

    def test_pair_across_bands_never_merges(self) -> None:
        pair = self._pair((300.0, 900.0), (1300.0, 1400.0))
        assert find_fragments(pair, stats(bands=[(200.0, 1200.0), (1250.0, 1900.0)])) == []


class TestSpanningAndOverlapBoundaries:
    def test_half_covered_band_does_not_count(self) -> None:
        bands = [(200.0, 1200.0), (1250.0, 1900.0)]
        assert pytest.approx(0.2) == BAND_COVERAGE_RATIO
        grazer = seg("g", rect(1100.0, 1260.0, 400.0, 500.0))
        assert find_spanning([grazer], stats(bands=bands)) == []

    def test_gutter_crosser_splits_at_the_midpoint(self) -> None:
        bands = [(200.0, 1200.0), (1250.0, 1900.0)]
        crosser = seg("c", rect(300.0, 1800.0, 400.0, 500.0), [[300.0, 450.0], [1800.0, 450.0]])
        (split,) = find_spanning([crosser], stats(bands=bands))
        assert split.cuts == [1225.0]
        assert len(split.pieces) == 2

    def test_below_defect_ratio_is_neighbours_not_a_defect(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 480.0], [1100.0, 480.0]])
        lower = seg("l", rect(300.0, 1100.0, 495.0, 595.0), [[300.0, 515.0], [1100.0, 515.0]])
        assert find_overlaps([upper, lower], stats()) == []

    def test_defect_pair_trims_to_disjoint_masks(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 480.0], [1100.0, 480.0]])
        lower = seg("l", rect(300.0, 1100.0, 460.0, 560.0), [[300.0, 540.0], [1100.0, 540.0]])
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is False
        assert trim.cut == pytest.approx(510.0)
        assert trim.ratio == pytest.approx(0.4)
        assert overlap_area(trim.upper_points, trim.lower_points) == 0.0
        assert trim.upper_loss == 0.0
        assert trim.lower_loss == pytest.approx(0.5)

    def test_duplicate_pair_reports_no_trim(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 480.0], [1100.0, 480.0]])
        lower = seg("l", rect(300.0, 1100.0, 405.0, 505.0), [[300.0, 482.0], [1100.0, 482.0]])
        assert OVERLAP_DUPLICATE_RATIO == 0.5
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is True
        assert trim.upper_points == []
        assert trim.cut == round(trim.cut, 1)
