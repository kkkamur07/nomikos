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
    clip_baseline,
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
        assert NARROW_WIDTH_FRACTION == 0.06
        candidate = seg("c", rect(0.0, 120.0, 400.0, 500.0))
        assert find_suspects(self._page(candidate), stats()) == []

    def test_just_under_the_narrow_limit_outside_is_noise(self) -> None:
        candidate = seg("c", rect(0.0, 119.9, 400.0, 500.0))
        (suspect,) = find_suspects(self._page(candidate), stats())
        assert suspect.line_id == "c"
        assert suspect.reasons == [
            "narrower than 6% of the page",
            "outside every column",
        ]

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
        assert BAND_MIN_WIDTH_FRACTION == 0.15
        (band,) = column_bands([seg("v", rect(0.0, 300.0, 0.0, 100.0))], W)
        assert band == (0.0, 300.0)

    def test_single_voter_at_the_vote_width_still_bands(self) -> None:
        # A lone 300px voter on a 2000px page spans exactly the 0.15 floor, so
        # the band survives; widening the floor to 0.16 would drop it.
        assert column_bands([seg("v", rect(300.0, 600.0, 0.0, 100.0))], W) == [(300.0, 600.0)]

    def test_band_exactly_at_the_floor_width_survives(self) -> None:
        (band,) = column_bands([seg("v", rect(0.0, 300.0, 0.0, 100.0))], W)
        assert band[1] - band[0] == pytest.approx(300.0)

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
        assert COLLINEAR_SPACING_RATIO == 0.4
        pair = self._pair((300.0, 700.0), (760.0, 900.0), second_dy=40.0)
        assert find_fragments(pair, stats()) == []

    def test_gap_boundary_offers_but_beyond_does_not(self) -> None:
        assert ADJACENT_GAP_HEIGHT_RATIO == 1.5
        at = self._pair((300.0, 700.0), (850.0, 990.0))
        assert len(find_fragments(at, stats())) == 1
        beyond = self._pair((300.0, 700.0), (850.1, 990.1))
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


class TestYAtPins:
    def test_clamps_read_three_point_ends(self) -> None:
        line = [[0.0, 0.0], [5.0, 100.0], [10.0, 20.0]]
        assert y_at(line, -1.0) == 0.0
        assert y_at(line, 99.0) == 20.0

    def test_interior_interpolation_uses_x(self) -> None:
        assert y_at([[10.0, 0.0], [20.0, 30.0]], 15.0) == 15.0
        assert y_at([[5.0, 50.0], [10.0, 20.0]], 10.0) == 20.0
        assert y_at([[0.0, 0.0], [10.0, 5.0]], 7.0) == 3.5


class TestClipPins:
    def test_fractional_cut_rounds_to_one_decimal(self) -> None:
        clipped = clip_half([[1.0, 1.0], [5.0, 3.0], [5.0, 5.0], [1.0, 5.0]], 2.55, keep_left=True)
        assert clipped == [[1.0, 1.0], [2.5, 1.8], [2.5, 5.0], [1.0, 5.0]]

    def test_cut_on_the_edge_keeps_everything(self) -> None:
        square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        assert clip_half(square, 4.0, keep_left=True) == square
        assert clip_half(square, 0.0, keep_left=False) == square

    def test_across_cut_rounds_the_crossing(self) -> None:
        clipped = clip_across([[0.0, 0.0], [6.0, 2.0], [0.0, 4.0]], 1.55, keep_above=True)
        assert clipped == [[0.0, 1.6], [0.0, 0.0], [4.7, 1.6]]


class TestClipBaselinePins:
    def test_full_range_returns_the_endpoints(self) -> None:
        assert clip_baseline([[0.0, 5.0], [20.0, 5.0]], None, None) == [[0, 5], [20, 5]]

    def test_fractional_window_rounds(self) -> None:
        assert clip_baseline([[0.0, 5.0], [20.0, 5.0]], 0.05, 19.95) == [
            [0.1, 5.0],
            [19.9, 5.0],
        ]

    def test_low_defaults_to_the_first_vertex(self) -> None:
        baseline = [[3.0, 7.0], [9.0, 1.0], [20.0, 4.0]]
        assert clip_baseline(baseline, None, None) == [[3, 7], [9.0, 1.0], [20, 4]]

    def test_given_low_clamps_to_the_line(self) -> None:
        baseline = [[3.0, 7.0], [9.0, 1.0], [20.0, 4.0]]
        assert clip_baseline(baseline, 5.0, None) == [[5, 5.0], [9.0, 1.0], [20, 4]]

    def test_given_high_clamps_to_the_line(self) -> None:
        baseline = [[3.0, 7.0], [9.0, 1.0], [20.0, 4.0]]
        assert clip_baseline(baseline, None, 15.0) == [[3, 7], [9.0, 1.0], [15, 2.6]]

    def test_narrow_window_returns_nothing(self) -> None:
        baseline = [[0.0, 5.0], [30.0, 5.0]]
        assert clip_baseline(baseline, 5.0, 12.0) == []
        assert clip_baseline(baseline, 5.0, 15.0) == [[5, 5.0], [15, 5.0]]
        assert clip_baseline(baseline, 5.0, 16.0) != []

    def test_inner_points_exclude_the_window_edges(self) -> None:
        assert clip_baseline([[5.0, 9.0], [10.0, 9.0], [20.0, 9.0]], 5.0, 20.0) == [
            [5, 9],
            [10.0, 9.0],
            [20, 9],
        ]
        assert clip_baseline([[0.0, 9.0], [20.0, 9.0], [30.0, 9.0]], 0.0, 30.0) == [
            [0, 9],
            [20.0, 9.0],
            [30, 9],
        ]
        assert clip_baseline([[0.0, 5.0], [10.0, 50.0], [20.0, 5.0]], 0.0, 20.0) == [
            [0, 5],
            [10.0, 50.0],
            [20, 5],
        ]

    def test_short_baseline_returns_nothing(self) -> None:
        assert clip_baseline([[1.0, 2.0]], None, None) == []


class TestBaselineDistancePins:
    def test_shifted_slanted_lines(self) -> None:
        first = [[5.0, 0.0], [9.0, 30.0]]
        second = [[7.0, 10.0], [20.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(16.7045455)

    def test_stacked_flat_lines(self) -> None:
        first = [[5.0, 5.0], [15.0, 15.0]]
        second = [[5.0, 15.0], [15.0, 15.0]]
        assert baseline_distance(first, second) == pytest.approx(5.0)

    def test_vertical_lines_use_mean_heights(self) -> None:
        first = [[5.0, 0.0], [5.0, 10.0]]
        second = [[5.0, 20.0], [5.0, 30.0]]
        assert baseline_distance(first, second) == 20.0


def test_clip_baseline_short_range() -> None:
    assert clip_baseline([[0.0, 1.0], [15.0, 1.0]], None, None) == [[0, 1], [15, 1]]


class TestPageStatsPins:
    def test_empty_points_segment_votes_for_nothing(self) -> None:
        blank = seg("b", [], baseline=[])
        assert column_bands([blank], W) == []

    def test_empty_page_has_zero_medians(self) -> None:
        result = page_stats([], W, H)
        assert result.median_width == 0.0
        assert result.median_height == 0.0
        assert result.line_spacing == 0.0
        assert result.bands == []

    def test_triangle_counts_as_usable(self) -> None:
        voter = seg("v", rect(1500.0, 1900.0, 400.0, 500.0))
        triangle = seg("t", [[300.0, 850.0], [1100.0, 850.0], [700.0, 950.0]])
        result = page_stats([voter, triangle], W, H)
        assert result.median_width == 600.0

    def test_span_start_exactly_on_the_window_edge_counts(self) -> None:
        (band,) = column_bands([seg("v", rect(-10.0, 300.0, 0.0, 100.0))], W)
        assert band == (-10.0, 300.0)

    def test_span_end_exactly_on_the_window_edge_counts(self) -> None:
        (band,) = column_bands([seg("v", rect(0.0, 320.0, 0.0, 100.0))], W)
        assert band == (0.0, 320.0)

    def test_duplicate_baselines_give_zero_spacing(self) -> None:
        first = seg("a", rect(300.0, 900.0, 400.0, 500.0), [[300.0, 400.0], [900.0, 400.0]])
        second = seg("b", rect(300.0, 900.0, 400.0, 500.0), [[300.0, 400.0], [900.0, 400.0]])
        assert line_spacing([first, second], [(200.0, 1200.0)]) == 0.0

    def test_zero_gap_needs_a_positive_gap_beside_it(self) -> None:
        rows = [
            seg(f"l{i}", rect(300.0, 900.0, y, y + 100.0), [[300.0, y], [900.0, y]])
            for i, y in enumerate((400.0, 400.0, 500.0))
        ]
        assert line_spacing(rows, [(200.0, 1200.0)]) == 100.0

    def test_half_pixel_gap_is_a_gap(self) -> None:
        rows = [
            seg(f"l{i}", rect(300.0, 900.0, y, y + 100.0), [[300.0, y], [900.0, y]])
            for i, y in enumerate((400.0, 400.5))
        ]
        assert line_spacing(rows, [(200.0, 1200.0)]) == 0.5


class TestBandIndexPins:
    def test_two_pixel_mark_outside_is_outside(self) -> None:
        from backend.annotation.application.segment_health.page_stats import _band_index

        mark = seg("m", rect(0.0, 2.0, 400.0, 500.0))
        assert _band_index(mark, [(1500.0, 1900.0)]) is None

    def test_exact_tie_prefers_the_first_band(self) -> None:
        from backend.annotation.application.segment_health.page_stats import _band_index

        segment = seg("s", rect(950.0, 1050.0, 400.0, 500.0))
        assert _band_index(segment, [(0.0, 1000.0), (1000.0, 2000.0)]) == 0

    def test_half_covered_segment_lives_in_the_band(self) -> None:
        from backend.annotation.application.segment_health.page_stats import _band_index

        segment = seg("s", rect(1450.0, 1550.0, 400.0, 500.0))
        assert _band_index(segment, [(1500.0, 1900.0)]) == 0

    def test_grazing_segment_lives_outside(self) -> None:
        from backend.annotation.application.segment_health.page_stats import _band_index

        segment = seg("s", rect(1420.0, 1520.0, 400.0, 500.0))
        assert _band_index(segment, [(1500.0, 1900.0)]) is None

    def test_barely_touching_segment_lives_outside(self) -> None:
        from backend.annotation.application.segment_health.page_stats import _band_index

        segment = seg("s", rect(1890.0, 1990.0, 400.0, 500.0))
        assert _band_index(segment, [(1500.0, 1900.0)]) is None


class TestSuspectKillerPins:
    def test_triangle_outside_is_flagged(self) -> None:
        voter = seg("v", rect(1500.0, 1900.0, 400.0, 500.0))
        triangle = seg("t", [[0.0, 450.0], [100.0, 450.0], [50.0, 550.0]])
        (suspect,) = find_suspects([voter, triangle], stats())
        assert suspect.line_id == "t"

    def test_subpixel_median_still_marks_a_sliver(self) -> None:
        voters = [
            seg(f"v{i}", rect(1500.0, 1900.0, 400.0, 400.0 + height))
            for i, height in enumerate((0.4, 0.5, 0.6))
        ]
        candidate = seg("c", [[1550.0, 460.0], [1590.0, 460.0], [1630.0, 460.1], [1630.0, 460.2]])
        page = voters + [candidate]
        result = page_stats(page, W, H)
        assert result.median_height == pytest.approx(0.45)
        (suspect,) = find_suspects(page, result)
        assert suspect.line_id == "c"

    def test_exact_sliver_height_is_not_tiny(self) -> None:
        voters = [
            seg(f"v{i}", rect(1500.0, 1900.0, y, y + 100.0)) for i, y in enumerate((400.0, 600.0))
        ]
        candidate = seg("c", rect(1550.0, 1600.0, 440.0, 500.0))
        page = voters + [candidate]
        assert find_suspects(page, page_stats(page, W, H)) == []

    def test_sliver_reasons_are_exact(self) -> None:
        points = [
            [300.0, 460.0],
            [340.0, 460.0],
            [380.0, 470.0],
            [380.0, 490.0],
            [340.0, 500.0],
            [300.0, 500.0],
        ]
        voter = seg("v", rect(1500.0, 1900.0, 400.0, 500.0))
        (suspect,) = find_suspects([voter, seg("c", points)], stats())
        assert suspect.reasons == [
            "narrower than 6% of the page",
            "only 6 corners and shorter than 60% of a typical line",
        ]


class TestSpanningKillerPins:
    def _bands(self) -> list[tuple[float, float]]:
        return [(200.0, 1200.0), (1250.0, 1900.0)]

    def test_triangle_crosser_splits(self) -> None:
        crosser = seg(
            "c",
            [[300.0, 400.0], [1800.0, 400.0], [1800.0, 600.0]],
            [[300.0, 450.0], [1800.0, 450.0]],
        )
        (split,) = find_spanning([crosser], stats(bands=self._bands()))
        assert split.cuts == [1225.0]

    def test_manual_triangle_crosser_is_left_alone(self) -> None:
        crosser = seg(
            "c",
            [[300.0, 400.0], [1800.0, 400.0], [1800.0, 600.0]],
            [[300.0, 450.0], [1800.0, 450.0]],
            manual=True,
        )
        assert find_spanning([crosser], stats(bands=self._bands())) == []

    def test_exact_half_coverage_counts(self) -> None:
        crosser = seg("c", rect(1100.0, 1600.0, 400.0, 500.0), [[1100.0, 450.0], [1600.0, 450.0]])
        (split,) = find_spanning([crosser], stats(bands=self._bands()))
        assert split.cuts == [1225.0]

    def test_wide_crosser_splits(self) -> None:
        crosser = seg("c", rect(1000.0, 2900.0, 400.0, 500.0), [[1000.0, 450.0], [2900.0, 450.0]])
        (split,) = find_spanning([crosser], stats(bands=self._bands()))
        assert split.cuts == [1225.0]

    def test_grazer_with_a_baseline_is_left_alone(self) -> None:
        grazer = seg("g", rect(1100.0, 1260.0, 400.0, 500.0), [[1100.0, 450.0], [1260.0, 450.0]])
        assert find_spanning([grazer], stats(bands=self._bands())) == []

    def test_split_pieces_meet_at_the_cut(self) -> None:
        crosser = seg("c", rect(300.0, 1800.0, 400.0, 500.0), [[300.0, 450.0], [1800.0, 450.0]])
        (split,) = find_spanning([crosser], stats(bands=self._bands()))
        (left_points, left_baseline), (right_points, right_baseline) = split.pieces
        assert max(point[0] for point in left_points) == 1225.0
        assert min(point[0] for point in right_points) == 1225.0
        assert left_baseline == [[300.0, 450.0], [1225.0, 450.0]]
        assert right_baseline == [[1225.0, 450.0], [1800.0, 450.0]]


class TestFragmentKillerPins:
    def test_triangle_pair_merges(self) -> None:
        first = seg(
            "a",
            [(300.0, 450.0), (700.0, 450.0), (500.0, 550.0)],
            [[300.0, 545.0], [700.0, 545.0]],
        )
        second = seg(
            "b",
            [(760.0, 500.0), (900.0, 500.0), (830.0, 560.0)],
            [[760.0, 555.0], [900.0, 555.0]],
        )
        (merge,) = find_fragments([first, second], stats())
        assert (merge.primary_id, merge.fragment_id) == ("a", "b")

    def test_manual_neighbour_is_not_a_candidate(self) -> None:
        first = seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]])
        second = seg("b", rect(760.0, 900.0, 450.0, 550.0), [[760.0, 550.0], [900.0, 550.0]])
        manual = seg(
            "m",
            [(980.0, 500.0), (1040.0, 500.0), (1010.0, 540.0)],
            [[980.0, 535.0], [1040.0, 535.0]],
            manual=True,
        )
        fourth = seg("c", rect(1100.0, 1200.0, 450.0, 550.0), [[1100.0, 550.0], [1200.0, 550.0]])
        (merge,) = find_fragments([first, second, manual, fourth], stats())
        assert (merge.primary_id, merge.fragment_id) == ("a", "b")

    def test_ten_pixel_offset_still_merges(self) -> None:
        first = seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]])
        second = seg("b", rect(760.0, 900.0, 460.0, 560.0), [[760.0, 560.0], [900.0, 560.0]])
        (merge,) = find_fragments([first, second], stats())
        assert (merge.primary_id, merge.fragment_id) == ("a", "b")

    def test_contained_fragment_merges(self) -> None:
        first = seg("a", rect(300.0, 1000.0, 450.0, 550.0), [[300.0, 550.0], [1000.0, 550.0]])
        second = seg("b", rect(400.0, 500.0, 460.0, 560.0), [[400.0, 560.0], [500.0, 560.0]])
        (merge,) = find_fragments([first, second], stats())
        assert (merge.primary_id, merge.fragment_id) == ("a", "b")

    def test_height_asymmetry_uses_the_taller_line(self) -> None:
        tall = seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]])
        short = seg("b", rect(800.0, 940.0, 530.0, 570.0), [[800.0, 570.0], [940.0, 570.0]])
        short_baseline = seg(
            "c", rect(800.0, 940.0, 520.0, 620.0), [[800.0, 570.0], [940.0, 570.0]]
        )
        assert len(find_fragments([tall, short], stats())) == 1
        flat = seg("d", rect(300.0, 700.0, 530.0, 570.0), [[300.0, 570.0], [700.0, 570.0]])
        assert len(find_fragments([flat, short_baseline], stats())) == 1

    def test_exact_sixty_percent_still_merges(self) -> None:
        first = seg("a", rect(300.0, 800.0, 450.0, 550.0), [[300.0, 550.0], [800.0, 550.0]])
        second = seg("b", rect(860.0, 1160.0, 450.0, 550.0), [[860.0, 550.0], [1160.0, 550.0]])
        (merge,) = find_fragments([first, second], stats())
        assert (merge.primary_id, merge.fragment_id) == ("a", "b")

    def test_claimed_fragment_is_not_reoffered(self) -> None:
        first = seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]])
        second = seg("b", rect(760.0, 900.0, 450.0, 550.0), [[760.0, 550.0], [900.0, 550.0]])
        third = seg("c", rect(960.0, 1040.0, 450.0, 550.0), [[960.0, 550.0], [1040.0, 550.0]])
        merges = find_fragments([first, second, third], stats())
        assert [(m.primary_id, m.fragment_id) for m in merges] == [("a", "b")]


class TestOverlapKillerPins:
    def test_slanted_baselines_set_the_cut(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 470.0], [1100.0, 490.0]])
        lower = seg("l", rect(300.0, 1100.0, 460.0, 560.0), [[300.0, 530.0], [1100.0, 550.0]])
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.cut == 510.0

    def test_triangle_pair_is_considered(self) -> None:
        first = seg(
            "a",
            [(300.0, 400.0), (1100.0, 400.0), (700.0, 600.0)],
            [[300.0, 480.0], [1100.0, 480.0]],
        )
        second = seg(
            "b",
            [(300.0, 500.0), (1100.0, 500.0), (700.0, 700.0)],
            [[300.0, 580.0], [1100.0, 580.0]],
        )
        assert len(find_overlaps([first, second], stats())) == 1

    def test_degenerate_first_does_not_hide_a_trim(self) -> None:
        degenerate = seg("d", [[300.0, 400.0], [700.0, 450.0], [1100.0, 500.0]])
        upper = seg("u", rect(300.0, 1100.0, 750.0, 850.0), [[300.0, 830.0], [1100.0, 830.0]])
        lower = seg("l", rect(300.0, 1100.0, 810.0, 910.0), [[300.0, 890.0], [1100.0, 890.0]])
        (trim,) = find_overlaps([degenerate, upper, lower], stats())
        assert (trim.upper_id, trim.lower_id) == ("u", "l")

    def test_reversed_tiny_order_still_reports_a_duplicate(self) -> None:
        big = seg("g", rect(200.0, 400.0, 390.0, 410.0), [[200.0, 405.0], [400.0, 405.0]])
        tiny = seg("t", [(300.0, 400.0), (301.0, 400.0), (300.0, 401.0)])
        (trim,) = find_overlaps([big, tiny], stats())
        assert trim.duplicate is True

    def test_touching_masks_share_nothing(self) -> None:
        upper = seg("u", rect(300.0, 900.0, 400.0, 500.0), [[300.0, 480.0], [900.0, 480.0]])
        lower = seg("l", rect(300.0, 900.0, 500.0, 600.0), [[300.0, 520.0], [900.0, 520.0]])
        assert find_overlaps([upper, lower], stats()) == []

    def test_contained_pair_is_a_duplicate(self) -> None:
        upper = seg("u", rect(300.0, 900.0, 400.0, 500.0), [[300.0, 450.0], [900.0, 450.0]])
        lower = seg("l", rect(350.0, 950.0, 450.0, 460.0), [[350.0, 455.0], [950.0, 455.0]])
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is True


class TestExtentAtPins:
    def _triangle(self) -> list[list[float]]:
        return [[5.0, 0.0], [15.0, 10.0], [5.0, 10.0]]

    def test_outside_vertical_is_nothing(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        assert _extent_at(rect(5.0, 15.0, 0.0, 10.0), -5.0) is None

    def test_touching_vertical_counts(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        assert _extent_at(self._triangle(), 5.0) == (0.0, 10.0)

    def test_just_outside_vertical_is_excluded(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        low, high = _extent_at(self._triangle(), 5.0 + 1.5e-9)
        assert low > 0.0
        assert high == 10.0

    def test_below_edge_is_excluded(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        low, high = _extent_at(self._triangle(), 5.0 - 1e-9)
        assert low < 0.0
        assert high == 10.0

    def test_above_edge_is_excluded(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        assert _extent_at(self._triangle(), 15.0 + 1.5e-9) is None

    def test_below_edge_tolerance_is_tight(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        assert _extent_at(self._triangle(), 5.0 - 1.5e-9) is None

    def test_exact_high_edge_counts(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        low, high = _extent_at(self._triangle(), 15.0 + 1e-9)
        assert high > 10.0

    def test_interior_interpolates(self) -> None:
        from backend.annotation.application.segment_health.geometry import _extent_at

        assert _extent_at(self._triangle(), 10.0) == (5.0, 10.0)


class TestMergeKillerPins:
    def test_stacked_same_xs_merges_to_a_ring(self) -> None:
        primary = rect(0.0, 10.0, 0.0, 10.0)
        fragment = rect(0.0, 10.0, 20.0, 30.0)
        assert merge_polygons(primary, fragment) == [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 30.0],
            [0.0, 30.0],
        ]

    def test_identical_rects_merge_to_one_ring(self) -> None:
        square = rect(0.0, 10.0, 0.0, 10.0)
        assert merge_polygons(square, square) == [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]

    def test_fractional_fallback_rounds(self) -> None:
        primary = [[0.05, 0.0], [0.05, 10.0], [0.05, 10.0]]
        fragment = [[0.05, 20.0], [0.05, 30.0], [0.05, 30.0]]
        assert merge_polygons(primary, fragment) == [
            [0.1, 0.0],
            [0.1, 10.0],
            [0.1, 20.0],
            [0.1, 30.0],
        ]

    def test_fractional_bridge_rounds(self) -> None:
        primary = [[0.05, 0.0], [10.0, 0.0], [10.0, 10.0], [0.05, 10.0]]
        fragment = [[14.0, 0.0], [20.0, 0.0], [20.0, 10.0], [14.0, 10.0]]
        merged = merge_polygons(primary, fragment)
        assert merged[0] == [0.1, 0.0]
        assert bbox(merged) == (0.1, 20.0, 0.0, 10.0)

    def test_dedupe_keeps_a_single_point(self) -> None:
        from backend.annotation.application.segment_health.geometry import _dedupe

        assert _dedupe([[5.0, 5.0]]) == [[5.0, 5.0]]
        assert _dedupe([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]) == [[1.0, 2.0], [3.0, 4.0]]

    def test_merge_baselines_keeps_every_point(self) -> None:
        assert merge_baselines([[0.0, 0.0], [10.0, 0.0]], [[14.0, 0.0], [30.0, 0.0]]) == [
            [0.0, 0.0],
            [10.0, 0.0],
            [14.0, 0.0],
            [30.0, 0.0],
        ]

    def test_merge_baselines_skips_near_duplicates(self) -> None:
        assert merge_baselines([[0.0, 0.0], [10.0, 0.0]], [[12.0, 1.0], [30.0, 0.0]]) == [
            [0.0, 0.0],
            [10.0, 0.0],
            [30.0, 0.0],
        ]

    def test_merge_baselines_two_point_result_keeps_ends(self) -> None:
        assert merge_baselines([[0.0, 0.0], [10.0, 0.0]], [[11.0, 0.0], [12.0, 0.0]]) == [
            [0.0, 0.0],
            [10.0, 0.0],
        ]

    def test_merge_baselines_single_cluster_falls_back_to_ends(self) -> None:
        assert merge_baselines([[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [2.5, 0.0]]) == [
            [0.0, 0.0],
            [2.5, 0.0],
        ]

    def test_merge_baselines_rounds_fractional_y(self) -> None:
        assert merge_baselines([[0.0, 0.05], [10.0, 0.0]], [[20.0, 0.0], [30.0, 0.0]]) == [
            [0.0, 0.1],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ]


class TestSlantedClipPins:
    def test_vertex_on_the_cut_is_kept_once(self) -> None:
        clipped = clip_half([[0.0, 0.0], [4.0, 2.0], [0.0, 4.0]], 4.0, keep_left=True)
        assert clipped == [[0.0, 0.0], [4.0, 2.0], [0.0, 4.0]]

    def test_vertex_on_the_right_cut_is_kept_once(self) -> None:
        clipped = clip_half([[0.0, 2.0], [4.0, 0.0], [4.0, 4.0]], 0.0, keep_left=False)
        assert clipped == [[0.0, 2.0], [4.0, 0.0], [4.0, 4.0]]


class TestSlantedBaselinePins:
    def test_low_endpoint_rounds_fractional_y(self) -> None:
        assert clip_baseline([[0.0, 0.0], [20.0, 10.0]], 0.05, 19.95) == [
            [0.1, 0.0],
            [19.9, 10.0],
        ]

    def test_three_point_endpoints_discriminate(self) -> None:
        first = [[3.0, 0.0], [9.0, 30.0], [25.0, 15.0]]
        second = [[5.0, 10.0], [20.0, 10.0], [30.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(10.0653409)

    def test_low_uses_the_first_endpoint(self) -> None:
        first = [[1.0, 0.0], [9.0, 30.0]]
        second = [[2.0, 10.0], [40.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(17.625)

    def test_low_uses_the_second_start(self) -> None:
        first = [[3.0, 0.0], [9.0, 30.0]]
        second = [[2.0, 10.0], [5.0, 10.0], [40.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(17.6363636)

    def test_high_uses_the_first_end(self) -> None:
        first = [[3.0, 0.0], [9.0, 30.0], [35.0, 15.0]]
        second = [[5.0, 10.0], [20.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(11.5874126)

    def test_high_uses_the_second_end(self) -> None:
        first = [[3.0, 0.0], [9.0, 30.0]]
        second = [[5.0, 10.0], [20.0, 10.0], [40.0, 10.0]]
        assert baseline_distance(first, second) == pytest.approx(18.0454545)


class TestOverlapKillerPins2:
    def _rect_pair(self, upper_y, lower_y):
        upper = seg(
            "u",
            rect(300.0, 1100.0, upper_y, upper_y + 100.0),
            [[300.0, upper_y + 80.0], [1100.0, upper_y + 80.0]],
        )
        lower = seg(
            "l",
            rect(300.0, 1100.0, lower_y, lower_y + 100.0),
            [[300.0, lower_y + 80.0], [1100.0, lower_y + 80.0]],
        )
        return upper, lower

    def test_exact_defect_ratio_offers(self) -> None:
        from backend.annotation.application.segment_health.finders import OVERLAP_DEFECT_RATIO

        assert OVERLAP_DEFECT_RATIO == 0.2
        upper, lower = self._rect_pair(400.0, 480.0)
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.ratio == pytest.approx(0.2)
        assert trim.duplicate is False

    def test_exact_duplicate_ratio_reports_no_trim(self) -> None:
        from backend.annotation.application.segment_health.finders import OVERLAP_DUPLICATE_RATIO

        assert OVERLAP_DUPLICATE_RATIO == 0.5
        upper, lower = self._rect_pair(400.0, 450.0)
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.ratio == pytest.approx(0.5)
        assert trim.duplicate is True

    def test_fractional_duplicate_cut_rounds_once(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 480.0], [1100.0, 480.0]])
        lower = seg("l", rect(300.0, 1100.0, 405.0, 505.0), [[300.0, 541.1], [1100.0, 541.1]])
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is True
        assert trim.cut == 510.6

    def test_fractional_trim_cut_rounds_once(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 400.0, 500.0), [[300.0, 470.05], [1100.0, 490.05]])
        lower = seg("l", rect(300.0, 1100.0, 460.0, 560.0), [[300.0, 530.05], [1100.0, 550.05]])
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is False
        assert trim.cut == 510.0

    def test_degenerate_first_skips_to_the_pair(self) -> None:
        degenerate = seg("d", [[300.0, 800.0], [700.0, 800.0], [1100.0, 800.0]])
        upper = seg("u", rect(300.0, 1100.0, 750.0, 850.0), [[300.0, 830.0], [1100.0, 830.0]])
        lower = seg("l", rect(300.0, 1100.0, 810.0, 910.0), [[300.0, 890.0], [1100.0, 890.0]])
        (trim,) = find_overlaps([degenerate, upper, lower], stats())
        assert (trim.upper_id, trim.lower_id) == ("u", "l")

    def test_degenerate_second_skips_the_pair(self) -> None:
        upper = seg("u", rect(300.0, 1100.0, 750.0, 850.0), [[300.0, 830.0], [1100.0, 830.0]])
        degenerate = seg("d", [[300.0, 800.0], [700.0, 800.0], [1100.0, 800.0]])
        assert find_overlaps([upper, degenerate], stats()) == []

    def test_reversed_tiny_order_reports_a_duplicate(self) -> None:
        big = seg("g", rect(200.0, 400.0, 390.0, 410.0), [[200.0, 405.0], [400.0, 405.0]])
        tiny = seg("t", [(300.0, 400.0), (301.0, 400.0), (300.0, 401.0)])
        (trim,) = find_overlaps([big, tiny], stats())
        assert trim.duplicate is True

    def test_sliver_guard_needs_both_sides_short(self) -> None:
        crosser = seg("c", rect(300.0, 1800.0, 400.0, 500.0), [[300.0, 450.0], [1800.0, 450.0]])
        assert len(find_spanning([crosser], stats(bands=[(200.0, 1200.0), (1250.0, 1900.0)]))) == 1

    def test_zero_spacing_finds_nothing(self) -> None:
        pair = [
            seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]]),
            seg("b", rect(760.0, 900.0, 450.0, 550.0), [[760.0, 550.0], [900.0, 550.0]]),
        ]
        assert find_fragments(pair, stats(spacing=0.0)) == []

    def test_half_spacing_still_merges(self) -> None:
        pair = [
            seg("a", rect(300.0, 700.0, 450.0, 550.0), [[300.0, 550.0], [700.0, 550.0]]),
            seg("b", rect(760.0, 900.0, 450.0, 550.0), [[760.0, 550.0], [900.0, 550.0]]),
        ]
        assert len(find_fragments(pair, stats(spacing=0.5))) == 1


class TestThreePointKeptPins:
    def test_triangle_kept_side_has_three_points(self) -> None:
        upper = seg(
            "u",
            [(300.0, 600.0), (1100.0, 600.0), (700.0, 400.0)],
            [[300.0, 450.0], [1100.0, 450.0]],
        )
        lower = seg(
            "l",
            rect(300.0, 1100.0, 550.0, 750.0),
            [[300.0, 700.0], [1100.0, 700.0]],
        )
        (trim,) = find_overlaps([upper, lower], stats())
        assert trim.duplicate is False
        assert len(trim.upper_points) == 3
