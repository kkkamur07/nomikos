"""Exact values from the overlap stage: ratios, cuts, drops and flags.

Every threshold is pinned on the boundary: a shared ratio equal to the cut
threshold acts, a piece of exactly half its polygon is refused, a duplicate
needs strictly more than half of the larger polygon.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det import overlaps
from nomikos_inference.architectures.ppocr_det.geometry import Baseline, Work
from nomikos_inference.architectures.ppocr_det.overlaps import (
    DEFAULT_OVERLAP_CUT_THRESHOLD,
    DUPLICATE_SHARED_RATIO,
    _candidate_pairs,
    _clip_baseline,
    _cut_pair,
    _half_plane_piece,
    _median_line_spacing,
    _resolve_close_pair,
    _resolve_pair,
    _shared_over_larger,
    _shared_ratio,
    resolve_overlaps,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines

FLAT = Polygon([(0, 0), (10, 0), (10, 0), (0, 0)])


def _box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    baseline: Baseline | None = None,
    *,
    score: float = 0.9,
    members: tuple[int, ...] = (0,),
) -> Work:
    if baseline is None:
        level = ymin + 0.75 * (ymax - ymin)
        baseline = ((xmin, level), (xmax, level))
    return Work(
        poly=Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]),
        baseline=baseline,
        score=score,
        members=members,
    )


def _flat() -> Work:
    return Work(poly=FLAT, baseline=((0.0, 0.0), (10.0, 0.0)), score=0.5, members=(9,))


def _quad(xmin: float, ymin: float, xmax: float, ymax: float) -> DetectedQuad:
    return DetectedQuad(points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], score=0.9)


def test_thresholds() -> None:
    assert DEFAULT_OVERLAP_CUT_THRESHOLD == 0.20
    assert DUPLICATE_SHARED_RATIO == 0.50
    assert overlaps._MAX_OVERLAP_PASSES == 100


def test_shared_ratios_divide_by_the_smaller_and_the_larger() -> None:
    small, large = _box(0, 0, 10, 10), _box(5, 0, 25, 10)

    assert _shared_ratio(small, large) == 0.5
    assert _shared_ratio(large, small) == 0.5
    assert _shared_over_larger(small, large) == 0.25
    assert _shared_over_larger(large, small) == 0.25


def test_shared_ratios_of_a_flat_polygon_are_zero() -> None:
    assert _shared_ratio(_flat(), _box(0, 0, 10, 10)) == 0.0
    assert _shared_over_larger(_flat(), _flat()) == 0.0


def test_shared_ratios_of_polygons_under_one_pixel() -> None:
    tiny = _box(0, 0, 0.5, 1)

    assert _shared_ratio(tiny, _box(0, 0, 0.5, 1)) == 1.0
    assert _shared_over_larger(tiny, _box(0, 0, 0.5, 1)) == 1.0


def test_median_line_spacing_is_the_median_row_pitch() -> None:
    quads = [_quad(0, y, 100, y + 20) for y in (0, 30, 70, 100)]

    assert _median_line_spacing(layout_lines(quads), 9.0) == 30.0


def test_median_line_spacing_averages_a_row_and_falls_back() -> None:
    row_of_two = [
        _quad(0, 0, 100, 20),
        _quad(0, 60, 100, 80),
        _quad(0, 120, 100, 140),
        _quad(0, 30, 40, 50),
        _quad(50, 40, 90, 60),
    ]
    layout = layout_lines(row_of_two)

    # Row centres 10, 45 (the mean of 40 and 50), 70, 130: pitches 35, 25, 60.
    assert layout.columns[0].rows == ((0,), (3, 4), (1,), (2,))
    assert _median_line_spacing(layout, 9.0) == 35.0
    assert _median_line_spacing(layout_lines([_quad(0, 0, 100, 20)]), 9.0) == 9.0
    assert _median_line_spacing(layout_lines([]), 9.0) == 9.0


def test_clip_baseline_inside_a_polygon_keeps_the_inner_segment() -> None:
    kept = Polygon([(10, 0), (40, 0), (40, 10), (10, 10)])

    assert _clip_baseline(((0.0, 5.0), (100.0, 5.0)), kept) == ((10.0, 5.0), (40.0, 5.0))


def test_clip_baseline_keeps_both_ends_of_a_sloped_segment() -> None:
    kept = Polygon([(10, -5), (40, -5), (40, 20), (10, 20)])

    assert _clip_baseline(((0.0, 0.0), (100.0, 10.0)), kept) == ((10.0, 1.0), (40.0, 4.0))


def test_clip_baseline_keeps_a_piece_shorter_than_a_pixel() -> None:
    bar = Polygon([(0, 0), (1.5, 0), (1.5, 10), (1, 10), (1, 5), (0.5, 5), (0.5, 10), (0, 10)])

    assert _clip_baseline(((0.0, 7.0), (1.5, 7.0)), bar) == ((0.0, 7.0), (0.5, 7.0))


def test_clip_baseline_missing_or_touching_the_polygon_is_none() -> None:
    kept = Polygon([(10, 0), (40, 0), (40, 10), (10, 10)])
    corner = Polygon([(10, -5), (20, 0), (10, 5)])

    assert _clip_baseline(((0.0, 20.0), (100.0, 20.0)), kept) is None
    assert _clip_baseline(((0.0, 0.0), (10.0, 0.0)), corner) is None


def _notched(left: float, right: float) -> Polygon:
    """A 100 x 10 bar with a notch from y 5 down between ``left`` and ``right``."""
    return Polygon(
        [(0, 0), (100, 0), (100, 10), (right, 10), (right, 5), (left, 5), (left, 10), (0, 10)]
    )


def test_clip_baseline_split_in_two_keeps_the_longest_piece() -> None:
    baseline = ((0.0, 7.0), (100.0, 7.0))

    assert _clip_baseline(baseline, _notched(20, 40)) == ((40.0, 7.0), (100.0, 7.0))
    assert _clip_baseline(baseline, _notched(60, 80)) == ((0.0, 7.0), (60.0, 7.0))
    # Equal pieces: the first one stays.
    assert _clip_baseline(baseline, _notched(40, 60)) == ((0.0, 7.0), (40.0, 7.0))


def test_half_plane_piece_keeps_more_than_half_only() -> None:
    work = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    down = np.array([0.0, 1e5])
    arm = np.array([1e5, 0.0])

    below_ten = _half_plane_piece(work, np.array([50.0, 10.0]), arm, down)
    below_fifteen = _half_plane_piece(work, np.array([50.0, 15.0]), arm, down)

    assert below_ten is not None
    assert below_ten[0].bounds == (0.0, 10.0, 100.0, 30.0)
    assert below_ten[1] == ((0.0, 20.0), (100.0, 20.0))
    assert below_fifteen is None


def test_half_plane_piece_without_its_baseline_is_none() -> None:
    work = _box(0, 0, 100, 30, ((0.0, 2.0), (100.0, 2.0)))

    piece = _half_plane_piece(
        work, np.array([50.0, 10.0]), np.array([1e5, 0.0]), np.array([0.0, 1e5])
    )

    assert piece is None


def test_cut_pair_clips_the_lower_line_at_the_mid_baseline() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,))

    assert _cut_pair(upper, lower) is True
    assert (upper.poly.bounds, upper.area) == ((0.0, 0.0, 100.0, 30.0), 3000.0)
    assert (lower.poly.bounds, lower.area) == ((0.0, 30.0, 100.0, 50.0), 2000.0)
    assert lower.baseline == ((0.0, 40.0), (100.0, 40.0))


def test_cut_pair_works_with_the_pair_and_a_baseline_reversed() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((100.0, 40.0), (0.0, 40.0)), members=(1,))

    assert _cut_pair(lower, upper) is True
    assert upper.poly.bounds == (0.0, 0.0, 100.0, 30.0)
    assert lower.poly.bounds == (0.0, 30.0, 100.0, 50.0)
    assert lower.baseline == ((100.0, 40.0), (0.0, 40.0))


def test_cut_pair_cuts_both_lines_when_the_mid_line_crosses_both() -> None:
    upper = _box(0, 0, 100, 40, ((0.0, 15.0), (100.0, 15.0)))
    lower = _box(0, 20, 100, 60, ((0.0, 45.0), (100.0, 45.0)), members=(1,))

    assert _cut_pair(upper, lower) is True
    assert (upper.poly.bounds, upper.area) == ((0.0, 0.0, 100.0, 30.0), 3000.0)
    assert (lower.poly.bounds, lower.area) == ((0.0, 30.0, 100.0, 60.0), 3000.0)


def test_cut_pair_keeps_a_line_whose_centre_is_just_past_the_cut() -> None:
    # The lower line, passed first, has its centroid half a pixel below the
    # mid-baseline line at y 30.5.
    lower = _box(0, 1, 100, 61, ((0.0, 56.0), (100.0, 56.0)))
    upper = _box(0, 0, 100, 30, ((0.0, 5.0), (100.0, 5.0)), members=(1,))

    assert _cut_pair(lower, upper) is True
    assert lower.poly.bounds == (0.0, 30.5, 100.0, 61.0)
    assert upper.poly.bounds == (0.0, 0.0, 100.0, 30.0)


def test_cut_pair_keeps_the_far_side_for_a_centroid_on_the_cut() -> None:
    # The triangle's centroid (20, 0) lies on the mid-baseline line y 0, so
    # it keeps the y > 0 side: 400 of its 900 px, not more than half.
    triangle = Work(
        poly=Polygon([(0, -10), (60, -10), (0, 20)]),
        baseline=((0.0, -5.0), (40.0, -5.0)),
        score=0.9,
        members=(0,),
    )
    other = _box(0, -2, 40, 12, ((0.0, 5.0), (40.0, 5.0)), members=(1,))

    assert _cut_pair(triangle, other) is False
    assert (triangle.area, other.area) == (900.0, 560.0)


def _sliver_pair(left_loss: float, right_loss: float) -> tuple[Work, Work]:
    """Two 10 px wide lines whose cut takes ``left_loss`` and ``right_loss`` px."""
    cut = 10.0 - left_loss / 10
    upper = _box(0, 0, 10, 10, ((0.0, 5.0), (10.0, 5.0)))
    lower_baseline = ((0.0, 2 * cut - 5), (10.0, 2 * cut - 5))
    lower = _box(0, cut - right_loss / 10, 10, 20, lower_baseline, members=(1,))
    return upper, lower


@pytest.mark.parametrize(("left_loss", "right_loss"), [(1.5e-9, 0.5e-9), (0.5e-9, 1.5e-9)])
def test_cut_pair_counts_an_area_change_over_a_nanopixel_on_either_side(
    left_loss: float, right_loss: float
) -> None:
    upper, lower = _sliver_pair(left_loss, right_loss)
    before = (upper.area, lower.area)

    assert _cut_pair(upper, lower) is True
    assert before[0] - upper.area == pytest.approx(left_loss, rel=1e-3)
    assert before[1] - lower.area == pytest.approx(right_loss, rel=1e-3)


def test_cut_pair_with_short_baselines_still_cuts() -> None:
    upper = _box(0, 0, 100, 30, ((50.0, 20.0), (50.4, 20.0)))
    lower = _box(0, 20, 100, 50, ((50.0, 40.0), (50.4, 40.0)), members=(1,))

    assert _cut_pair(upper, lower) is True
    assert lower.poly.bounds == (0.0, 30.0, 100.0, 50.0)


def test_cut_pair_with_perpendicular_baselines_keeps_their_directions() -> None:
    def cut(lower_baseline: Baseline) -> tuple[float, float, Baseline, Baseline]:
        upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
        lower = _box(0, 20, 100, 50, lower_baseline, members=(1,))
        assert _cut_pair(upper, lower) is True
        return upper.area, lower.area, upper.baseline, lower.baseline

    # A zero dot product is not flipped: the mean direction is (100, 20),
    # against (100, -20) for the downward baseline.
    assert cut(((80.0, 30.0), (80.0, 50.0))) == (
        2577.5,
        2277.5,
        ((15.0, 20.0), (100.0, 20.0)),
        ((80.0, 33.0), (80.0, 50.0)),
    )
    assert cut(((80.0, 50.0), (80.0, 30.0))) == (
        2877.5,
        1700.0,
        ((0.0, 20.0), (100.0, 20.0)),
        ((80.0, 50.0), (80.0, 30.0)),
    )


def test_cut_pair_refuses_to_keep_exactly_half() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 10, 100, 40, ((0.0, 30.0), (100.0, 30.0)), members=(1,))

    assert _cut_pair(upper, lower) is False
    assert upper.poly.bounds == (0.0, 0.0, 100.0, 30.0)
    assert lower.poly.bounds == (0.0, 10.0, 100.0, 40.0)


def test_cut_pair_without_a_direction_does_nothing() -> None:
    upper = _box(0, 0, 100, 30, ((50.0, 20.0), (50.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((50.0, 40.0), (50.0, 40.0)), members=(1,))

    assert _cut_pair(upper, lower) is False
    assert lower.poly.bounds == (0.0, 20.0, 100.0, 50.0)


def test_cut_pair_of_lines_already_apart_reports_no_change() -> None:
    upper = _box(0, 0, 100, 20, ((0.0, 15.0), (100.0, 15.0)))
    lower = _box(0, 30, 100, 60, ((0.0, 45.0), (100.0, 45.0)), members=(1,))

    assert _cut_pair(upper, lower) is False
    assert (upper.area, lower.area) == (2000.0, 3000.0)


def test_candidate_pairs_are_touching_boxes_in_index_order() -> None:
    works = [
        _box(0, 0, 10, 10),
        _box(100, 0, 110, 10),
        _box(5, 5, 15, 15),
        _box(105, 5, 115, 15),
    ]

    assert _candidate_pairs(works, [0, 1, 2, 3], 0.2) == [(0, 2), (1, 3)]
    assert _candidate_pairs(works, [1, 2, 3], 0.2) == [(1, 3)]
    assert _candidate_pairs(works, [0, 2], 0.2) == [(0, 2)]
    assert _candidate_pairs(works, [3], 0.2) == []
    assert _candidate_pairs(works, [], 0.2) == []


def test_a_zero_threshold_scans_every_pair() -> None:
    works = [_box(0, 0, 10, 10), _box(100, 0, 110, 10), _box(200, 0, 210, 10)]

    assert _candidate_pairs(works, [0, 1, 2], 0.0) == [(0, 1), (0, 2), (1, 2)]
    assert _candidate_pairs(works, [2], 0.0) == []


def test_close_pair_with_an_initial_is_never_dropped() -> None:
    for initial_side in (0, 1):
        pair = [_box(0, 0, 10, 10, score=0.9), _box(0, 0, 10, 10, score=0.1, members=(1,))]
        pair[initial_side].role = "initial"

        assert _resolve_close_pair(pair[0], pair[1]) is False
        assert [(work.dropped, work.overlap_unresolved) for work in pair] == [(False, True)] * 2


def test_close_pair_sharing_half_of_the_larger_is_kept_and_flagged() -> None:
    left, right = _box(0, 0, 20, 10), _box(0, 0, 10, 10, members=(1,))

    assert _shared_over_larger(left, right) == 0.5
    assert _resolve_close_pair(left, right) is False
    assert (left.dropped, right.dropped) == (False, False)
    assert (left.overlap_unresolved, right.overlap_unresolved) == (True, True)


def test_close_pair_drops_the_lower_score() -> None:
    left, right = _box(0, 0, 10, 10, score=0.4), _box(0, 1, 10, 11, score=0.6, members=(1,))

    assert _resolve_close_pair(left, right) is True
    assert (left.dropped, right.dropped) == (True, False)
    assert (left.overlap_unresolved, right.overlap_unresolved) == (False, False)

    left, right = _box(0, 0, 10, 10, score=0.6), _box(0, 1, 10, 11, score=0.4, members=(1,))

    assert _resolve_close_pair(left, right) is True
    assert (left.dropped, right.dropped) == (False, True)


def test_close_pair_on_equal_scores_keeps_the_higher_members() -> None:
    left, right = _box(0, 0, 10, 10, members=(2,)), _box(0, 1, 10, 11, members=(5,))

    assert _resolve_close_pair(left, right) is True
    assert (left.dropped, right.dropped) == (True, False)

    left, right = _box(0, 0, 10, 10, members=(5,)), _box(0, 1, 10, 11, members=(2,))

    assert _resolve_close_pair(left, right) is True
    assert (left.dropped, right.dropped) == (False, True)


def test_resolve_pair_skips_dropped_works_and_low_shares() -> None:
    dropped, other = _box(0, 0, 10, 10), _box(0, 0, 10, 10, members=(1,))
    dropped.dropped = True

    assert _resolve_pair(dropped, other, 30.0, 0.2) is False
    assert _resolve_pair(other, dropped, 30.0, 0.2) is False
    assert other.overlap_unresolved is False

    left, right = _box(0, 0, 10, 10), _box(8, 0, 18, 10, members=(1,))

    # Shares 0.2 of the smaller polygon: just below 0.21, on 0.2.
    assert _resolve_pair(left, right, 30.0, 0.21) is False
    assert (left.overlap_unresolved, right.overlap_unresolved) == (False, False)
    assert _resolve_pair(left, right, 30.0, 0.2) is False
    assert (left.overlap_unresolved, right.overlap_unresolved) == (True, True)


def test_resolve_pair_measures_closeness_against_half_the_spacing() -> None:
    def pair() -> tuple[Work, Work]:
        upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
        lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,), score=0.5)
        return upper, lower

    # Baselines 20 apart: spacing 40 puts them exactly on the half, so stacked.
    upper, lower = pair()
    assert _resolve_pair(upper, lower, 40.0, 0.2) is True
    assert lower.poly.bounds == (0.0, 30.0, 100.0, 50.0)
    assert (upper.dropped, lower.dropped) == (False, False)

    # Spacing 41: closer than half, a close pair. Sharing a third of the
    # larger polygon, it is kept and flagged.
    upper, lower = pair()
    assert _resolve_pair(upper, lower, 41.0, 0.2) is False
    assert (upper.overlap_unresolved, lower.overlap_unresolved) == (True, True)
    assert lower.poly.bounds == (0.0, 20.0, 100.0, 50.0)


def test_resolve_pair_flags_a_stacked_pair_it_cannot_cut() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 10, 100, 40, ((0.0, 30.0), (100.0, 30.0)), members=(1,))

    assert _resolve_pair(upper, lower, 10.0, 0.2) is False
    assert (upper.overlap_unresolved, lower.overlap_unresolved) == (True, True)


def test_resolve_overlaps_falls_back_to_the_median_height_for_spacing() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,))
    one_row = layout_lines([_quad(0, 0, 100, 20)])

    # No row pitch on the page: spacing is the median height. 40 cuts (20 is
    # not under half of it); 41 makes the baselines close, a flagged pair.
    cut = resolve_overlaps([upper, lower], one_row, median_height=40.0)
    assert [work.poly.bounds for work in cut] == [(0.0, 0.0, 100.0, 30.0), (0.0, 30.0, 100.0, 50.0)]
    assert [work.overlap_unresolved for work in cut] == [False, False]

    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,))
    flagged = resolve_overlaps([upper, lower], one_row, median_height=41.0)
    assert [work.overlap_unresolved for work in flagged] == [True, True]


def test_resolve_overlaps_threshold_decides_what_is_touched() -> None:
    def pair() -> list[Work]:
        return [_box(0, 0, 10, 10, score=0.9), _box(8, 1, 18, 11, score=0.5, members=(1,))]

    # The pair shares 0.18 of the smaller polygon.
    untouched = resolve_overlaps(pair(), layout_lines([]), median_height=100.0)
    assert [work.overlap_unresolved for work in untouched] == [False, False]

    flagged = resolve_overlaps(pair(), layout_lines([]), cut_threshold=0.18, median_height=100.0)
    assert [work.overlap_unresolved for work in flagged] == [True, True]


def test_resolve_overlaps_drops_duplicates_and_returns_the_rest_in_order() -> None:
    works = [
        _box(0, 0, 100, 20, score=0.5),
        _box(0, 100, 100, 120, score=0.7, members=(1,)),
        _box(0, 1, 100, 21, score=0.8, members=(2,)),
    ]

    kept = resolve_overlaps(works, layout_lines([]), median_height=40.0)

    assert [work.members for work in kept] == [(1,), (2,)]
    assert works[0].dropped is True
    assert [work.overlap_unresolved for work in kept] == [False, False]


def test_resolve_overlaps_acts_on_a_share_equal_to_the_threshold() -> None:
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,))
    threshold = _shared_ratio(upper, lower)

    kept = resolve_overlaps(
        [upper, lower], layout_lines([]), cut_threshold=threshold, median_height=40.0
    )

    assert [work.poly.bounds for work in kept] == [
        (0.0, 0.0, 100.0, 30.0),
        (0.0, 30.0, 100.0, 50.0),
    ]
    assert [work.overlap_unresolved for work in kept] == [False, False]


def test_resolve_overlaps_default_spacing_is_one_pixel() -> None:
    """Baselines 0.75 apart are stacked against a 1 px spacing, not close.

    The cut would keep a third of the upper line, so both are flagged; as a
    close pair the near-identical lower one would have been dropped.
    """
    upper = _box(0, 0, 100, 30, ((0.0, 10.0), (100.0, 10.0)), score=0.9)
    lower = _box(0, 1, 100, 31, ((0.0, 10.75), (100.0, 10.75)), score=0.5, members=(1,))

    kept = resolve_overlaps([upper, lower], layout_lines([]))

    assert [(work.members, work.overlap_unresolved, work.dropped) for work in kept] == [
        ((0,), True, False),
        ((1,), True, False),
    ]


def test_resolve_overlaps_stops_after_the_first_pass_without_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = overlaps._overlap_pass

    def counted(works: list[Work], spacing: float, cut_threshold: float) -> bool:
        changed = original(works, spacing, cut_threshold)
        calls.append(changed)
        return changed

    monkeypatch.setattr(overlaps, "_overlap_pass", counted)
    upper = _box(0, 0, 100, 30, ((0.0, 20.0), (100.0, 20.0)))
    lower = _box(0, 20, 100, 50, ((0.0, 40.0), (100.0, 40.0)), members=(1,))

    resolve_overlaps([upper, lower], layout_lines([]), median_height=40.0)

    assert calls == [True, False]
    assert lower.poly.bounds == (0.0, 30.0, 100.0, 50.0)


def test_median_line_spacing_ignores_rows_with_the_same_centre() -> None:
    # Three nested lines centred on y 20 are separate rows with no pitch.
    quads = [
        _quad(0, 0, 100, 40),
        _quad(0, 10, 100, 30),
        _quad(0, 15, 100, 25),
        _quad(0, 50, 100, 70),
    ]
    layout = layout_lines(quads)

    assert layout.columns[0].rows == ((0,), (1,), (2,), (3,))
    assert _median_line_spacing(layout, 9.0) == 40.0


def test_cut_pair_along_a_forty_five_degree_line() -> None:
    upper = _box(0, 0, 20, 20, ((0.0, 0.0), (20.0, 20.0)))
    lower = _box(0, 10, 20, 30, ((0.0, 10.0), (20.0, 30.0)), members=(1,))

    assert _cut_pair(upper, lower) is True
    assert (upper.area, lower.area) == (287.5, 287.5)


def test_cut_pair_with_baselines_far_under_a_pixel_still_cuts() -> None:
    upper = _box(0, 0, 100, 30, ((50.0, 20.0), (50.0005, 20.0)))
    lower = _box(0, 20, 100, 50, ((50.0, 40.0), (50.0005, 40.0)), members=(1,))

    assert _cut_pair(upper, lower) is True
    assert lower.poly.bounds == (0.0, 30.0, 100.0, 50.0)


def test_resolve_overlaps_takes_the_largest_share_first() -> None:
    """Line 0 shares 0.9 with line 1 and 0.6 with line 2.

    Largest first, 0 drops against the higher-scored 1, and the 0.6 pair
    is then skipped, so 2 survives next to 1 (flagged: they share 0.5 of
    the larger). Smallest first would drop 2 against 0 and then 0 against 1.
    """
    works = [
        _box(0, 0, 100, 20, score=0.7),
        _box(0, 2, 100, 22, score=0.9, members=(1,)),
        _box(0, -8, 100, 12, score=0.5, members=(2,)),
    ]

    kept = resolve_overlaps(works, layout_lines([]), median_height=40.0)

    assert [(work.members, work.overlap_unresolved) for work in kept] == [
        ((1,), True),
        ((2,), True),
    ]
