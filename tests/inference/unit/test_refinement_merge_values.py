"""Exact values from the row-fragment merge stage.

The gap, overlap and height thresholds are each pinned at the boundary and
just past it, and the merged polygon, baseline and score are compared as
numbers rather than as properties.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.geometry import work_of_quad
from nomikos_inference.architectures.ppocr_det.merge import (
    DEFAULT_MERGE_GAP_RATIO,
    DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    _bridge,
    _merge_run,
    _merged_polygon,
    _row_runs,
    mark_initials,
    merge_row_fragments,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _points(quads: list[DetectedQuad]) -> list[list[list[float]]]:
    return [[[float(x), float(y)] for x, y in quad.points] for quad in quads]


def _runs(quads: list[DetectedQuad], row: tuple[int, ...], **options: float) -> list[list[int]]:
    settings = {"gap_min": -2.0, "gap_max": 5.0, "height_ratio": 2.0, **options}
    return list(_row_runs(row, quads, **settings))


def test_default_thresholds() -> None:
    assert DEFAULT_MERGE_GAP_RATIO == 0.25
    assert DEFAULT_MERGE_MAX_OVERLAP_RATIO == 0.1
    assert DEFAULT_MERGE_MAX_HEIGHT_RATIO == 2.0


def test_bridge_spans_the_gap_inside_the_shared_y_range() -> None:
    prev = work_of_quad(0, _quad(0, 0, 10, 20), 0.75)
    nxt = work_of_quad(1, _quad(13, 5, 30, 25), 0.75)

    bridge = _bridge(prev, nxt)

    assert bridge is not None
    assert list(bridge.exterior.coords) == [(10, 5), (30, 5), (30, 20), (10, 20), (10, 5)]


def test_no_bridge_when_the_y_ranges_only_touch_or_miss() -> None:
    prev = work_of_quad(0, _quad(0, 0, 10, 20), 0.75)
    touching = work_of_quad(1, _quad(13, 20, 30, 40), 0.75)
    apart = work_of_quad(2, _quad(13, 21, 30, 40), 0.75)

    assert _bridge(prev, touching) is None
    assert _bridge(prev, apart) is None


def test_merged_polygon_unions_touching_parts() -> None:
    merged = _merged_polygon(
        [Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]), Polygon([(2, 0), (4, 0), (4, 1), (2, 1)])]
    )

    assert merged.area == 4.0
    assert merged.bounds == (0.0, 0.0, 4.0, 1.0)


def test_merged_polygon_of_disjoint_parts_is_their_hull() -> None:
    merged = _merged_polygon(
        [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])]
    )

    assert merged.area == 4.0
    assert merged.bounds == (0.0, 0.0, 4.0, 1.0)


def test_merged_polygon_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match=r"^merge produced no polygon$"):
        _merged_polygon([])


def test_merge_run_unions_members_with_an_area_weighted_score() -> None:
    quads = [_quad(0, 0, 10, 20, score=0.8), _quad(13, 5, 30, 25, score=0.4)]

    work = _merge_run([1, 0], quads, _points(quads), 0.75)

    # 200 px at 0.8 and 340 px at 0.4, plus the 17 x 15 bridge.
    assert work.score == pytest.approx((0.8 * 200 + 0.4 * 340) / 540)
    assert work.poly.area == 200 + 340 + 3 * 15
    assert work.members == (0, 1)
    assert work.merged_from == 2
    assert work.baseline == ((0.0, 15.0), (30.0, 20.0))
    assert work.role == "line"


def test_merge_run_of_flat_members_scores_zero() -> None:
    # Two diagonal quads with no area, joined by a real 12 x 10 bridge.
    quads = [
        DetectedQuad(points=[[0, 0], [10, 10], [10, 10], [0, 0]], score=0.5),
        DetectedQuad(points=[[12, 0], [22, 10], [22, 10], [12, 0]], score=0.75),
    ]

    work = _merge_run([0, 1], quads, _points(quads), 0.75)

    assert work.score == 0.0
    assert work.poly.bounds == (10.0, 0.0, 22.0, 10.0)
    assert work.area == 120.0


def test_merge_run_takes_the_baseline_from_the_outer_ends() -> None:
    # The left member reaches furthest left, the middle one is tallest, the
    # right member reaches furthest right.
    quads = [_quad(0, 10, 20, 30), _quad(22, 0, 40, 40), _quad(42, 20, 60, 40)]

    work = _merge_run([2, 0, 1], quads, _points(quads), 0.5)

    assert work.baseline == ((0.0, 20.0), (60.0, 30.0))
    assert work.members == (0, 1, 2)
    assert work.merged_from == 3


def test_merge_run_baseline_ends_are_the_outer_points_of_skewed_quads() -> None:
    quads = [
        DetectedQuad(points=[[0, 10], [10, 0], [10, 20], [0, 30]], score=0.5),
        DetectedQuad(points=[[20, 10], [30, 0], [30, 20], [20, 30]], score=0.5),
    ]

    work = _merge_run([0, 1], quads, _points(quads), 0.5)

    # The outer ends, not the lowest or highest baseline points.
    assert work.baseline == ((0.0, 20.0), (30.0, 10.0))


def test_merge_run_breaks_a_left_edge_tie_by_the_top() -> None:
    quads = [_quad(0, 10, 10, 30), _quad(0, 0, 12, 20)]

    work = _merge_run([0, 1], quads, _points(quads), 0.5)

    assert work.baseline == ((0.0, 10.0), (12.0, 10.0))


def test_merge_run_breaks_a_right_edge_tie_by_the_top() -> None:
    quads = [_quad(0, 0, 20, 40), _quad(5, 10, 20, 30)]

    work = _merge_run([0, 1], quads, _points(quads), 0.25)

    assert work.baseline == ((0.0, 10.0), (20.0, 15.0))


def test_merge_run_of_members_under_one_pixel_still_weighs_scores() -> None:
    quads = [_quad(0, 0, 0.5, 1, score=0.8), _quad(0.5, 0, 1, 1, score=0.4)]

    work = _merge_run([0, 1], quads, _points(quads), 0.5)

    assert work.score == pytest.approx(0.6)


def test_row_runs_include_the_gap_bounds_and_exclude_past_them() -> None:
    quads = [
        _quad(0, 0, 10, 20),
        _quad(15, 0, 25, 20),  # gap 5: the upper bound
        _quad(23, 0, 33, 20),  # gap -2: the lower bound
        _quad(38.5, 0, 48, 20),  # gap 5.5: a new run
        _quad(45.5, 0, 55, 20),  # gap -2.5: a new run
    ]

    assert _runs(quads, (4, 3, 2, 1, 0)) == [[0, 1, 2], [3], [4]]


def test_row_runs_keep_the_widest_right_edge() -> None:
    quads = [_quad(0, 0, 100, 20), _quad(98, 0, 99, 20), _quad(105, 0, 120, 20)]

    # The contained fragment must not pull the run's right edge back to 99.
    assert _runs(quads, (0, 1, 2)) == [[0, 1, 2]]


def test_row_runs_include_the_height_ratio_bound() -> None:
    quads = [
        _quad(0, 100, 10, 110),
        _quad(12, 100, 22, 120),  # 20 <= 2 * 10
        _quad(24, 100, 34, 121),  # 21 > 2 * 10
    ]

    assert _runs(quads, (0, 1, 2)) == [[0, 1], [2]]
    assert _runs(quads, (0, 1, 2), height_ratio=2.1) == [[0, 1, 2]]


def test_row_runs_track_the_shortest_member_after_a_reset() -> None:
    quads = [
        _quad(0, 0, 10, 40),
        _quad(30, 0, 40, 10),  # a new run by its gap
        _quad(42, 0, 52, 20),  # 20 <= 2 * 10, joins
        _quad(54, 0, 64, 25),  # 25 > 2 * 10, but not > 2 * 40
    ]

    assert _runs(quads, (0, 1, 2, 3)) == [[0], [1, 2], [3]]


def test_row_runs_order_ties_by_top_then_coordinates() -> None:
    quads = [_quad(0, 30, 10, 50), _quad(0, 0, 10, 20), _quad(0, 0, 10, 21)]

    assert _runs(quads, (0, 1, 2), gap_min=-100.0) == [[1, 2, 0]]


def test_row_runs_read_left_to_right_whatever_the_tops() -> None:
    quads = [_quad(0, 5, 10, 25), _quad(12, 0, 22, 20)]

    assert _runs(quads, (1, 0)) == [[0, 1]]


def test_row_runs_break_a_left_edge_tie_by_the_top_not_the_right_edge() -> None:
    quads = [_quad(0, 0, 10, 20), _quad(0, 30, 8, 50)]

    assert _runs(quads, (1, 0), gap_min=-100.0) == [[0, 1]]


def test_row_runs_treat_coordinates_equal_to_six_places_as_a_tie() -> None:
    nudged = DetectedQuad(points=[[0, 0], [10, 0], [10, 20.0000001], [0, 20.0000001]], score=0.5)
    quads = [nudged, _quad(0, 0, 10, 20)]

    assert _runs(quads, (0, 1), gap_min=-100.0) == [[0, 1]]
    assert _runs(quads, (1, 0), gap_min=-100.0) == [[1, 0]]


def test_row_runs_of_an_empty_row_yield_nothing() -> None:
    assert _runs([], ()) == []


def _page(*extra: DetectedQuad) -> list[DetectedQuad]:
    """A 20 px tall column of builders; the median quad height is 20."""
    return [_quad(0, y, 200, y + 20) for y in (0, 40, 80, 160, 200)] + list(extra)


def _merged_members(quads: list[DetectedQuad], **options: float) -> list[tuple[int, ...]]:
    works = merge_row_fragments(quads, layout_lines(quads), **options)
    return sorted(work.members for work in works)


def test_row_mates_merge_up_to_a_quarter_line_gap() -> None:
    at_bound = _page(_quad(0, 120, 100, 140), _quad(105, 120, 200, 140))
    past_bound = _page(_quad(0, 120, 100, 140), _quad(105.5, 120, 200, 140))

    assert (5, 6) in _merged_members(at_bound)
    assert (5, 6) not in _merged_members(past_bound)
    assert (5, 6) in _merged_members(past_bound, gap_ratio=0.3)


def test_row_mates_merge_up_to_a_tenth_line_overlap() -> None:
    at_bound = _page(_quad(0, 120, 100, 140), _quad(98, 120, 200, 140))
    past_bound = _page(_quad(0, 120, 100, 140), _quad(97.5, 120, 200, 140))

    assert (5, 6) in _merged_members(at_bound)
    assert (5, 6) not in _merged_members(past_bound)
    assert (5, 6) in _merged_members(past_bound, max_overlap_ratio=0.2)


def test_merge_passes_the_baseline_fraction_through() -> None:
    quads = _page(_quad(0, 120, 100, 140), _quad(103, 120, 200, 140))

    works = merge_row_fragments(quads, layout_lines(quads), baseline_fraction=0.5)
    merged = next(work for work in works if work.members == (5, 6))
    lone = next(work for work in works if work.members == (0,))

    assert merged.baseline == ((0.0, 130.0), (200.0, 130.0))
    assert lone.baseline == ((0.0, 10.0), (200.0, 10.0))


def test_merge_height_ratio_is_configurable() -> None:
    quads = _page(_quad(0, 115, 100, 145), _quad(103, 120, 200, 140))

    default = merge_row_fragments(quads, layout_lines(quads))
    strict = merge_row_fragments(quads, layout_lines(quads), height_ratio=1.4)

    assert (5, 6) in sorted(work.members for work in default)
    assert (5, 6) not in sorted(work.members for work in strict)
    tall = next(work for work in strict if work.members == (5,))
    assert tall.role == "initial"


def test_spanning_quads_pass_through_in_index_order() -> None:
    left = [_quad(0, y, 100, y + 20) for y in (40, 80, 120)]
    right = [_quad(300, y, 400, y + 20) for y in (40, 80, 120)]
    heads = [_quad(0, 200, 400, 220, score=0.3), _quad(0, 0, 400, 20, score=0.4)]
    quads = left + right + heads

    works = merge_row_fragments(quads, layout_lines(quads))

    assert [work.members for work in works[-2:]] == [(6,), (7,)]
    assert [work.score for work in works[-2:]] == [0.3, 0.4]
    assert len(works) == 8


def test_a_lone_tall_quad_is_an_initial_against_the_builder_median() -> None:
    quads = _page(_quad(0, 110, 30, 155))

    works = merge_row_fragments(quads, layout_lines(quads))

    tall = next(work for work in works if work.members == (5,))
    assert tall.role == "initial"
    assert [work.role for work in works if work.members != (5,)] == ["line"] * 5


def test_mark_initials_uses_the_builder_median_and_the_ratio() -> None:
    quads = _page(_quad(0, 110, 30, 150.5))
    layout = layout_lines(quads)

    loose = [work_of_quad(i, quad, 0.75) for i, quad in enumerate(quads)]
    mark_initials(loose, quads, layout, 2.0)
    strict = [work_of_quad(i, quad, 0.75) for i, quad in enumerate(quads)]
    mark_initials(strict, quads, layout, 2.1)

    assert [work.role for work in loose] == ["line"] * 5 + ["initial"]
    assert [work.role for work in strict] == ["line"] * 6


def test_a_row_keeps_its_runs_after_a_merged_one() -> None:
    quads = _page(_quad(0, 120, 60, 140), _quad(63, 120, 120, 140), _quad(150, 120, 200, 140))

    assert _merged_members(quads) == [(0,), (1,), (2,), (3,), (4,), (5, 6), (7,)]
