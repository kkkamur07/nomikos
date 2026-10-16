"""Exact values from the shared refinement geometry.

Each helper is pinned on inputs that sit on its boundaries, so a changed
operator, index or constant shows up as a different number.
"""

from __future__ import annotations

import inspect

import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.geometry import (
    RefinedLine,
    Work,
    baseline_distance,
    bbox,
    is_initial,
    median,
    quad_height,
    synthetic_baseline_points,
    work_of_quad,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad

SKEWED = [[10.0, 20.0], [110.0, 30.0], [120.0, 70.0], [0.0, 60.0]]


def _work(baseline: tuple[tuple[float, float], tuple[float, float]]) -> Work:
    return Work(
        poly=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        baseline=baseline,
        score=0.5,
        members=(0,),
    )


def test_synthetic_baseline_interpolates_both_sides_of_a_skewed_quad() -> None:
    assert synthetic_baseline_points(SKEWED, 0.25) == [[7.5, 30.0], [112.5, 40.0]]
    assert synthetic_baseline_points(SKEWED, 0.0) == [[10.0, 20.0], [110.0, 30.0]]
    assert synthetic_baseline_points(SKEWED, 1.0) == [[0.0, 60.0], [120.0, 70.0]]


def test_bbox_and_quad_height_of_a_skewed_quad() -> None:
    assert bbox(SKEWED) == (0.0, 20.0, 120.0, 70.0)
    assert quad_height(DetectedQuad(points=SKEWED, score=0.5)) == 50.0


def test_median_is_the_upper_median_with_a_fallback() -> None:
    assert median([], 7.0) == 7.0
    assert median([4.0, 1.0, 3.0, 2.0], 0.0) == 3.0
    assert median([5.0, 1.0, 3.0], 0.0) == 3.0
    assert median([2.0], 0.0) == 2.0


def test_is_initial_compares_against_the_other_row_mates_only() -> None:
    heights = [40.0, 20.0, 20.0, 100.0]
    # The row median leaves the candidate out: mates are 20 and 20.
    assert is_initial(0, [0, 1, 2], heights, 1.0, 1.9) is True
    # Exactly the ratio is not taller than it.
    assert is_initial(0, [0, 1, 2], heights, 1.0, 2.0) is False
    # With the tall quad 3 in the row the upper median of 20, 20, 100 is 20.
    assert is_initial(0, [0, 1, 2, 3], heights, 1.0, 1.9) is True
    assert is_initial(1, [0, 1, 2, 3], heights, 1.0, 0.4) is True
    assert is_initial(1, [0, 1, 2, 3], heights, 1.0, 0.5) is False


def test_is_initial_falls_back_and_refuses_a_zero_reference() -> None:
    heights = [30.0]
    assert is_initial(0, [0], heights, 10.0, 2.9) is True
    assert is_initial(0, [0], heights, 10.0, 3.0) is False
    assert is_initial(0, [0], heights, 0.0, 0.5) is False
    assert is_initial(0, [0], heights, 0.5, 2.0) is True
    assert is_initial(0, (0,), [30.0, 0.0], 5.0, 0.5) is True


def test_work_of_quad_keeps_index_score_area_and_baseline() -> None:
    work = work_of_quad(4, DetectedQuad(points=SKEWED, score=0.625), 0.25)

    assert work.members == (4,)
    assert work.score == 0.625
    assert isinstance(work.score, float)
    assert work.baseline == ((7.5, 30.0), (112.5, 40.0))
    assert work.area == pytest.approx(Polygon(SKEWED).area)
    assert work.area == 4400.0
    assert (work.role, work.merged_from, work.dropped, work.suspect) == ("line", 1, False, False)
    assert (work.overlap_unresolved, work.suspect_reason) == (False, "")


def test_baseline_distance_flips_an_antiparallel_baseline() -> None:
    forward = _work(((0.0, 0.0), (10.0, 0.0)))
    backward = _work(((10.0, 4.0), (0.0, 4.0)))
    # Without the flip the directions cancel and the plain midpoint gap of
    # 4 would come out as well, so shift the second along x to tell them apart.
    shifted_back = _work(((13.0, 4.0), (3.0, 4.0)))

    assert baseline_distance(forward, backward) == 4.0
    assert baseline_distance(forward, shifted_back) == 4.0
    assert baseline_distance(shifted_back, forward) == 4.0


def test_baseline_distance_uses_the_mean_direction_of_tilted_baselines() -> None:
    first = _work(((0.0, 0.0), (10.0, 0.0)))
    second = _work(((3.0, 10.0), (13.0, 20.0)))
    # Mean direction (20, 10), normal (-10, 20)/sqrt(500); offset (3, 15).
    assert baseline_distance(first, second) == pytest.approx(270 / 500**0.5)


def test_short_baselines_under_one_pixel_still_have_a_direction() -> None:
    first = _work(((0.0, 0.0), (0.25, 0.0)))
    second = _work(((0.0, 3.0), (0.25, 3.0)))
    third = _work(((4.0, 3.0), (4.25, 3.0)))
    # Mean length 0.5: the perpendicular gap is 3, not the 5 of the midpoints.
    assert baseline_distance(first, second) == 3.0
    assert baseline_distance(first, third) == 3.0


def test_baseline_distance_of_zero_length_baselines_is_the_midpoint_gap() -> None:
    first = _work(((0.0, 0.0), (0.0, 0.0)))
    second = _work(((3.0, 4.0), (3.0, 4.0)))
    assert baseline_distance(first, second) == 5.0


def test_perpendicular_baselines_are_not_flipped() -> None:
    first = _work(((0.0, 0.0), (10.0, 0.0)))
    second = _work(((0.0, 5.0), (0.0, 15.0)))
    # Dot product 0: mean direction (10, 10), offset (-5, 10).
    assert baseline_distance(first, second) == pytest.approx(150 / 200**0.5)


def test_refined_line_and_work_defaults() -> None:
    line = RefinedLine(points=[[0.0, 0.0]], baseline=[[0.0, 0.0], [1.0, 0.0]], score=0.5)
    assert line.members == ()
    assert (line.role, line.suspect, line.suspect_reason) == ("line", False, "")
    assert (line.merged_from, line.overlap_unresolved) == (1, False)
    work = _work(((0.0, 0.0), (1.0, 0.0)))
    assert (work.role, work.merged_from, work.overlap_unresolved) == ("line", 1, False)
    assert (work.dropped, work.suspect, work.suspect_reason) == (False, False, "")
    assert work.area == 1.0


def test_work_area_is_derived_never_passed() -> None:
    assert "area" not in inspect.signature(Work).parameters
