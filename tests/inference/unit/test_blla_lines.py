"""BLLA baseline vectorization and reading order on synthetic heatmaps.

The decoder is a port of kraken's; these pin the behaviours a page depends
on: which way a baseline points, which skeleton endpoints survive the budget,
and the order lines and regions are read in.
"""

from __future__ import annotations

import numpy as np

from nomikos_inference.architectures.blla.blla_decoder.lines import (
    reading_order_indices,
    vectorize_lines,
    vectorize_regions,
)


def _ridges(*rows: tuple[int, int, int], height: int = 60, width: int = 120) -> np.ndarray:
    """Start, end and baseline channels with a 3 px baseline ridge per row."""
    channels = np.zeros((3, height, width), dtype=np.float32)
    for y, x_start, x_stop in rows:
        channels[2, y - 1 : y + 2, x_start:x_stop] = 1.0
    return channels


def _vectorize(channels: np.ndarray, **overrides: float) -> list[list[list[int]]]:
    options = {"threshold": 0.17, "min_length": 5.0, **overrides}
    return vectorize_lines(channels, **options)


def test_a_ridge_becomes_one_left_to_right_baseline_in_xy() -> None:
    assert _vectorize(_ridges((30, 10, 110))) == [[[10, 30], [108, 29]]]


def test_start_and_end_marks_decide_the_direction() -> None:
    right_to_left = _ridges((30, 10, 110))
    right_to_left[0, 25:35, 100:115] = 1.0
    right_to_left[1, 25:35, 5:20] = 1.0
    left_to_right = _ridges((30, 10, 110))
    left_to_right[0, 25:35, 5:20] = 1.0
    left_to_right[1, 25:35, 100:115] = 1.0

    assert _vectorize(right_to_left) == [[[108, 29], [10, 30]]]
    assert _vectorize(left_to_right) == [[[10, 30], [108, 29]]]


def test_marks_that_disagree_fall_back_to_left_to_right() -> None:
    both_starts = _ridges((30, 10, 110))
    both_starts[0, 25:35, 5:20] = 1.0
    both_starts[0, 25:35, 100:115] = 1.0

    assert _vectorize(both_starts) == [[[10, 30], [108, 29]]]


def test_lines_shorter_than_min_length_are_dropped() -> None:
    assert _vectorize(_ridges((30, 10, 110)), min_length=98.0) == [[[10, 30], [108, 29]]]
    assert _vectorize(_ridges((30, 10, 110)), min_length=99.0) == []


def test_a_map_without_two_endpoints_has_no_lines() -> None:
    assert _vectorize(np.zeros((3, 60, 120), dtype=np.float32)) == []


def test_every_line_survives_when_the_endpoints_fit_the_budget() -> None:
    lines = _vectorize(_ridges((15, 10, 110), (45, 10, 60)))

    assert sorted(lines) == [[[10, 15], [108, 14]], [[10, 45], [58, 44]]]


def test_over_budget_keeps_the_largest_lines_that_fit() -> None:
    """Two lines, four endpoints, a budget of two: the longer line stays."""
    lines = _vectorize(_ridges((15, 10, 110), (45, 10, 60)), max_endpoints=2)

    assert lines == [[[10, 15], [108, 14]]]


def test_a_comb_of_endpoints_is_noise_and_drops_out_over_budget() -> None:
    channels = _ridges((60, 10, 110), height=80)
    for x in range(10, 110, 6):
        channels[2, 10:40, x] = 1.0
    channels[2, 25, 10:110] = 1.0

    within_budget = _vectorize(channels)
    over_budget = _vectorize(channels, max_endpoints=3)

    # Inside the budget the comb's teeth are all traced as lines of their own.
    assert len(within_budget) > 10
    assert over_budget == [[[10, 60], [108, 59]]]


def test_lines_read_top_to_bottom_in_a_column_then_across() -> None:
    lower_left = [[10, 50], [100, 50]]
    upper_left = [[10, 10], [100, 10]]
    right = [[150, 30], [250, 30]]

    assert reading_order_indices([lower_left, upper_left, right]) == [1, 0, 2]


def test_a_region_reads_as_one_block_in_its_own_order() -> None:
    left = [[10, 30], [100, 30]]
    region_lower = [[160, 40], [240, 40]]
    region_upper = [[160, 20], [240, 20]]
    region = [[140, 0], [260, 0], [260, 60], [140, 60]]
    far_right = [[300, 5], [350, 5]]

    order = reading_order_indices([left, region_lower, region_upper, far_right], [region])

    assert order == [0, 2, 1, 3]


def test_regions_without_lines_are_ignored() -> None:
    upper = [[10, 10], [100, 10]]
    lower = [[10, 50], [100, 50]]
    empty_region = [[300, 300], [400, 300], [400, 400], [300, 400]]

    assert reading_order_indices([lower, upper], [empty_region]) == [1, 0]


def test_a_region_map_becomes_its_outline_in_xy() -> None:
    probability = np.zeros((60, 120), dtype=np.float32)
    probability[10:40, 20:90] = 1.0

    assert vectorize_regions(probability) == [[[20, 10], [89, 10], [89, 39], [20, 39], [20, 10]]]


def test_separate_regions_stay_separate() -> None:
    probability = np.zeros((60, 120), dtype=np.float32)
    probability[10:40, 20:90] = 1.0
    probability[45:55, 100:115] = 1.0

    outlines = vectorize_regions(probability)

    assert len(outlines) == 2
    assert outlines[1] == [[101, 45], [114, 45], [114, 54], [100, 54], [101, 45]]


def test_a_stray_pixel_is_not_a_region() -> None:
    probability = np.zeros((60, 120), dtype=np.float32)
    probability[5, 5] = 1.0

    assert vectorize_regions(probability) == []
