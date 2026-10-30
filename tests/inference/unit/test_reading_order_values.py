"""Exact values from the PP-OCR reading-order helpers.

Each fraction is pinned at its boundary and just past it: the narrow and
wide width classes, the column and row overlaps, the spanning share and the
tall-quad height. Layout fields are compared whole.
"""

from __future__ import annotations

import dataclasses

import pytest

from nomikos_inference.architectures.ppocr_det import reading_order
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import (
    _attach_wide,
    _builder_components,
    _extents,
    _nearest_column,
    _overlap,
    _place_in_rows,
    _row_groups,
    _shares_row,
    _width_classes,
    layout_lines,
    order_lines,
)


def _quad(xmin: float, ymin: float, xmax: float, ymax: float) -> DetectedQuad:
    return DetectedQuad(points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], score=0.9)


def _ext(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, ...]:
    return _extents([_quad(xmin, ymin, xmax, ymax)])[0]


def test_fractions() -> None:
    assert reading_order.NARROW_WIDTH_FRACTION == 0.5
    assert reading_order.COLUMN_OVERLAP_FRACTION == 0.5
    assert reading_order.WIDE_WIDTH_FRACTION == 1.5
    assert reading_order.SPANNING_OVERLAP_FRACTION == 0.10
    assert reading_order.ROW_OVERLAP_FRACTION == 0.5
    assert reading_order.TALL_HEIGHT_FRACTION == 1.8


def test_extents_of_a_skewed_quad() -> None:
    quad = DetectedQuad(points=[[10, 20], [110, 30], [120, 70], [0, 60]], score=0.9)

    assert _extents([quad]) == [(0, 120, 60.0, 45.0, 120, 20, 70, 50)]


def test_overlap_is_signed() -> None:
    assert _overlap(0, 10, 5, 20) == 5
    assert _overlap(5, 20, 0, 10) == 5
    assert _overlap(0, 10, 15, 20) == -5


def test_nearest_column_measures_the_gap_on_either_side() -> None:
    ranges = [(0.0, 100.0), (200.0, 300.0), (400.0, 500.0)]

    assert _nearest_column(ranges, 120, 130) == 0
    assert _nearest_column(ranges, 170, 180) == 1
    assert _nearest_column(ranges, 320, 330) == 1
    assert _nearest_column(ranges, 600, 610) == 2
    assert _nearest_column(ranges, 250, 260) == 1
    # Equal gaps: the first column wins.
    assert _nearest_column(ranges, 145, 155) == 0
    assert _nearest_column([], 0, 1) == 0


def test_shares_row_needs_half_the_shorter_height() -> None:
    row = [0.0, 20.0, 0.0, 100.0]

    assert _shares_row(_ext(110, 10, 200, 30), row) is True
    assert _shares_row(_ext(110, 10.5, 200, 30.5), row) is False
    # The shorter is the quad here: 10 tall, 5 shared.
    assert _shares_row(_ext(110, 15, 200, 25), row) is True
    assert _shares_row(_ext(110, 15.5, 200, 25.5), row) is False


def test_shares_row_refuses_quads_stacked_in_the_same_x_range() -> None:
    row = [0.0, 20.0, 0.0, 100.0]

    # Narrower width 50: an x overlap of 25 is stacked, 24 is side by side.
    assert _shares_row(_ext(75, 0, 125, 20), row) is False
    assert _shares_row(_ext(76, 0, 126, 20), row) is True
    # The narrower is the row here: 40 wide, 20 shared is stacked.
    assert _shares_row(_ext(80, 0, 300, 20), [0.0, 20.0, 60.0, 100.0]) is False
    assert _shares_row(_ext(81, 0, 300, 20), [0.0, 20.0, 60.0, 100.0]) is True


def test_place_in_rows_extends_the_last_row_only() -> None:
    rows: list[list[int]] = [[0], [1]]
    ranges = [[0.0, 20.0, 0.0, 100.0], [40.0, 60.0, 0.0, 100.0]]

    _place_in_rows(2, _ext(110, 45, 200, 65), False, rows, ranges)
    _place_in_rows(3, _ext(110, 5, 200, 25), False, rows, ranges)

    assert rows == [[0], [1, 2], [3]]
    assert ranges == [
        [0.0, 20.0, 0.0, 100.0],
        [40.0, 65.0, 0.0, 200.0],
        [5.0, 25.0, 110.0, 200.0],
    ]


def test_place_in_rows_puts_a_tall_quad_in_the_first_row_it_shares() -> None:
    rows: list[list[int]] = [[0], [1]]
    ranges = [[0.0, 20.0, 0.0, 100.0], [30.0, 50.0, 0.0, 100.0]]

    _place_in_rows(2, _ext(110, 0, 140, 60), True, rows, ranges)

    assert rows == [[0, 2], [1]]
    assert ranges == [[0.0, 20.0, 0.0, 100.0], [30.0, 50.0, 0.0, 100.0]]


def test_place_in_rows_opens_a_row_for_a_tall_quad_sharing_none() -> None:
    rows: list[list[int]] = [[0]]
    ranges = [[0.0, 20.0, 0.0, 100.0]]

    _place_in_rows(1, _ext(0, 100, 30, 200), True, rows, ranges)

    assert rows == [[0], [1]]
    assert ranges == [[0.0, 20.0, 0.0, 100.0], [100, 200, 0, 30]]


def test_place_in_rows_opens_the_first_row() -> None:
    rows: list[list[int]] = []
    ranges: list[list[float]] = []

    _place_in_rows(4, _ext(1, 2, 3, 5), False, rows, ranges)

    assert (rows, ranges) == ([[4]], [[2, 5, 1, 3]])


def _groups(
    quads: list[DetectedQuad], builders: set[int], direction: str = "ltr"
) -> list[list[int]]:
    extents = _extents(quads)
    geoms = [tuple(float(c) for point in quad.points for c in point) for quad in quads]
    return _row_groups(list(range(len(quads))), extents, geoms, builders, direction)


LINE = _quad(100, 0, 200, 20)
# Side by side with LINE but below its 20 px: only a row stretched by a
# taller mate reaches it.
LOWER_MATE = _quad(250, 22, 300, 42)


def test_row_groups_treat_exactly_the_tall_fraction_as_ordinary() -> None:
    # 36 = 1.8 * 20 is not tall: it stretches the row down to 36 and the
    # lower mate joins. 36.5 is tall: the row stays 0 to 20.
    assert _groups([LINE, LOWER_MATE, _quad(0, 0, 50, 36)], {0, 1}) == [[2, 0, 1]]
    assert _groups([LINE, LOWER_MATE, _quad(0, 0, 50, 36.5)], {0, 1}) == [[2, 0], [1]]


def test_row_groups_use_the_builders_median_height() -> None:
    tall = _quad(0, 0, 50, 36.5)
    giant = _quad(500, 0, 550, 700)

    # Builders 0 and 1 have median 20: the giant non-builder changes nothing
    # and, tall itself, joins the first row it shares.
    assert _groups([LINE, LOWER_MATE, tall, giant], {0, 1}) == [[2, 0, 3], [1]]
    # Builders 0 and 2 have upper median 36.5: the same quad is not tall.
    assert _groups([LINE, LOWER_MATE, tall], {0, 2}) == [[2, 0, 1]]


def test_row_groups_run_right_to_left_for_rtl() -> None:
    quads = [_quad(0, 0, 90, 20), _quad(100, 0, 190, 20), _quad(200, 0, 290, 20)]

    assert _groups(quads, {0, 1, 2}) == [[0, 1, 2]]
    assert _groups(quads, {0, 1, 2}, "rtl") == [[2, 1, 0]]


def test_row_groups_rank_rows_by_their_top_centre() -> None:
    # The second row's first quad sits higher than the first row's centre.
    quads = [_quad(0, 50, 90, 70), _quad(0, 0, 90, 20), _quad(100, 10, 190, 30)]

    assert _groups(quads, {0, 1, 2}) == [[1, 2], [0]]


def test_width_classes_on_the_boundaries() -> None:
    widths = [100.0, 100.0, 100.0, 50.0, 49.0, 150.0, 151.0]
    extents = [_ext(0, 10 * i, width, 10 * i + 5) for i, width in enumerate(widths)]

    assert _width_classes(extents) == ([4], [6], [0, 1, 2, 3, 5])


def test_width_classes_take_the_upper_median() -> None:
    extents = [_ext(0, 0, width, 5) for width in (10.0, 20.0, 31.0, 40.0)]

    # Upper median 31: 10 < 15.5 is narrow, the lower median 20 would not
    # make it narrow (10 is not under 10).
    assert _width_classes(extents) == ([0], [], [1, 2, 3])


def test_width_classes_of_zero_widths_are_all_builders() -> None:
    extents = [_ext(5, 0, 5, 10), _ext(5, 20, 5, 30), _ext(0, 40, 100, 50)]

    assert _width_classes(extents) == ([], [], [0, 1, 2])


def test_builder_components_join_at_half_the_narrower_width() -> None:
    extents = [_ext(0, 0, 100, 10), _ext(50, 20, 150, 30), _ext(101, 40, 201, 50)]

    # 0-1 share 50 of 100 (joined); 1-2 share 49 (not); 0-2 none.
    assert _builder_components([0, 1, 2], extents) == [[0, 1], [2]]


def test_builder_components_join_transitively_in_builder_order() -> None:
    extents = [
        _ext(0, 0, 100, 10),
        _ext(150, 0, 250, 10),
        _ext(50, 20, 150, 30),
        _ext(100, 40, 200, 50),
    ]

    assert _builder_components([3, 1, 0, 2], extents) == [[3, 1, 0, 2]]
    assert _builder_components([1, 0], extents) == [[1], [0]]
    assert _builder_components([], extents) == []


def test_attach_wide_joins_one_column_and_widens_it() -> None:
    columns = [[0], [1]]
    ranges = [(0.0, 100.0), (300.0, 400.0)]
    extents = [
        _ext(0, 0, 100, 10),
        _ext(300, 0, 400, 10),
        _ext(-50, 20, 160, 30),
        _ext(150, 40, 360, 50),
    ]

    spanning = _attach_wide([2, 3], columns, ranges, extents)

    # Quad 2 widens column 0 to (-50, 160). Quad 3 then shares 10 of 210
    # with it (not more than a tenth) and 60 of 100 with column 1.
    assert spanning == []
    assert columns == [[0, 2], [1, 3]]
    assert ranges == [(-50, 160), (150, 400)]


def test_attach_wide_spans_two_columns_past_a_tenth_of_each() -> None:
    extents = [
        _ext(0, 0, 100, 10),
        _ext(300, 0, 400, 10),
        _ext(89, 20, 311, 30),
        _ext(90, 40, 310, 50),
    ]
    columns = [[0], [1]]
    ranges = [(0.0, 100.0), (300.0, 400.0)]

    spanning = _attach_wide([2, 3], columns, ranges, extents)

    # Quad 2 shares 11 of each column; quad 3 exactly 10, which is not more.
    assert spanning == [2]
    assert columns == [[0, 3], [1]]
    assert ranges == [(0.0, 310), (300.0, 400.0)]


def test_attach_wide_without_overlap_joins_the_nearest_column() -> None:
    extents = [_ext(0, 0, 100, 10), _ext(300, 0, 400, 10), _ext(420, 20, 700, 30)]
    columns = [[0], [1]]
    ranges = [(0.0, 100.0), (300.0, 400.0)]

    assert _attach_wide([2], columns, ranges, extents) == []
    assert columns == [[0], [1, 2]]
    assert ranges == [(0.0, 100.0), (300.0, 700)]


def _spread() -> list[DetectedQuad]:
    left = [_quad(0, y, 100, y + 20) for y in (40, 80)]
    right = [_quad(300, y, 400, y + 20) for y in (40, 80)]
    return left + right + [_quad(0, 0, 400, 20), _quad(110, 60, 120, 70)]


def test_layout_of_a_spread_with_a_header_and_a_numeral() -> None:
    layout = layout_lines(_spread())

    assert [column.members for column in layout.columns] == [(0, 1, 5), (2, 3)]
    assert [column.builders for column in layout.columns] == [(0, 1), (2, 3)]
    assert [column.rows for column in layout.columns] == [((0,), (5,), (1,)), ((2,), (3,))]
    assert [column.sequence for column in layout.columns] == [(0, 5, 1), (2, 3)]
    assert [(column.xmin, column.xmax) for column in layout.columns] == [(0, 100), (300, 400)]
    assert layout.column_order == (0, 1)
    assert layout.spanning == (4,)
    assert layout.direction == "ltr"
    assert layout.centre_y == (50.0, 90.0, 50.0, 90.0, 10.0, 65.0)


def test_rtl_layout_reverses_the_column_order_only() -> None:
    layout = layout_lines(_spread(), direction="rtl")

    assert layout.column_order == (1, 0)
    assert [column.members for column in layout.columns] == [(0, 1, 5), (2, 3)]
    assert layout.direction == "rtl"


def test_spanning_quads_are_sorted_top_to_bottom() -> None:
    quads = _spread() + [_quad(0, 200, 400, 220), _quad(0, 120, 400, 140)]

    assert layout_lines(quads).spanning == (4, 7, 6)


def test_order_lines_puts_a_line_level_with_a_band_before_it() -> None:
    left = [_quad(0, y, 100, y + 20) for y in (0, 40)]
    right = [_quad(300, y, 400, y + 20) for y in (0, 40)]
    # The header's centre y 50 equals lines 1 and 3: both read before it.
    header = _quad(0, 45, 400, 55)

    assert order_lines(left + right + [header]) == [0, 1, 2, 3, 4]
    assert order_lines(left + right + [header], direction="rtl") == [2, 3, 0, 1, 4]


def test_order_lines_of_two_bands_and_nothing_between() -> None:
    column = [_quad(0, y, 100, y + 20) for y in (100, 140)]
    other = [_quad(300, y, 400, y + 20) for y in (100, 140)]
    heads = [_quad(0, 0, 400, 20), _quad(0, 40, 400, 60)]

    assert order_lines(column + other + heads) == [4, 5, 0, 1, 2, 3]


def test_bad_direction_message() -> None:
    with pytest.raises(ValueError, match=r'^reading direction must be "ltr" or "rtl"$'):
        layout_lines([], direction="ttb")


def test_nearest_column_prefers_an_overlapped_column_to_one_under_a_pixel_away() -> None:
    assert _nearest_column([(0.0, 100.0), (200.0, 300.0)], 100.5, 250) == 1


def test_place_in_rows_lowers_the_row_top_and_left_edge() -> None:
    rows: list[list[int]] = [[0]]
    ranges = [[10.0, 30.0, 300.0, 400.0]]

    _place_in_rows(1, _ext(410, 5, 500, 35), False, rows, ranges)
    _place_in_rows(2, _ext(200, 6, 290, 34), False, rows, ranges)

    assert rows == [[0, 1, 2]]
    assert ranges == [[5, 35, 200, 500]]


FLAT_LEFT = _quad(0, 10, 100, 10)
FLAT_RIGHT = _quad(210, 18, 300, 18)


def test_row_groups_with_flat_builders_have_no_tall_quads() -> None:
    # Median builder height 0: nothing is tall, so the 20 px quad stretches
    # the row down to 20 and the flat right line at y 18 joins it.
    assert _groups([FLAT_LEFT, FLAT_RIGHT, _quad(110, 0, 200, 20)], {0, 1}) == [[0, 2, 1]]


def test_row_groups_with_builders_under_a_pixel_still_find_tall_quads() -> None:
    # Median builder height 0.5: the 20 px quad is tall and leaves the row
    # at 10 to 10.5, so the right line at 18 opens a row of its own.
    left, right = _quad(0, 10, 100, 10.5), _quad(210, 18, 300, 18.5)

    assert _groups([left, right, _quad(110, 1, 200, 21)], {0, 1}) == [[0, 2], [1]]


def test_rows_with_the_same_centre_read_from_the_higher_top() -> None:
    assert _groups([_quad(0, 0, 100, 40), _quad(0, 10, 100, 30)], {0, 1}) == [[0], [1]]
    assert _groups([_quad(0, 10, 100, 30), _quad(0, 0, 100, 40)], {0, 1}) == [[1], [0]]


def test_row_groups_order_level_quads_by_centre_x_before_coordinates() -> None:
    # The wide quad and the short one inside it share centre y 30. The short
    # one has the smaller centre x, so it opens its row first and the side
    # quad is compared with the wide quad's row, which it shares.
    quads = [_quad(0, 0, 100, 60), _quad(30, 25, 60, 35), _quad(200, 34, 300, 74)]

    assert _groups(quads, {0, 1, 2}) == [[0, 2], [1]]


def test_rows_with_the_same_centre_and_top_read_from_the_higher_bottom() -> None:
    # Both rows have top centre 20 and top 0; the second row reaches 60 and
    # starts further left, so the bottom, not the left edge, decides.
    quads = [_quad(20, 0, 120, 40), _quad(10, 0, 220, 40), _quad(230, 10, 330, 60)]

    assert _groups(quads, {0, 1, 2}) == [[0], [1, 2]]


def test_width_classes_under_a_pixel() -> None:
    extents = [
        _ext(0, 10 * i, width, 10 * i + 5) for i, width in enumerate([0.8, 0.8, 0.8, 0.3, 1.3])
    ]

    assert _width_classes(extents) == ([3], [4], [0, 1, 2])


def test_narrow_quads_join_the_column_nearest_their_own_edges() -> None:
    columns = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    columns += [_quad(200, y, 300, y + 20) for y in (0, 40, 80)]
    # 20 px from the left column, 35 from the right one.
    nearer_left = _quad(120, 10, 165, 15)
    # 33 px from the left column, 22 from the right one.
    nearer_right = _quad(133, 50, 178, 60)

    layout = layout_lines(columns + [nearer_left, nearer_right])

    assert [column.members for column in layout.columns] == [(0, 1, 2, 6), (3, 4, 5, 7)]


def test_spanning_quads_sort_by_centre_not_by_width() -> None:
    columns = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    columns += [_quad(200, y, 300, y + 20) for y in (0, 40, 80)]

    lower_narrower = [_quad(0, 200, 290, 220), _quad(0, 120, 300, 140)]
    level = [_quad(0, 120, 300, 140), _quad(10, 120, 250, 140)]

    assert layout_lines(columns + lower_narrower).spanning == (7, 6)
    # Same centre y: the one centred further left reads first.
    assert layout_lines(columns + level).spanning == (7, 6)


def test_layouts_are_frozen() -> None:
    layout = layout_lines(_spread())

    for target, name in ((layout, "direction"), (layout.columns[0], "xmin")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, name, 5.0)
