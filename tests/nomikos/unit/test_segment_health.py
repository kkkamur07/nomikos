"""Geometry behind the segment-health suggestions.

Several of these guard a mistake that was actually made while writing the
module and caught by measuring against two real manuscripts, rather than a
hazard imagined afterwards. Those are called out where they sit.
"""

from __future__ import annotations

from backend.annotation.application.segment_health import (
    PageStats,
    Segment,
    baseline_distance,
    bbox,
    clip_baseline,
    clip_polygon,
    column_bands,
    convex_hull,
    find_fragments,
    find_overlaps,
    find_spanning,
    find_suspects,
    merge_baseline_group,
    merge_polygon_group,
    merge_polygons,
    overlap_area,
    page_stats,
    polygon_area,
)

PAGE_WIDTH = 2400.0
PAGE_HEIGHT = 3400.0
LEFT_COLUMN = (300.0, 1100.0)
RIGHT_COLUMN = (1300.0, 2100.0)


def line(
    identifier: str,
    x0: float,
    x1: float,
    y: float,
    *,
    height: float = 100.0,
    corners: int = 4,
    manual: bool = False,
    has_text: bool = False,
    is_paired: bool = False,
) -> Segment:
    """A rectangular text line, optionally with extra corners along its top."""
    top, bottom = y - height / 2, y + height / 2
    points = [[x0, top]]
    for index in range(1, max(1, corners - 3)):
        points.append([x0 + (x1 - x0) * index / max(1, corners - 3), top])
    points += [[x1, top], [x1, bottom], [x0, bottom]]
    return Segment(
        id=identifier,
        points=points,
        baseline=[[x0, bottom], [x1, bottom]],
        manual_geometry=manual,
        has_text=has_text,
        is_paired=is_paired,
    )


def tapered(x0: float, x1: float, y: float, *, height: float = 100.0) -> list[list[float]]:
    """A polygon shaped the way kraken draws one, rather than a rectangle.

    One vertex at each end, so the outline is a single point tall there, and the
    top and bottom vertices at different x in between.
    """
    top, bottom = y - height / 2, y + height / 2
    span = x1 - x0
    return (
        [[x0, y - height / 8]]
        + [[x0 + span * f, top + height / 12 * i] for i, f in enumerate((0.2, 0.5, 0.8))]
        + [[x1, y + height / 8]]
        + [[x0 + span * f, bottom - height / 12 * i] for i, f in enumerate((0.75, 0.45, 0.15))]
    )


def mask(
    identifier: str,
    top: float,
    bottom: float,
    baseline_y: float,
    *,
    x0: float = 300.0,
    x1: float = 1100.0,
    manual: bool = False,
) -> Segment:
    """A line mask with its baseline where a baseline really sits.

    Inside the mask rather than along its bottom edge, because ascenders and
    descenders are why two neighbouring masks touch at all, and a cut midway
    between two baselines only makes sense against real ones.
    """
    return Segment(
        id=identifier,
        points=[[x0, top], [x1, top], [x1, bottom], [x0, bottom]],
        baseline=[[x0, baseline_y], [x1, baseline_y]],
        manual_geometry=manual,
    )


def overlapping(
    share: float, *, manual: bool = False, height: float = 120.0
) -> tuple[list[Segment], PageStats]:
    """Two stacked masks sharing a given fraction of one mask's height."""
    # Below the twenty rows the two-column page draws, so the pair under test
    # is the only overlap on it.
    top = 2800.0
    shared = height * share
    upper = mask("U", top, top + height, top + height * 0.58)
    lower = mask(
        "L",
        top + height - shared,
        top + 2 * height - shared,
        top + 1.58 * height - shared,
        manual=manual,
    )
    segments = two_column_page() + [upper, lower]
    return segments, page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)


def two_column_page(spanning: list[Segment] | None = None) -> list[Segment]:
    """Twenty lines a side, at a 120px pitch, with an empty gutter between."""
    segments: list[Segment] = []
    for row in range(20):
        y = 400.0 + row * 120.0
        segments.append(line(f"L{row}", *LEFT_COLUMN, y))
        segments.append(line(f"R{row}", *RIGHT_COLUMN, y))
    return segments + list(spanning or [])


class TestColumnBands:
    def test_finds_both_columns_and_the_gutter_between_them(self) -> None:
        bands = column_bands(two_column_page(), PAGE_WIDTH)
        assert len(bands) == 2
        assert bands[0][1] < bands[1][0]

    def test_a_few_segments_crossing_the_gutter_do_not_weld_the_columns(self) -> None:
        # The failure this catches, and it is the one the first implementation
        # shipped with. Merging overlapping x-ranges makes the segments that
        # span the gutter define a single band covering the whole text area.
        # The spanning segments are then inside one column, so nothing detects
        # them, and line spacing is measured across two interleaved columns and
        # comes out at half its real value, which silently disables the
        # fragment detector too. One bad segment must not outvote forty good
        # ones.
        crossers = [
            line(f"X{index}", LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0 + index * 120.0)
            for index in range(3)
        ]
        bands = column_bands(two_column_page(crossers), PAGE_WIDTH)
        assert len(bands) == 2, f"gutter lost to {len(crossers)} spanning segments: {bands}"

    def test_a_single_column_page_yields_one_band(self) -> None:
        segments = [line(f"L{row}", 300.0, 2100.0, 400.0 + row * 120.0) for row in range(20)]
        assert len(column_bands(segments, PAGE_WIDTH)) == 1

    def test_a_page_of_nothing_but_specks_has_no_bands(self) -> None:
        specks = [
            line(f"S{index}", 100.0 + index * 30, 120.0 + index * 30, 200.0) for index in range(10)
        ]
        assert column_bands(specks, PAGE_WIDTH) == []


class TestBaselineDistance:
    def test_two_pieces_of_one_line_read_as_close_despite_the_gap(self) -> None:
        # The failure this catches. Measuring point-to-polyline distance makes
        # the horizontal gap between two fragments dominate the answer, so two
        # halves of one text line report as further apart than two different
        # lines stacked on top of each other, and no fragment is ever found.
        left = [[300.0, 500.0], [800.0, 500.0]]
        right = [[1000.0, 502.0], [1100.0, 502.0]]
        stacked = [[300.0, 620.0], [1100.0, 620.0]]
        assert baseline_distance(left, right) < baseline_distance(left, stacked)
        assert baseline_distance(left, right) < 10

    def test_a_line_and_its_neighbour_read_as_a_line_apart(self) -> None:
        first = [[300.0, 500.0], [1100.0, 500.0]]
        second = [[300.0, 620.0], [1100.0, 620.0]]
        assert baseline_distance(first, second) == 120.0


class TestClipping:
    def test_cutting_a_rectangle_keeps_each_side_whole(self) -> None:
        rectangle = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
        left = clip_polygon(rectangle, None, 60.0)
        right = clip_polygon(rectangle, 60.0, None)
        assert bbox(left) == (0.0, 60.0, 0.0, 50.0)
        assert bbox(right) == (60.0, 100.0, 0.0, 50.0)

    def test_a_baseline_cut_too_short_is_refused_rather_than_returned_tiny(self) -> None:
        assert clip_baseline([[0.0, 10.0], [100.0, 10.0]], 0.0, 5.0) == []


class TestSpanning:
    def test_a_segment_across_the_gutter_is_cut_between_the_columns(self) -> None:
        crosser = line("X", LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0)
        segments = two_column_page([crosser])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        found = [split for split in find_spanning(segments, stats) if split.line_id == "X"]
        assert len(found) == 1
        (cut,) = found[0].cuts
        assert LEFT_COLUMN[1] < cut < RIGHT_COLUMN[0]
        assert len(found[0].pieces) == 2

    def test_an_ordinary_line_inside_one_column_is_left_alone(self) -> None:
        segments = two_column_page()
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_spanning(segments, stats) == []


class TestFragments:
    def test_a_line_end_is_offered_for_merging_into_its_line(self) -> None:
        segments = two_column_page()
        primary = line("P", 300.0, 900.0, 2900.0)
        fragment = line("F", 960.0, 1060.0, 2900.0)
        segments += [primary, fragment]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        merges = [merge for merge in find_fragments(segments, stats) if merge.fragment_id == "F"]
        assert len(merges) == 1
        assert merges[0].primary_id == "P"

    def test_a_fragment_carrying_transcribed_text_is_never_offered(self) -> None:
        segments = two_column_page()
        segments += [
            line("P", 300.0, 900.0, 2900.0),
            line("F", 960.0, 1060.0, 2900.0, has_text=True),
        ]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [m for m in find_fragments(segments, stats) if m.fragment_id == "F"] == []

    def test_two_halves_of_similar_width_are_taken_as_two_real_lines(self) -> None:
        segments = two_column_page()
        segments += [line("A", 300.0, 700.0, 2900.0), line("B", 740.0, 1100.0, 2900.0)]
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [m for m in find_fragments(segments, stats) if m.fragment_id in {"A", "B"}] == []


class TestMergePolygons:
    def test_the_merged_outline_spans_the_gap_instead_of_dropping_the_fragment(self) -> None:
        # The failure this catches is the one the reference implementation
        # documents: bridging by a buffer round trip alone silently drops the
        # fragment, leaving the primary unchanged and the merge a no-op that
        # still reports success.
        primary = [[300.0, 450.0], [900.0, 450.0], [900.0, 550.0], [300.0, 550.0]]
        fragment = [[960.0, 450.0], [1060.0, 450.0], [1060.0, 550.0], [960.0, 550.0]]
        merged = merge_polygons(primary, fragment)
        box = bbox(merged)
        assert box[0] == 300.0
        assert box[1] == 1060.0
        # A point in the gap has to be inside the outline, which is what makes
        # this one line rather than two pieces sharing a bounding box.
        assert _contains(merged, (930.0, 500.0))

    def test_the_outline_of_two_kraken_shaped_pieces_does_not_cross_itself(self) -> None:
        # The failure this catches, and it is the one that shipped twice,
        # because every fixture above is a rectangle. A rectangle puts its top
        # and bottom vertices at the same x and is a full line height tall at
        # both ends. A kraken mask does neither: it tapers to one vertex at each
        # end, and its top and bottom vertices sit at different x. The end
        # vertex then belongs to the top edge and the bottom edge at once, so
        # the outline arrived at it twice and closed against itself, and the
        # attempt at walking around it instead sent the return path back across
        # the outward one. Both are outlines no consumer of geometry accepts,
        # and both scored well on bounding box and point count, which is all the
        # rectangles could ever have checked.
        merged = merge_polygons(tapered(300.0, 900.0, 500.0), tapered(960.0, 1160.0, 504.0))
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        assert _contains(merged, (930.0, 500.0)), "the gap between the pieces is not enclosed"

    def test_a_fragment_sitting_inside_the_primary_still_gives_a_simple_outline(self) -> None:
        # Not every pair offered is side by side. A blot under a line overlaps
        # it in x completely, and cutting the two apart at the middle of that
        # overlap leaves nothing of one of them, which used to leave the two
        # edges interleaved and crossing.
        primary = tapered(1381.0, 1446.0, 2840.0, height=120.0)
        fragment = tapered(1399.0, 1414.0, 2985.0, height=280.0)
        merged = merge_polygons(primary, fragment)
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        assert bbox(merged)[1] >= 1446.0, "the wider piece was clipped away"

    def test_pieces_overlapping_in_x_still_produce_a_simple_outline(self) -> None:
        primary = [[300.0, 450.0], [900.0, 450.0], [900.0, 550.0], [300.0, 550.0]]
        fragment = [[850.0, 450.0], [1000.0, 450.0], [1000.0, 550.0], [850.0, 550.0]]
        merged = merge_polygons(primary, fragment)
        assert bbox(merged) == (300.0, 1000.0, 450.0, 550.0)
        assert len(merged) >= 4


class TestMergeGroups:
    """The same join over any number of pieces, which is what a free merge needs."""

    def test_three_pieces_come_back_as_one_outline_that_encloses_both_gaps(self) -> None:
        merged = merge_polygon_group(
            [
                tapered(300.0, 600.0, 500.0),
                tapered(700.0, 900.0, 502.0),
                tapered(1000.0, 1100.0, 498.0),
            ]
        )
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        assert _contains(merged, (650.0, 500.0)), "the first gap is not enclosed"
        assert _contains(merged, (950.0, 500.0)), "the second gap is not enclosed"
        assert bbox(merged)[1] == 1100.0

    def test_two_lines_stacked_one_above_the_other_merge_into_one_simple_outline(self) -> None:
        """Not every free selection sits side by side; a reviewer merges rows too."""
        merged = merge_polygon_group([tapered(300.0, 900.0, 500.0), tapered(300.0, 900.0, 640.0)])
        assert _is_simple(merged), f"outline crosses or touches itself: {merged}"
        box = bbox(merged)
        assert box[2] <= 450.0 and box[3] >= 690.0
        assert _contains(merged, (600.0, 570.0)), "the space between the rows is not enclosed"

    def test_a_single_piece_comes_back_as_itself(self) -> None:
        only = [[300.0, 450.0], [900.0, 450.0], [900.0, 550.0], [300.0, 550.0]]
        assert merge_polygon_group([only]) == only

    def test_pieces_too_thin_to_have_an_envelope_fall_back_to_the_hull(self) -> None:
        """Everything on one x: there is no chain to walk, and concatenating
        two verticals gives a ring that crosses itself."""
        merged = merge_polygon_group([[[10.0, 0.0], [10.0, 5.0]], [[10.0, 8.0], [10.0, 12.0]]])
        assert bbox(merged) == (10.0, 10.0, 0.0, 12.0)

    def test_the_hull_of_a_point_cloud_is_simple_and_keeps_the_extremes(self) -> None:
        hull = convex_hull(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0], [5.0, 5.0]]
        )
        assert _is_simple(hull)
        assert sorted(tuple(point) for point in hull) == [
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 0.0),
            (10.0, 10.0),
        ]

    def test_baselines_join_left_to_right_whatever_order_they_arrive_in(self) -> None:
        joined = merge_baseline_group(
            [
                [[1000.0, 500.0], [1100.0, 500.0]],
                [[300.0, 500.0], [600.0, 500.0]],
                [[700.0, 500.0], [900.0, 500.0]],
            ]
        )
        xs = [point[0] for point in joined]
        assert xs == sorted(xs)
        assert xs[0] == 300.0 and xs[-1] == 1100.0

    def test_a_piece_without_a_baseline_contributes_nothing(self) -> None:
        joined = merge_baseline_group([[[300.0, 500.0], [600.0, 500.0]], []])
        assert joined == [[300.0, 500.0], [600.0, 500.0]]

    def test_nothing_with_a_baseline_gives_nothing_rather_than_a_placeholder(self) -> None:
        assert merge_baseline_group([[], []]) == []


class TestOverlaps:
    def test_two_masks_sharing_a_third_are_offered_a_cut_between_the_baselines(self) -> None:
        segments, stats = overlapping(0.3)
        (trim,) = [t for t in find_overlaps(segments, stats) if t.upper_id == "U"]
        assert trim.lower_id == "L"
        assert not trim.duplicate
        upper, lower = (
            next(s for s in segments if s.id == "U"),
            next(s for s in segments if s.id == "L"),
        )
        assert upper.baseline[0][1] < trim.cut < lower.baseline[0][1]
        # The point of the cut: afterwards they share nothing at all.
        assert overlap_area(trim.upper_points, trim.lower_points) == 0.0

    def test_lines_that_merely_touch_are_left_alone(self) -> None:
        # Neighbouring masks always share a little, which is a descender doing
        # its job. Across two manuscripts 1388 of 1617 overlapping pairs shared
        # under 5% of the smaller mask, and flagging those would bury the
        # nineteen that mean something.
        segments, stats = overlapping(0.05)
        assert [t for t in find_overlaps(segments, stats) if t.upper_id == "U"] == []

    def test_one_line_drawn_twice_is_reported_without_a_cut(self) -> None:
        # The failure this catches. A cut midway between the baselines is the
        # right fix for two lines that bled into each other and the wrong one
        # for a single line kraken emitted twice: it halves both copies and
        # leaves two mutilated masks where a human would have deleted one. The
        # worst real pair shared 98% and would have lost half its area.
        segments, stats = overlapping(0.9)
        (trim,) = [t for t in find_overlaps(segments, stats) if t.upper_id == "U"]
        assert trim.duplicate
        assert trim.upper_points == []
        assert trim.lower_points == []

    def test_a_hand_drawn_mask_is_never_trimmed(self) -> None:
        segments, stats = overlapping(0.3, manual=True)
        assert [t for t in find_overlaps(segments, stats) if t.upper_id == "U"] == []

    def test_the_trim_says_what_each_side_would_lose(self) -> None:
        # Offered, not applied, so the cost has to travel with the offer. On
        # real pages the cut takes a sixth of a mask on average.
        segments, stats = overlapping(0.3)
        (trim,) = [t for t in find_overlaps(segments, stats) if t.upper_id == "U"]
        assert 0 < trim.upper_loss < 1
        assert 0 < trim.lower_loss < 1
        segments_by_id = {segment.id: segment for segment in segments}
        kept = polygon_area(trim.upper_points) + polygon_area(trim.lower_points)
        whole = sum(polygon_area(segments_by_id[i].points) for i in ("U", "L"))
        assert kept < whole

    def test_two_tapered_masks_are_measured_by_what_they_share_not_their_boxes(self) -> None:
        # Two kraken masks can have boxes that overlap heavily while the masks
        # themselves barely touch, because each tapers to a point at its ends.
        # Comparing boxes would flag the pair; comparing the masks does not.
        first = tapered(300.0, 1100.0, 500.0, height=120.0)
        second = tapered(300.0, 1100.0, 560.0, height=120.0)
        first_box, second_box = bbox(first), bbox(second)
        box_overlap = (min(first_box[1], second_box[1]) - max(first_box[0], second_box[0])) * (
            min(first_box[3], second_box[3]) - max(first_box[2], second_box[2])
        )
        assert overlap_area(first, second) < box_overlap * 0.8


class TestAreas:
    def test_the_area_of_a_rectangle_is_its_sides(self) -> None:
        assert polygon_area([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]) == 40.0

    def test_the_shared_area_of_two_rectangles_is_the_part_they_both_cover(self) -> None:
        first = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
        second = [[6.0, 2.0], [20.0, 2.0], [20.0, 9.0], [6.0, 9.0]]
        assert overlap_area(first, second) == 8.0

    def test_outlines_that_pass_over_each_other_share_nothing(self) -> None:
        # The failure this catches: the shared height at a column is the
        # overlap of two vertical extents, and where one outline sits well
        # above the other that number is negative. Added up rather than floored
        # at zero it subtracts real overlap elsewhere, and two masks that
        # genuinely grew into each other come out reading as clean.
        above = [[0.0, 0.0], [100.0, 0.0], [100.0, 10.0], [0.0, 10.0]]
        below = [[0.0, 400.0], [100.0, 400.0], [100.0, 410.0], [0.0, 410.0]]
        assert overlap_area(above, below) == 0.0

    def test_outlines_that_do_not_touch_share_nothing(self) -> None:
        first = [[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]
        second = [[40.0, 0.0], [50.0, 0.0], [50.0, 4.0], [40.0, 4.0]]
        assert overlap_area(first, second) == 0.0


class TestSuspects:
    def test_running_text_is_never_suspect(self) -> None:
        segments = two_column_page()
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []

    def test_a_speck_in_the_outer_margin_is_flagged(self) -> None:
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert [s.line_id for s in find_suspects(segments, stats)] == ["N"]

    def test_a_hand_drawn_segment_is_never_suspect(self) -> None:
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0, manual=True)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []

    def test_a_narrow_segment_someone_has_transcribed_is_never_suspect(self) -> None:
        # Untranscribed marginalia is exactly what this whole feature must not
        # delete on its own judgement; once it carries text, it is not even a
        # candidate.
        speck = line("N", 2280.0, 2320.0, 800.0, height=30.0, has_text=True)
        segments = two_column_page([speck])
        stats = page_stats(segments, PAGE_WIDTH, PAGE_HEIGHT)
        assert find_suspects(segments, stats) == []


def _contains(polygon: list[list[float]], point: tuple[float, float]) -> bool:
    """Ray casting, so the merge test asserts containment rather than extent."""
    x, y = point
    inside = False
    for index in range(len(polygon)):
        ax, ay = polygon[index]
        bx, by = polygon[index - 1]
        if (ay > y) != (by > y) and x < ax + (bx - ax) * (y - ay) / (by - ay):
            inside = not inside
    return inside


def _is_simple(polygon: list[list[float]]) -> bool:
    """No vertex used twice, and no two edges that are not neighbours crossing."""
    if len({(p[0], p[1]) for p in polygon}) != len(polygon):
        return False
    count = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % count]) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _crosses(*edges[i], *edges[j]):
                return False
    return True


def _crosses(a, b, c, d) -> bool:
    def side(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(value) < 1e-9 else (1 if value > 0 else -1)

    def on(p, q, r):
        return (
            side(p, q, r) == 0
            and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and (min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
        )

    d1, d2, d3, d4 = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    if d1 != d2 and d3 != d4:
        return True
    return any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))
