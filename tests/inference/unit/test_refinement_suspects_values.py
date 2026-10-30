"""Exact values from suspect classification and the refinement entry point.

Band edges, the 1 px tolerance, the gutter's strict edges and the 20 degree
angle limit are each pinned on the boundary; the pipeline switches are
pinned by what each stage leaves on the lines.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det import refinement
from nomikos_inference.architectures.ppocr_det.geometry import Work
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import (
    ColumnLayout,
    PageLayout,
    layout_lines,
)
from nomikos_inference.architectures.ppocr_det.refinement import (
    BAND_TOLERANCE_PX,
    SUSPECT_ANGLE_DEGREES,
    _bbox_angle,
    _builder_bands,
    _in_gutter,
    _outside_bands,
    _refined_line,
    _text_block,
    classify_suspects,
    refine_to_lines,
)


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _box(xmin: float, ymin: float, xmax: float, ymax: float) -> Work:
    return Work(
        poly=Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]),
        baseline=((xmin, ymax), (xmax, ymax)),
        score=0.9,
        members=(0,),
    )


def _column(members: tuple[int, ...], builders: tuple[int, ...]) -> ColumnLayout:
    return ColumnLayout(
        members=members,
        builders=builders,
        rows=tuple((i,) for i in members),
        sequence=members,
        xmin=0.0,
        xmax=0.0,
    )


def _layout(*columns: ColumnLayout) -> PageLayout:
    return PageLayout(columns=columns, column_order=(), spanning=(), direction="ltr", centre_y=())


def test_constants() -> None:
    assert BAND_TOLERANCE_PX == 1.0
    assert SUSPECT_ANGLE_DEGREES == 20.0


def test_builder_bands_are_sorted_and_skip_columns_without_builders() -> None:
    quads = [
        _quad(300, 0, 400, 20),
        _quad(0, 0, 100, 20),
        _quad(10, 30, 120, 50),
        _quad(500, 0, 510, 9),
    ]
    layout = _layout(_column((0,), (0,)), _column((1, 2), (1, 2)), _column((3,), ()))

    assert _builder_bands(layout, quads) == [(0, 120), (300, 400)]


def test_text_block_spans_the_builders_only() -> None:
    quads = [_quad(0, 10, 100, 20), _quad(0, 50, 100, 70), _quad(0, 0, 5, 200)]
    layout = _layout(_column((0, 1, 2), (0, 1)))

    assert _text_block(layout, quads) == (10, 70)
    assert _text_block(_layout(_column((2,), ())), quads) is None
    assert _text_block(_layout(), quads) is None


def test_bbox_angle_of_boxes() -> None:
    assert _bbox_angle([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]) == pytest.approx(45.0)
    assert _bbox_angle([[0.0, 0.0], [30.0, 0.0], [30.0, 10.0]]) == pytest.approx(18.434948822922)
    assert _bbox_angle([[5.0, 0.0], [5.0, 10.0]]) == 90.0
    assert _bbox_angle([[0.0, 0.0], [10.0, 0.0]]) == 0.0


GAPS = [((0.0, 100.0), (200.0, 300.0))]
BLOCK = (10.0, 90.0)


def test_in_gutter_is_strictly_between_the_bands() -> None:
    assert _in_gutter(_box(101, 40, 199, 60), BLOCK, GAPS) is True
    assert _in_gutter(_box(100, 40, 150, 60), BLOCK, GAPS) is False
    assert _in_gutter(_box(150, 40, 200, 60), BLOCK, GAPS) is False


def test_in_gutter_needs_the_centre_inside_the_text_block() -> None:
    # Centre y 10 and 90 are on the block edges and count.
    assert _in_gutter(_box(120, 0, 180, 20), BLOCK, GAPS) is True
    assert _in_gutter(_box(120, 80, 180, 100), BLOCK, GAPS) is True
    assert _in_gutter(_box(120, 0, 180, 19), BLOCK, GAPS) is False
    assert _in_gutter(_box(120, 81, 180, 100), BLOCK, GAPS) is False
    assert _in_gutter(_box(120, 40, 180, 60), None, GAPS) is False
    assert _in_gutter(_box(120, 40, 180, 60), BLOCK, []) is False


def test_outside_bands_allows_one_pixel_either_side() -> None:
    bands = [(0.0, 100.0), (200.0, 300.0)]

    assert _outside_bands(_box(-1, 0, 101, 10), bands) is False
    assert _outside_bands(_box(199, 0, 301, 10), bands) is False
    assert _outside_bands(_box(-1.5, 0, 50, 10), bands) is True
    assert _outside_bands(_box(50, 0, 101.5, 10), bands) is True
    assert _outside_bands(_box(90, 0, 210, 10), bands) is True
    assert _outside_bands(_box(0, 0, 10, 10), []) is True


def _classified(quads: list[DetectedQuad], works: list[Work]) -> list[tuple[bool, str]]:
    classify_suspects(works, layout_lines(quads), quads)
    return [(work.suspect, work.suspect_reason) for work in works]


def _tilted(xmin: float, ymin: float, width: float, rise: float) -> Work:
    """A thin parallelogram whose bounding box rises ``rise`` over ``width``."""
    return Work(
        poly=Polygon(
            [
                (xmin, ymin + rise),
                (xmin + width, ymin),
                (xmin + width, ymin + 1),
                (xmin, ymin + rise + 1),
            ]
        ),
        baseline=((xmin, ymin + rise), (xmin + width, ymin)),
        score=0.9,
        members=(9,),
    )


def test_angle_more_than_twenty_degrees_off_the_page_is_a_suspect() -> None:
    quads = [_quad(0, y, 100, y + 20) for y in (0, 40, 80, 120, 160)]
    works = [_box(0, y, 100, y + 20) for y in (0, 40, 80, 120, 160)]
    page_angle = _bbox_angle([[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]])
    # Bounding-box heights rise + 1 over width 50, chosen to land just under
    # and just over 20 degrees off the page angle of atan(20 / 100).
    under = math.tan(math.radians(page_angle + 19.9)) * 50 - 1
    over = math.tan(math.radians(page_angle + 20.1)) * 50 - 1
    works.append(_tilted(20, 60, 50, under))
    works.append(_tilted(20, 100, 50, over))
    for work in works[-2:]:
        work.merged_from = 2

    assert _classified(quads, works)[-2:] == [(False, ""), (True, "angle")]


def test_an_angle_exactly_the_limit_off_is_not_a_suspect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact angles need a limit of 45: flat boxes 0, a square 45, an upright 90."""
    monkeypatch.setattr(refinement, "SUSPECT_ANGLE_DEGREES", 45.0)
    quads = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    works = [_box(0, y, 100, y) for y in (10, 50, 90)]
    works += [_box(20, 30, 30, 40), _box(50, 60, 50, 70)]
    for work in works:
        work.merged_from = 2

    assert _classified(quads, works) == [(False, "")] * 4 + [(True, "angle")]


def test_initials_and_merged_lines_are_spared_outside_the_bands() -> None:
    quads = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    works = [_box(0, y, 100, y + 20) for y in (0, 40, 80)]
    outside = [_box(400, 30, 450, 40) for _ in range(3)]
    outside[1].role = "initial"
    outside[2].merged_from = 2
    works.extend(outside)

    assert _classified(quads, works)[-3:] == [(True, "outside_bands"), (False, ""), (False, "")]


def test_a_line_inside_its_band_is_not_a_suspect() -> None:
    quads = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    works = [_box(0, y, 100, y + 20) for y in (0, 40, 80)]

    assert _classified(quads, works) == [(False, "")] * 3


def test_refined_line_drops_the_closing_point_and_copies_the_flags() -> None:
    work = _box(0, 0, 10, 5)
    work.role, work.suspect, work.suspect_reason = "initial", True, "angle"
    work.merged_from, work.overlap_unresolved, work.members = 3, True, (4, 7)
    work.score = 0.25

    line = _refined_line(work)

    assert line.points == [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
    assert line.baseline == [[0.0, 5.0], [10.0, 5.0]]
    assert all(isinstance(c, float) for point in line.points + line.baseline for c in point)
    assert (line.score, line.members, line.role) == (0.25, (4, 7), "initial")
    assert (line.suspect, line.suspect_reason) == (True, "angle")
    assert (line.merged_from, line.overlap_unresolved) == (3, True)


def _page() -> list[DetectedQuad]:
    """Five builders, two row fragments, a stacked overlap and an outlier."""
    return [
        _quad(0, 0, 200, 20),
        _quad(0, 40, 200, 60),
        _quad(0, 80, 200, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(103, 120, 200, 140, score=0.6),
        _quad(0, 160, 200, 190),
        _quad(0, 180, 200, 210),
        _quad(500, 50, 520, 70),
    ]


def _summary(lines) -> list[tuple[tuple[int, ...], int, bool, str, bool]]:
    return [
        (line.members, line.merged_from, line.suspect, line.suspect_reason, line.overlap_unresolved)
        for line in lines
    ]


def test_full_refinement_merges_cuts_and_flags() -> None:
    quads = _page()

    lines = refine_to_lines(quads, layout_lines(quads))

    assert _summary(lines) == [
        ((0,), 1, False, "", False),
        ((1,), 1, False, "", False),
        ((7,), 1, True, "outside_bands", False),
        ((2,), 1, False, "", False),
        ((3, 4), 2, False, "", False),
        ((5,), 1, False, "", False),
        ((6,), 1, False, "", False),
    ]
    assert [line.role for line in lines] == ["line"] * 7
    assert lines[4].score == pytest.approx((0.8 * 2000 + 0.6 * 1940) / 3940)
    assert lines[4].baseline == [[0.0, 135.0], [200.0, 135.0]]
    # The stacked pair 5 and 6 is cut at the mid-baseline line, y 192.5.
    assert lines[5].points == [[0.0, 190.0], [200.0, 190.0], [200.0, 160.0], [0.0, 160.0]]
    assert lines[6].points == [[0.0, 210.0], [200.0, 210.0], [200.0, 192.5], [0.0, 192.5]]
    assert lines[6].baseline == [[0.0, 202.5], [200.0, 202.5]]


def test_without_merging_every_quad_is_its_own_line() -> None:
    quads = _page()

    lines = refine_to_lines(quads, layout_lines(quads), merge=False)

    assert [line.members for line in lines] == [(i,) for i in range(8)]
    assert [line.score for line in lines][3:5] == [0.8, 0.6]
    assert lines[7].suspect_reason == "outside_bands"
    tall = refine_to_lines(quads, layout_lines(quads), merge=False, merge_max_height_ratio=1.0)
    assert [line.role for line in tall] == ["line"] * 5 + ["initial"] * 2 + ["line"]


def test_without_resolving_the_stacked_pair_keeps_its_overlap() -> None:
    quads = _page()

    lines = refine_to_lines(quads, layout_lines(quads), resolve=False)

    assert lines[5].points == [[0.0, 160.0], [200.0, 160.0], [200.0, 190.0], [0.0, 190.0]]
    assert lines[6].points == [[0.0, 180.0], [200.0, 180.0], [200.0, 210.0], [0.0, 210.0]]
    assert [line.overlap_unresolved for line in lines] == [False] * 7


def test_without_classifying_nothing_is_a_suspect() -> None:
    quads = _page()

    lines = refine_to_lines(quads, layout_lines(quads), classify=False)

    assert [(line.suspect, line.suspect_reason) for line in lines] == [(False, "")] * 7


def test_refinement_options_reach_their_stages() -> None:
    quads = _page()
    layout = layout_lines(quads)

    halfway = refine_to_lines(quads, layout, baseline_fraction=0.5)
    narrow_gap = refine_to_lines(quads, layout, merge_gap_ratio=0.1)
    high_cut = refine_to_lines(quads, layout, overlap_cut_threshold=0.5)
    tall = refine_to_lines(quads, layout, merge_max_height_ratio=1.0)

    assert halfway[0].baseline == [[0.0, 10.0], [200.0, 10.0]]
    assert halfway[4].baseline == [[0.0, 130.0], [200.0, 130.0]]
    assert [line.members for line in narrow_gap][4:6] == [(3,), (4,)]
    assert high_cut[5].points == [[0.0, 160.0], [200.0, 160.0], [200.0, 190.0], [0.0, 190.0]]
    assert [line.role for line in tall][5:] == ["initial", "initial"]


def test_refinement_merge_overlap_ratio_reaches_the_merge() -> None:
    quads = [_quad(0, y, 200, y + 20) for y in (0, 40, 80)]
    quads += [_quad(0, 120, 100, 140), _quad(98.5, 120, 200, 140)]
    layout = layout_lines(quads)

    loose = refine_to_lines(quads, layout)
    strict = refine_to_lines(quads, layout, merge_max_overlap_ratio=0.0)

    assert [line.members for line in loose][-1] == (3, 4)
    assert [line.members for line in strict][-2:] == [(3,), (4,)]


def test_builder_bands_continue_past_a_column_without_builders() -> None:
    quads = [_quad(500, 0, 510, 9), _quad(0, 0, 100, 20)]
    layout = _layout(_column((0,), ()), _column((1,), (1,)))

    assert _builder_bands(layout, quads) == [(0, 100)]


def test_bbox_angle_of_a_point_and_of_a_box_under_a_pixel() -> None:
    assert _bbox_angle([[5.0, 5.0], [5.0, 5.0]]) == 90.0
    assert _bbox_angle([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]) == pytest.approx(45.0)


def test_classification_continues_past_a_spared_initial() -> None:
    quads = [_quad(0, y, 100, y + 20) for y in (0, 40, 80)]
    works = [_box(0, y, 100, y + 20) for y in (0, 40, 80)]
    works[0].role = "initial"
    works.append(_box(400, 30, 450, 40))

    assert _classified(quads, works) == [
        (False, ""),
        (False, ""),
        (False, ""),
        (True, "outside_bands"),
    ]


def test_refined_line_keeps_a_tilted_baseline() -> None:
    work = _box(0, 0, 10, 5)
    work.baseline = ((0.0, 4.0), (10.0, 2.0))

    assert _refined_line(work).baseline == [[0.0, 4.0], [10.0, 2.0]]


def test_refinement_spacing_falls_back_to_the_median_quad_height() -> None:
    """No rows on the layout: the stacked pair is judged against height 30.

    Baselines 20 apart are not under half of 30, so the pair is cut at the
    mid-baseline line y 32.5.
    """
    quads = [_quad(0, 0, 100, 30), _quad(0, 20, 100, 50)]

    lines = refine_to_lines(quads, layout_lines([]), merge=False, classify=False)

    assert [line.points for line in lines] == [
        [[0.0, 30.0], [100.0, 30.0], [100.0, 0.0], [0.0, 0.0]],
        [[0.0, 50.0], [100.0, 50.0], [100.0, 32.5], [0.0, 32.5]],
    ]
