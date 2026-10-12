"""Refinement of raw detection quads: merge, overlap cuts, layout sharing.

Synthetic quads only, no model needed. Fragments that share a column and a
row merge below a gap ratio; tall initials stay separate; stacked neighbours
are cut apart at the mid-baseline line; duplicates drop the lower score.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import (
    merge_row_fragments,
    resolve_overlaps,
)


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _refine(quads: list[DetectedQuad]):
    layout = layout_lines(quads, direction="ltr")
    heights = [
        max(point[1] for point in quad.points) - min(point[1] for point in quad.points)
        for quad in quads
    ]
    median_height = sorted(heights)[len(heights) // 2]
    merged = merge_row_fragments(quads, layout)
    return resolve_overlaps(merged, layout, median_height=median_height)


def _column(x0: float, x1: float, ys: list[float], height: float = 20.0) -> list[DetectedQuad]:
    return [_quad(x0, y, x1, y + height) for y in ys]


def test_row_mates_merge_into_one_valid_polygon() -> None:
    quads = _column(0, 200, [0, 40, 80])
    left = _quad(0, 120, 100, 140, score=0.8)
    right = _quad(103, 122, 220, 142, score=0.6)
    quads.extend([left, right])

    works = _refine(quads)
    merged = [work for work in works if work.merged_from == 2]

    assert len(merged) == 1
    assert len(works) == len(quads) - 1
    assert merged[0].poly.is_valid
    left_poly = Polygon(left.points)
    right_poly = Polygon(right.points)
    # Side-by-side members, so the union is both areas plus the small
    # bridge over the 3 px gap inside their shared y range.
    assert merged[0].poly.area >= left_poly.area + right_poly.area - 1e-6
    assert merged[0].poly.area < left_poly.area + right_poly.area + 200
    assert merged[0].poly.covers(left_poly)
    assert merged[0].poly.covers(right_poly)
    expected = (0.8 * left_poly.area + 0.6 * right_poly.area) / (left_poly.area + right_poly.area)
    assert merged[0].score == pytest.approx(expected)
    assert len(merged[0].baseline) == 2


def test_gap_above_the_ratio_does_not_merge() -> None:
    quads = _column(0, 340, [0, 40, 80])
    quads.append(_quad(0, 120, 150, 140))
    quads.append(_quad(190, 120, 340, 140))

    works = _refine(quads)

    assert all(work.merged_from == 1 for work in works)
    assert len(works) == len(quads)


def test_tall_initial_does_not_merge_and_gets_the_role() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(60, 120, 200, 140))
    quads.append(_quad(0, 100, 50, 160))

    works = _refine(quads)
    initials = [work for work in works if work.role == "initial"]

    assert len(initials) == 1
    assert initials[0].merged_from == 1
    assert len(initials[0].members) == 1
    assert all(work.role == "line" for work in works if work not in initials)
    assert len(works) == len(quads)


def test_merging_never_crosses_columns() -> None:
    quads = _column(0, 100, [0, 40, 80, 120])
    quads.extend(_column(300, 400, [0, 40, 80, 120]))

    works = _refine(quads)

    assert len(works) == len(quads)
    for work in works:
        xmin = work.poly.bounds[0]
        xmax = work.poly.bounds[2]
        assert (xmax <= 100.0) or (xmin >= 300.0)


def test_shuffled_input_gives_the_same_merge() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(0, 120, 120, 140, score=0.8))
    quads.append(_quad(100, 122, 220, 142, score=0.6))
    shuffled = [quads[4], quads[0], quads[3], quads[1], quads[2]]

    first = _refine(quads)
    second = _refine(shuffled)

    # Members are input positions, so match works across runs by geometry.
    assert len(first) == len(second)
    for work in first:
        match = [other for other in second if other.poly.equals(work.poly)]
        assert len(match) == 1
        assert match[0].role == work.role
        assert match[0].merged_from == work.merged_from
        assert match[0].score == pytest.approx(work.score)


def test_stacked_pair_sharing_thirty_percent_is_cut_to_zero() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 95))
    quads.append(_quad(0, 85, 100, 110))

    before = [Polygon(quad.points).area for quad in quads[-2:]]
    works = _refine(quads)
    pair = [work for work in works if set(work.members) & {3, 4}]

    assert len(pair) == 2
    assert pair[0].poly.intersection(pair[1].poly).area == pytest.approx(0.0)
    assert not pair[0].overlap_unresolved and not pair[1].overlap_unresolved
    for work, area in zip(sorted(pair, key=lambda w: min(w.members)), before, strict=True):
        assert work.area > 0.5 * area
        assert work.poly.covers(Point(work.baseline[0]))
        assert work.poly.covers(Point(work.baseline[1]))


def test_ninety_percent_duplicate_drops_the_lower_score() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 100, score=0.9))
    quads.append(_quad(5, 72, 95, 98, score=0.7))

    works = _refine(quads)

    assert len(works) == len(quads) - 1
    survivor = [work for work in works if 3 in work.members or 4 in work.members]
    assert len(survivor) == 1
    assert survivor[0].score == pytest.approx(0.9)


def test_six_percent_overlap_is_left_alone() -> None:
    quads = _column(0, 100, [0, 100, 200], height=20.0)
    quads.append(_quad(0, 240, 100, 290))
    quads.append(_quad(0, 287, 100, 337))

    works = _refine(quads)

    assert len(works) == len(quads)
    assert not any(work.overlap_unresolved for work in works)


def _classify(quads: list[DetectedQuad]):
    from nomikos_inference.architectures.ppocr_det.refinement import classify_suspects

    layout = layout_lines(quads, direction="ltr")
    heights = [
        max(point[1] for point in quad.points) - min(point[1] for point in quad.points)
        for quad in quads
    ]
    median_height = sorted(heights)[len(heights) // 2]
    merged = merge_row_fragments(quads, layout)
    works = resolve_overlaps(merged, layout, median_height=median_height)
    classify_suspects(works, layout, quads)
    return works, layout


def test_outside_singleton_is_a_suspect_with_its_reason() -> None:
    quads = _column(0, 100, [0, 40, 80])
    quads.append(_quad(430, 30, 450, 50))

    works, _ = _classify(quads)
    flagged = [work for work in works if work.suspect]

    assert len(flagged) == 1
    assert flagged[0].suspect_reason == "outside_bands"
    assert flagged[0].merged_from == 1


def test_gutter_numeral_between_two_columns_is_not_a_suspect() -> None:
    quads = _column(0, 100, [0, 40, 80])
    quads.extend(_column(300, 400, [0, 40, 80]))
    quads.append(_quad(190, 30, 210, 50))

    works, _ = _classify(quads)

    assert not any(work.suspect for work in works)


def test_tall_initial_is_not_a_suspect_despite_its_angle() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(60, 120, 200, 140))
    quads.append(_quad(0, 100, 50, 160))

    works, _ = _classify(quads)

    assert not any(work.suspect for work in works)
    assert sum(1 for work in works if work.role == "initial") == 1


def test_steep_diagonal_is_an_angle_suspect() -> None:
    quads = _column(0, 200, [0, 40, 80, 120])
    diagonal = DetectedQuad(
        points=[[80.0, 160.0], [120.0, 160.0], [140.0, 200.0], [100.0, 200.0]],
        score=0.9,
    )
    quads.append(diagonal)

    works, _ = _classify(quads)
    flagged = [work for work in works if work.suspect]

    assert len(flagged) == 1
    assert flagged[0].suspect_reason == "angle"


def test_row_mates_overlapping_by_half_a_line_do_not_merge() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(0, 120, 100, 140))
    quads.append(_quad(90, 120, 200, 140))

    works = _refine(quads)

    assert all(work.merged_from == 1 for work in works)
    assert len(works) == len(quads)


def test_row_mates_with_a_tenth_line_gap_do_merge() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(0, 120, 100, 140))
    quads.append(_quad(102, 120, 200, 140))

    works = _refine(quads)
    merged = [work for work in works if work.merged_from == 2]

    assert len(merged) == 1
    assert len(works) == len(quads) - 1


def test_merged_line_outside_its_band_is_not_a_suspect() -> None:
    quads = _column(0, 200, [0, 40, 80])
    quads.append(_quad(0, 120, 100, 140, score=0.8))
    quads.append(_quad(103, 122, 220, 142, score=0.6))

    works, _ = _classify(quads)

    assert not any(work.suspect for work in works)
    assert sum(1 for work in works if work.merged_from == 2) == 1


def _respond(quads: list[DetectedQuad], noise_policy: str, classify: bool):
    from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines
    from nomikos_inference.architectures.ppocr_det.response import (
        build_refined_ppocr_det_response,
    )

    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(quads, layout, classify=classify)
    return build_refined_ppocr_det_response(
        500, 300, quads, items, layout, noise_policy=noise_policy
    )


def _suspect_page() -> list[DetectedQuad]:
    # The suspect sits above the body in reading order, so `flag` must move it.
    quads = [_quad(430, -40, 450, -20)]
    quads.extend(_column(0, 200, [0, 40]))
    return quads


def test_flag_orders_suspects_last() -> None:
    response = _respond(_suspect_page(), "flag", True)

    assert len(response.lines) == 3
    assert [line.order for line in response.lines] == [0, 1, 2]
    assert response.lines[-1].source_metadata.get("suspect") is True
    assert response.lines[-1].source_metadata.get("suspect_reason") == "outside_bands"
    assert not any(line.source_metadata.get("suspect") for line in response.lines[:2])
    assert min(point[0] for point in response.lines[-1].points) == 430.0


def test_drop_removes_suspects() -> None:
    response = _respond(_suspect_page(), "drop", True)

    assert len(response.lines) == 2
    assert not any(line.source_metadata.get("suspect") for line in response.lines)


def test_off_changes_nothing() -> None:
    response = _respond(_suspect_page(), "off", False)

    assert len(response.lines) == 3
    assert not any(line.source_metadata.get("suspect") for line in response.lines)
    # Natural reading order keeps the top suspect first: nothing is moved.
    assert min(point[0] for point in response.lines[0].points) == 430.0


def test_containment_small_higher_score_both_survive_unresolved() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 100, score=0.7))
    quads.append(_quad(10, 78, 50, 92, score=0.9))

    works = _refine(quads)
    pair = [work for work in works if set(work.members) & {3, 4}]

    assert len(pair) == 2
    assert len(works) == len(quads)
    assert all(work.overlap_unresolved for work in pair)


def test_containment_small_lower_score_both_survive_unresolved() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 100, score=0.9))
    quads.append(_quad(10, 78, 50, 92, score=0.7))

    works = _refine(quads)
    pair = [work for work in works if set(work.members) & {3, 4}]

    assert len(pair) == 2
    assert len(works) == len(quads)
    assert all(work.overlap_unresolved for work in pair)


def test_near_identical_shifted_two_px_drops_the_lower_score() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 100, score=0.9))
    quads.append(_quad(2, 70, 102, 100, score=0.7))

    works = _refine(quads)

    assert len(works) == len(quads) - 1
    survivor = [work for work in works if 3 in work.members or 4 in work.members]
    assert len(survivor) == 1
    assert survivor[0].score == pytest.approx(0.9)


def test_initial_overlapping_line_by_ninety_percent_both_survive() -> None:
    quads = _column(0, 100, [0, 25, 50], height=15.0)
    quads.append(_quad(0, 70, 100, 100, score=0.9))
    quads.append(_quad(0, 70, 90, 100, score=0.7))

    layout = layout_lines(quads, direction="ltr")
    merged = merge_row_fragments(quads, layout)
    for work in merged:
        if 4 in work.members:
            work.role = "initial"
    heights = [
        max(point[1] for point in quad.points) - min(point[1] for point in quad.points)
        for quad in quads
    ]
    median_height = sorted(heights)[len(heights) // 2]
    works = resolve_overlaps(merged, layout, median_height=median_height)
    pair = [work for work in works if set(work.members) & {3, 4}]

    assert len(pair) == 2
    assert len(works) == len(quads)
    assert all(work.overlap_unresolved for work in pair)


def test_cut_that_would_remove_more_than_half_marks_unresolved() -> None:
    # The small box sits exactly on the mid-baseline line, so its keep-side
    # piece is exactly half and the guard refuses the cut.
    quads = _column(0, 100, [-100, -85, -70, -55, -40], height=8.0)
    quads.append(_quad(0, 0, 100, 100))
    quads.append(_quad(0, 70, 100, 90))

    works = _refine(quads)
    big = next(work for work in works if 5 in work.members)
    small = next(work for work in works if 6 in work.members)

    assert big.overlap_unresolved and small.overlap_unresolved
    assert big.poly.area == pytest.approx(10000.0)
    assert small.poly.area == pytest.approx(2000.0)


def _refined_response(
    quads: list[DetectedQuad],
    width: int,
    height: int,
    direction: str = "ltr",
):
    from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines
    from nomikos_inference.architectures.ppocr_det.response import (
        build_refined_ppocr_det_response,
    )

    layout = layout_lines(quads, direction=direction)
    items = refine_to_lines(quads, layout)
    return build_refined_ppocr_det_response(
        width, height, quads, items, layout, reading_direction=direction
    )


def test_refined_builder_emits_one_region_per_column_in_reading_order() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
    ]
    response = _refined_response(quads, 1000, 500, "ltr")

    assert [block.external_id for block in response.blocks] == [
        "ppocr-det-region-1",
        "ppocr-det-region-2",
    ]
    assert [block.order for block in response.blocks] == [0, 1]
    assert response.blocks[0].box == {
        "points": [[100.0, 100.0], [400.0, 100.0], [400.0, 230.0], [100.0, 230.0]]
    }
    assert response.blocks[1].box == {
        "points": [[600.0, 100.0], [900.0, 100.0], [900.0, 230.0], [600.0, 230.0]]
    }
    assert [line.block_external_id for line in response.lines] == [
        "ppocr-det-region-1",
        "ppocr-det-region-1",
        "ppocr-det-region-2",
        "ppocr-det-region-2",
    ]


def test_refined_builder_rtl_reads_the_right_column_first() -> None:
    quads = [
        _quad(600, 100, 900, 130),
        _quad(100, 200, 400, 230),
        _quad(600, 200, 900, 230),
        _quad(100, 100, 400, 130),
    ]
    response = _refined_response(quads, 1000, 500, "rtl")

    assert response.blocks[0].box == {
        "points": [[600.0, 100.0], [900.0, 100.0], [900.0, 230.0], [600.0, 230.0]]
    }
    assert response.blocks[1].box == {
        "points": [[100.0, 100.0], [400.0, 100.0], [400.0, 230.0], [100.0, 230.0]]
    }
    assert [line.block_external_id for line in response.lines] == [
        "ppocr-det-region-1",
        "ppocr-det-region-1",
        "ppocr-det-region-2",
        "ppocr-det-region-2",
    ]


def test_refined_builder_puts_a_spanning_header_in_a_region_of_its_own() -> None:
    quads = [_quad(0, 50, 1900, 90)]
    quads.extend(_quad(100, y, 800, y + 30) for y in (200, 300, 400))
    quads.extend(_quad(1100, y, 1800, y + 30) for y in (200, 300, 400))
    response = _refined_response(quads, 2000, 600, "ltr")

    assert len(response.blocks) == 3
    header = next(line for line in response.lines if max(p[0] for p in line.points) == 1900.0)
    others = {line.block_external_id for line in response.lines if line is not header}
    assert header.block_external_id is not None
    assert header.block_external_id not in others
    # Regions are numbered by their first line's reading position.
    first_seen = list(dict.fromkeys(line.block_external_id for line in response.lines))
    assert [block.external_id for block in response.blocks] == first_seen


def test_refined_builder_single_column_yields_one_bounding_block() -> None:
    quads = _column(0, 200, [0, 40, 80])
    response = _refined_response(quads, 500, 300, "ltr")

    assert [block.external_id for block in response.blocks] == ["ppocr-det-region-1"]
    xs = [point[0] for line in response.lines for point in line.points]
    ys = [point[1] for line in response.lines for point in line.points]
    assert response.blocks[0].box == {
        "points": [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    }
    assert {line.block_external_id for line in response.lines} == {"ppocr-det-region-1"}
