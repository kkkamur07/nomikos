"""Refinement geometry helpers: baseline distance and the initial rule.

``mark_initials`` is the path taken when merging is switched off, so it is
exercised here directly and through ``refine_to_lines(merge=False)``.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.geometry import Work, baseline_distance
from nomikos_inference.architectures.ppocr_det.merge import mark_initials
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    return DetectedQuad(
        points=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        score=score,
    )


def _point_work(x: float, y: float) -> Work:
    return Work(
        poly=Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]),
        baseline=((x, y), (x, y)),
        score=0.5,
        members=(0,),
    )


def test_zero_length_baselines_measure_the_gap_between_them() -> None:
    # Far from the origin, at one spot: the old fallback returned the
    # midpoint's distance from the page origin (about 1414 here).
    assert baseline_distance(_point_work(1000, 1000), _point_work(1000, 1000)) == 0.0
    assert baseline_distance(_point_work(1000, 1000), _point_work(1003, 1004)) == 5.0


def test_baseline_distance_is_perpendicular_for_parallel_baselines() -> None:
    upper = Work(
        poly=Polygon([(0, 0), (100, 0), (100, 20), (0, 20)]),
        baseline=((0.0, 15.0), (100.0, 15.0)),
        score=0.5,
        members=(0,),
    )
    lower = Work(
        poly=Polygon([(40, 30), (140, 30), (140, 50), (40, 50)]),
        baseline=((140.0, 45.0), (40.0, 45.0)),
        score=0.5,
        members=(1,),
    )
    # Reversed direction is folded onto the first; the x shift does not count.
    assert baseline_distance(upper, lower) == 30.0


def _column_with_tall_quad() -> list[DetectedQuad]:
    quads = [_quad(100, y, 500, y + 20) for y in (0, 40, 80, 120)]
    # Beside the second line, sharing its row, three times its height.
    quads.append(_quad(40, 35, 95, 95))
    return quads


def _works(quads: list[DetectedQuad]) -> list[Work]:
    return [
        Work(poly=Polygon(quad.points), baseline=((0, 0), (1, 0)), score=quad.score, members=(i,))
        for i, quad in enumerate(quads)
    ]


def test_tall_row_mate_is_marked_initial_and_the_lines_are_not() -> None:
    quads = _column_with_tall_quad()
    layout = layout_lines(quads)
    works = _works(quads)

    mark_initials(works, quads, layout, height_ratio=2.0)

    assert [work.role for work in works] == ["line"] * 4 + ["initial"]


def test_height_ratio_decides_the_initial() -> None:
    quads = _column_with_tall_quad()
    layout = layout_lines(quads)
    works = _works(quads)

    # The tall quad is 3x its row mate: a ratio of 3 is not exceeded.
    mark_initials(works, quads, layout, height_ratio=3.0)

    assert all(work.role == "line" for work in works)


def test_lone_tall_quad_falls_back_to_the_builder_median() -> None:
    quads = [_quad(100, y, 500, y + 20) for y in (0, 40, 80)]
    # Alone in its row (no mates), 60 px against a 20 px builder median.
    quads.append(_quad(100, 200, 500, 260))
    layout = layout_lines(quads)
    works = _works(quads)

    mark_initials(works, quads, layout, height_ratio=2.0)

    assert works[3].role == "initial"
    assert [work.role for work in works[:3]] == ["line"] * 3


def test_unmerged_refinement_keeps_the_initial_role_and_spares_it_as_suspect() -> None:
    quads = _column_with_tall_quad()
    layout = layout_lines(quads)

    lines = refine_to_lines(quads, layout, merge=False, resolve=False)

    assert [line.members for line in lines] == [(i,) for i in range(5)]
    assert lines[4].role == "initial"
    assert not lines[4].suspect
    assert all(line.role == "line" for line in lines[:4])
