"""Poly output mode for the PP-OCRv6 detection segmenter.

Synthetic probability maps and hand-built quads only, no model needed. The
quad path stays byte-identical (same quads, scores and responses as quad
mode); the polygon follows the blob, merges union, cuts stay disjoint,
simplification caps at 64 points, failures fall back to the quad, and every
served polygon stays inside the page and about one line tall.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.polygons import (
    MAX_POLYGON_POINTS,
    _long_axis,
    intersect_outline,
    polyline_baseline,
    ring_area,
    simplify_outline,
    union_outlines,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad, detect_lines
from nomikos_inference.architectures.ppocr_det.ppocr_det import DEFAULT_BOX_TYPE, _box_type
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines
from nomikos_inference.architectures.ppocr_det.response import (
    build_ppocr_det_response,
    build_refined_ppocr_det_response,
    synthetic_baseline_points,
)


def _banana_map() -> np.ndarray:
    prob = np.zeros((160, 160), dtype=np.float32)
    cv2.ellipse(prob, (80, 80), (60, 30), 0, 10, 170, 1.0, 14)
    return prob


def _rect_map() -> np.ndarray:
    prob = np.zeros((1, 1, 128, 128), dtype=np.float32)
    prob[0, 0, 20:36, 20:110] = 1.0
    return prob


def _detect(prob: np.ndarray, box_type: str, **kwargs) -> list[DetectedQuad]:
    size = prob.shape[-1]
    return detect_lines(
        prob.reshape(prob.shape[-2], prob.shape[-1]),
        orig_width=size,
        orig_height=size,
        ratio_h=1.0,
        ratio_w=1.0,
        box_type=box_type,
        **kwargs,
    )


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return DetectedQuad(points=points, score=score, polygon=[list(p) for p in points])


def _point_to_segment(point: list[float], first: list[float], second: list[float]) -> float:
    point = np.asarray(point, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    leg = second - first
    length = float(np.linalg.norm(leg))
    if length <= 0:
        return float(np.linalg.norm(point - first))
    return abs(float(leg[0] * (first[1] - point[1]) - leg[1] * (first[0] - point[0]))) / length


def test_curved_blob_polygon_hugs_the_blob() -> None:
    quads = _detect(_banana_map(), "poly", box_thresh=0.2)

    assert len(quads) == 1
    quad = quads[0]
    assert quad.polygon is not None
    assert not quad.polygon_fallback
    assert len(quad.polygon) > 4
    ratio = Polygon(quad.polygon).area / Polygon(quad.points).area
    assert ratio <= 0.8


def test_straight_blob_baseline_matches_quad_baseline() -> None:
    quads = _detect(_rect_map(), "poly")
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(quads, layout, box_type="poly", page_width=128, page_height=128)
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    assert len(poly_items) == len(quad_items) == 1
    item, quad_item = poly_items[0], quad_items[0]
    assert item.members == quad_item.members
    assert len(item.baseline) >= 2
    first, second = quad_item.baseline
    # Interior samples sit on the quad baseline; the end samples may dip
    # toward the line centre where the 0.5 px outline keeps the rounded end
    # caps the old 1.0 px epsilon flattened, so only they are exempt.
    for point in item.baseline[1:-1]:
        assert _point_to_segment(point, first, second) <= 1.0
    for point in item.baseline:
        assert _point_to_segment(point, first, second) <= 6.0
    # Direction: the served polyline runs the same way as the quad baseline.
    poly_direction = np.asarray(item.baseline[-1]) - np.asarray(item.baseline[0])
    quad_direction = np.asarray(second) - np.asarray(first)
    assert float(poly_direction @ quad_direction) > 0
    # Extent: both ends reach the quad baseline ends within 2 px.
    _, axis, _, _ = _long_axis(quads[0].points)
    if float(axis @ quad_direction) < 0:
        axis = -axis
    origin = np.asarray(quads[0].points).mean(axis=0)
    quad_span = sorted(float((np.asarray(p) - origin) @ axis) for p in (first, second))
    poly_span = sorted(float((np.asarray(p) - origin) @ axis) for p in item.baseline)
    assert abs(poly_span[0] - quad_span[0]) <= 2.0
    assert abs(poly_span[-1] - quad_span[-1]) <= 2.0


def test_quad_mode_matches_poly_quads_and_is_deterministic() -> None:
    first = _detect(_rect_map(), "quad")
    second = _detect(_rect_map(), "quad")
    poly = _detect(_rect_map(), "poly")

    assert [quad.points for quad in first] == [quad.points for quad in second]
    assert [quad.points for quad in first] == [quad.points for quad in poly]
    assert [quad.score for quad in first] == [quad.score for quad in poly]
    assert all(quad.polygon is None for quad in first)
    assert all(len(quad.points) == 4 for quad in first)

    layout = layout_lines(first, direction="ltr")
    items = refine_to_lines(first, layout, box_type="quad")
    response = build_refined_ppocr_det_response(128, 128, first, items, layout)
    again = build_refined_ppocr_det_response(
        128, 128, first, refine_to_lines(first, layout, box_type="quad"), layout
    )
    assert response.model_dump_json() == again.model_dump_json()
    for line in response.lines:
        assert len(line.points) == 4
        assert len(line.baseline["points"]) == 2
        assert "polygon_fallback" not in line.source_metadata


def test_bad_box_type_raises() -> None:
    with pytest.raises(ValueError):
        _detect(_rect_map(), "circle")
    with pytest.raises(ValueError):
        _box_type({"box_type": "circle"})
    quads = _detect(_rect_map(), "quad")
    layout = layout_lines(quads, direction="ltr")
    with pytest.raises(ValueError):
        refine_to_lines(quads, layout, box_type="circle")
    with pytest.raises(ValueError):
        build_ppocr_det_response(128, 128, quads, box_type="circle")


def test_library_defaults_are_quad_and_served_default_is_poly() -> None:
    assert _box_type({}) == "quad"
    assert DEFAULT_BOX_TYPE == "poly"
    assert _box_type({}, DEFAULT_BOX_TYPE) == "poly"
    assert _detect(_rect_map(), "quad")[0].polygon is None
    default_detected = detect_lines(
        _rect_map().reshape(128, 128),
        orig_width=128,
        orig_height=128,
        ratio_h=1.0,
        ratio_w=1.0,
    )
    assert all(quad.polygon is None for quad in default_detected)


def test_merge_unions_touching_polygons() -> None:
    quads = [
        _quad(0, 0, 200, 20),
        _quad(0, 40, 200, 60),
        _quad(0, 80, 200, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(100, 120, 200, 140, score=0.6),
    ]
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(
        quads, layout, box_type="poly", page_width=300, page_height=300, vertical_growth=1.0
    )
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    merged = [item for item in poly_items if item.merged_from == 2]
    assert len(merged) == 1
    left = Polygon(quads[3].polygon)
    right = Polygon(quads[4].polygon)
    assert Polygon(merged[0].points).area == pytest.approx(left.area + right.area, rel=0.05)
    assert [item.members for item in poly_items] == [item.members for item in quad_items]


def test_merge_hulls_separated_fragments() -> None:
    quads = [
        _quad(0, 0, 200, 20),
        _quad(0, 40, 200, 60),
        _quad(0, 80, 200, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(103, 122, 220, 142, score=0.6),
    ]
    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(
        quads, layout, box_type="poly", page_width=300, page_height=300, vertical_growth=1.0
    )
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    merged = [item for item in items if item.merged_from == 2]
    assert len(merged) == 1
    # Simplification may shave sub-epsilon corners, so allow a 1 px slack.
    outline = Polygon(merged[0].points).buffer(1.0)
    assert outline.covers(Polygon(quads[3].polygon))
    assert outline.covers(Polygon(quads[4].polygon))
    # Containment: the served outline stays inside the merged quad. The
    # check runs over the whole outline area, not just its vertices, so a
    # hull edge bulging across the merge notch is caught.
    merged_quad = Polygon(next(item for item in quad_items if item.merged_from == 2).points)
    assert Polygon(merged[0].points).difference(merged_quad).area <= 0.5


def test_overlap_cut_leaves_polygons_disjoint() -> None:
    quads = [
        _quad(0, 0, 100, 15),
        _quad(0, 25, 100, 40),
        _quad(0, 50, 100, 65),
        _quad(0, 70, 100, 95),
        _quad(0, 85, 100, 110),
    ]
    layout = layout_lines(quads, direction="ltr")
    poly_items = refine_to_lines(
        quads, layout, box_type="poly", page_width=200, page_height=200, vertical_growth=1.0
    )
    quad_items = refine_to_lines(quads, layout, box_type="quad")

    pair = [item for item in poly_items if set(item.members) & {3, 4}]
    assert len(pair) == 2
    cut = intersect_outline(pair[0].points, pair[1].points)
    assert cut is None or ring_area(cut) == pytest.approx(0.0)
    assert [item.members for item in poly_items] == [item.members for item in quad_items]
    assert [item.role for item in poly_items] == [item.role for item in quad_items]


def test_simplify_caps_dense_rings_at_64_points() -> None:
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = [[100.0 + 80 * float(np.cos(a)), 100.0 + 80 * float(np.sin(a))] for a in angles]

    simplified = simplify_outline(circle, page_width=200, page_height=200)

    assert simplified is not None
    assert 4 <= len(simplified) <= MAX_POLYGON_POINTS
    assert all(simplified[i] != simplified[i + 1] for i in range(len(simplified) - 1))


def test_triangle_blob_falls_back_to_the_quad() -> None:
    prob = np.zeros((128, 128), dtype=np.float32)
    cv2.fillPoly(prob, [np.array([[10, 110], [120, 110], [65, 10]])], 1.0)
    quads = detect_lines(
        prob, orig_width=128, orig_height=128, ratio_h=1.0, ratio_w=1.0, box_type="poly"
    )

    assert len(quads) == 1
    assert quads[0].polygon_fallback
    assert quads[0].polygon == quads[0].points

    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(quads, layout, box_type="poly", page_width=128, page_height=128)
    response = build_refined_ppocr_det_response(128, 128, quads, items, layout, box_type="poly")
    assert response.lines[0].source_metadata["polygon_fallback"] is True


def test_polygons_stay_inside_the_page_and_about_one_line_tall() -> None:
    quads = _detect(_banana_map(), "poly", box_thresh=0.2) + _detect(_rect_map(), "poly")
    layout = layout_lines(quads, direction="ltr")
    items = refine_to_lines(
        quads, layout, box_type="poly", page_width=160, page_height=160, vertical_growth=1.0
    )

    assert len(items) == 2
    quads_by_member = {member: quads[member] for item in items for member in item.members}
    for item in items:
        for x, y in item.points:
            assert 0.0 <= x <= 160.0
            assert 0.0 <= y <= 160.0
        assert 4 <= len(item.points) <= MAX_POLYGON_POINTS
        poly_height = max(p[1] for p in item.points) - min(p[1] for p in item.points)
        quad = quads_by_member[item.members[0]]
        quad_height = max(p[1] for p in quad.points) - min(p[1] for p in quad.points)
        assert poly_height <= quad_height + 2.0


def _rotated_quad(base: list[list[float]], angle: float) -> list[list[float]]:
    points = np.asarray(base, dtype=np.float64)
    centre = points.mean(axis=0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return [list(point) for point in (points - centre) @ rotation.T + centre]


def test_poly_baseline_direction_follows_quad_baseline() -> None:
    base = [[350.0, 100.0], [650.0, 100.0], [650.0, 140.0], [350.0, 140.0]]
    for angle in np.linspace(-0.6, 0.6, 200):
        quad = _rotated_quad(base, float(angle))
        quad_baseline = synthetic_baseline_points(quad, 0.75)
        polyline = polyline_baseline(quad, quad, 0.75, baseline=quad_baseline)

        assert polyline is not None and len(polyline) >= 2
        poly_direction = np.asarray(polyline[-1]) - np.asarray(polyline[0])
        quad_direction = np.asarray(quad_baseline[1]) - np.asarray(quad_baseline[0])
        assert float(poly_direction @ quad_direction) > 0


def test_poly_baseline_vertical_line_is_not_reversed() -> None:
    quad = [[50.0, 0.0], [60.0, 0.0], [60.0, 200.0], [50.0, 200.0]]
    quad_baseline = synthetic_baseline_points(quad, 0.75)
    polyline = polyline_baseline(quad, quad, 0.75, baseline=quad_baseline)

    assert polyline is not None and len(polyline) >= 2
    poly_direction = np.asarray(polyline[-1]) - np.asarray(polyline[0])
    quad_direction = np.asarray(quad_baseline[1]) - np.asarray(quad_baseline[0])
    assert float(poly_direction @ quad_direction) >= -1e-9


def test_polygon_cut_never_vetoes_quad_cut() -> None:
    upper = _quad(100, 100, 300, 140)
    lower_points = [[100.0, 125.0], [300.0, 125.0], [300.0, 185.0], [100.0, 185.0]]
    lower = DetectedQuad(
        points=lower_points,
        score=0.7,
        polygon=[[100.0, 125.0], [300.0, 125.0], [300.0, 150.0], [100.0, 150.0]],
    )
    quads = [upper, lower]
    layout = layout_lines(quads, direction="ltr")
    quad_items = refine_to_lines(quads, layout, box_type="quad")
    poly_items = refine_to_lines(
        quads, layout, box_type="poly", page_width=400, page_height=400, vertical_growth=1.0
    )

    assert [item.members for item in poly_items] == [item.members for item in quad_items]
    assert [item.role for item in poly_items] == [item.role for item in quad_items]
    assert [item.overlap_unresolved for item in poly_items] == [
        item.overlap_unresolved for item in quad_items
    ]
    assert not any(item.overlap_unresolved for item in poly_items)
    by_members = {member: item for item in quad_items for member in item.members}
    for poly_item in poly_items:
        quad_item = by_members[poly_item.members[0]]
        assert sorted(map(tuple, poly_item.points)) == pytest.approx(
            sorted(map(tuple, quad_item.points))
        )
    lower_poly = next(item for item in poly_items if item.members == (1,))
    assert lower_poly.polygon_fallback

    quad_response = build_refined_ppocr_det_response(400, 400, quads, quad_items, layout)
    poly_response = build_refined_ppocr_det_response(
        400, 400, quads, poly_items, layout, box_type="poly"
    )
    quad_meta = next(
        line.source_metadata for line in quad_response.lines if line.source_metadata["score"] == 0.7
    )
    poly_meta = next(
        line.source_metadata for line in poly_response.lines if line.source_metadata["score"] == 0.7
    )
    assert set(poly_meta) - set(quad_meta) == {"polygon_fallback"}
    assert poly_meta["polygon_fallback"] is True


def test_polyline_extent_reaches_quad_ends() -> None:
    quad = [[350.0, 100.0], [650.0, 100.0], [650.0, 140.0], [350.0, 140.0]]
    tapered = [
        [360.0, 112.0],
        [500.0, 84.0],
        [640.0, 112.0],
        [640.0, 128.0],
        [500.0, 156.0],
        [360.0, 128.0],
    ]
    for outline in (quad, tapered):
        quad_baseline = synthetic_baseline_points(quad, 0.75)
        polyline = polyline_baseline(outline, quad, 0.75, baseline=quad_baseline)

        assert polyline is not None and len(polyline) >= 2
        _, axis, _, _ = _long_axis(quad)
        quad_direction = np.asarray(quad_baseline[1]) - np.asarray(quad_baseline[0])
        if float(axis @ quad_direction) < 0:
            axis = -axis
        origin = np.asarray(quad).mean(axis=0)
        quad_span = sorted(float((np.asarray(point) - origin) @ axis) for point in quad_baseline)
        poly_span = sorted(float((np.asarray(point) - origin) @ axis) for point in polyline)
        assert abs(poly_span[0] - quad_span[0]) <= 2.0
        assert abs(poly_span[-1] - quad_span[-1]) <= 2.0


def test_served_clipper_calls_keep_fractions() -> None:
    fractional = [[0.0, 0.0], [10.7, 0.2], [10.7, 5.3], [0.4, 5.3]]

    simplified = simplify_outline(fractional)
    assert simplified is not None
    assert _ring_deviation(simplified, fractional) <= 0.01

    united = union_outlines([fractional])
    assert united is not None
    assert _ring_deviation(united, fractional) <= 0.01

    cut = intersect_outline(
        fractional, [[-10.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-10.0, 10.0]]
    )
    assert cut is not None
    assert _ring_deviation(cut, fractional) <= 0.01


def _ring_deviation(found: list[list[float]], want: list[list[float]]) -> float:
    """Largest corner distance from each found point to the wanted ring."""
    worst = 0.0
    for point in found:
        best = min(float(np.hypot(point[0] - x, point[1] - y)) for x, y in want)
        worst = max(worst, best)
    return worst


def test_merged_outline_stays_inside_merged_quad() -> None:
    quads = [
        _quad(0, 0, 220, 20),
        _quad(0, 40, 220, 60),
        _quad(0, 80, 220, 100),
        _quad(0, 120, 100, 140, score=0.8),
        _quad(104, 128, 220, 148, score=0.6),
    ]
    layout = layout_lines(quads, direction="ltr")
    quad_items = refine_to_lines(quads, layout, box_type="quad")
    poly_items = refine_to_lines(
        quads, layout, box_type="poly", page_width=300, page_height=300, vertical_growth=1.0
    )

    merged = [item for item in poly_items if item.merged_from == 2]
    assert len(merged) == 1
    merged_quad = Polygon(next(item for item in quad_items if item.merged_from == 2).points)
    assert Polygon(merged[0].points).difference(merged_quad).area <= 0.5


def test_unrefined_poly_threads_outline_tolerance() -> None:
    teeth = 25
    tall_outline = [[200.0 * i / (teeth - 1), 0.0 if i % 2 == 0 else 2.0] for i in range(teeth)] + [
        [200.0 - 200.0 * i / (teeth - 1), 300.0 if i % 2 == 0 else 298.0] for i in range(teeth)
    ]
    quads = [
        DetectedQuad(
            points=[[0.0, 0.0], [200.0, 0.0], [200.0, 300.0], [0.0, 300.0]],
            score=0.9,
            polygon=tall_outline,
        ),
        _quad(0, 320, 200, 440, score=0.8),
        _quad(0, 460, 200, 580, score=0.7),
    ]
    # Growth off so the served points equal the simplified outline exactly.
    response = build_ppocr_det_response(
        200, 600, quads, box_type="poly", vertical_growth=1.0, outline_tolerance_px=0.5
    )

    tall_line = next(line for line in response.lines if line.source_metadata["score"] == 0.9)
    expected = simplify_outline(
        tall_outline, outline_tolerance_px=0.5, page_width=200, page_height=600
    )
    assert expected is not None
    assert len(expected) > 4
    assert tall_line.points == expected
    coarse = build_ppocr_det_response(
        200, 600, quads, box_type="poly", vertical_growth=1.0, outline_tolerance_px=4.0
    )
    coarse_line = next(line for line in coarse.lines if line.source_metadata["score"] == 0.9)
    assert len(coarse_line.points) < len(tall_line.points)


def test_simplify_guarantees_64_point_cap(monkeypatch) -> None:
    import cv2 as cv2_module

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = np.array(
        [[100.0 + 80 * float(np.cos(a)), 100.0 + 80 * float(np.sin(a))] for a in angles],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        cv2_module,
        "approxPolyDP",
        lambda contour, epsilon, closed: np.tile(np.asarray(contour), (3, 1)),
    )
    simplified = simplify_outline(circle)

    assert simplified is not None
    assert len(simplified) <= MAX_POLYGON_POINTS
