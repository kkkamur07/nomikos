"""Vertical growth of served PP-OCRv6 line polygons.

Hand-built quads and outlines only, no model needed. Growth scales each
outline across its own line direction about its centre line, splits
neighbour overlaps by nearest ungrown outline, and never touches quads or
baselines. Each test names the bug it would catch.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon

from nomikos_inference.architectures.ppocr_det.polygons import (
    MAX_POLYGON_POINTS,
    _contested_losses,
    grow_outlines_vertical,
    ring_area,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad, detect_lines
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import refine_to_lines


def _rect(xmin: float, ymin: float, xmax: float, ymax: float) -> list[list[float]]:
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def _quad(xmin: float, ymin: float, xmax: float, ymax: float, score: float = 0.9) -> DetectedQuad:
    points = _rect(xmin, ymin, xmax, ymax)
    return DetectedQuad(points=points, score=score, polygon=[list(p) for p in points])


def _axis_extent(points: list[list[float]], direction: np.ndarray) -> tuple[float, float]:
    unit = direction / float(np.linalg.norm(direction))
    proj = [float(np.dot(np.asarray(p), unit)) for p in points]
    return min(proj), max(proj)


def test_growth_is_vertical_only() -> None:
    """Catches uniform scaling: the along-line extent must not move."""
    outline = _rect(0.0, 0.0, 300.0, 20.0)
    quad = _rect(0.0, 0.0, 300.0, 20.0)

    (grown,) = grow_outlines_vertical([outline], [quad], factor=1.35)

    along = np.array([1.0, 0.0])
    before = _axis_extent(outline, along)
    after = _axis_extent(grown, along)
    assert abs(after[0] - before[0]) <= 1.0
    assert abs(after[1] - before[1]) <= 1.0
    across = np.array([0.0, 1.0])
    before_across = _axis_extent(outline, across)
    after_across = _axis_extent(grown, across)
    assert (after_across[1] - after_across[0]) == pytest.approx(
        1.35 * (before_across[1] - before_across[0]), abs=1.0
    )


def test_slanted_line_grows_perpendicular_to_its_direction() -> None:
    """Catches page-axis growth: a 30 degree line must grow across itself."""
    angle = math.radians(30.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    base = np.asarray(_rect(-150.0, -10.0, 150.0, 10.0))
    centre = np.array([200.0, 200.0])
    outline = [(point @ rotation.T + centre).tolist() for point in base]
    quad = [list(p) for p in outline]

    (grown,) = grow_outlines_vertical([outline], [quad], factor=1.35)

    along = rotation @ np.array([1.0, 0.0])
    across = rotation @ np.array([0.0, 1.0])
    before_along = _axis_extent(outline, along)
    after_along = _axis_extent(grown, along)
    assert abs(after_along[0] - before_along[0]) <= 1.0
    assert abs(after_along[1] - before_along[1]) <= 1.0
    before_across = _axis_extent(outline, across)
    after_across = _axis_extent(grown, across)
    assert (after_across[1] - after_across[0]) == pytest.approx(
        1.35 * (before_across[1] - before_across[0]), abs=1.5
    )


def test_close_lines_never_overlap_and_boundary_lies_between() -> None:
    """Catches a missing neighbour split: grown pair must be disjoint."""
    upper = _rect(0.0, 0.0, 200.0, 20.0)
    lower = _rect(0.0, 26.0, 200.0, 46.0)
    grown_upper, grown_lower = grow_outlines_vertical(
        [upper, lower], [upper, lower], factor=1.35, page_width=200.0, page_height=60.0
    )

    assert ring_area(grown_upper) > 0
    assert ring_area(grown_lower) > 0
    assert Polygon(grown_upper).intersection(Polygon(grown_lower)).area == pytest.approx(0.0)
    # Each line keeps its own ink: the grown outline covers the ungrown one.
    assert Polygon(grown_upper).covers(Polygon(upper).buffer(-1.0))
    assert Polygon(grown_lower).covers(Polygon(lower).buffer(-1.0))
    # The split runs in the gap: below the lower edge of the upper ungrown
    # outline and above the upper edge of the lower one.
    assert max(p[1] for p in grown_upper) < 26.0
    assert min(p[1] for p in grown_lower) > 20.0


def _block(rows: int, cols: int, top: int, bottom: int) -> np.ndarray:
    """Boolean mask with rows top (inclusive) to bottom (exclusive) filled."""
    mask = np.zeros((rows, cols), dtype=bool)
    mask[top:bottom, :] = True
    return mask


def _distances(mask: np.ndarray) -> np.ndarray:
    """Outside distance of a mask with the production transform settings."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def _kept(zone: np.ndarray, own: np.ndarray, other: np.ndarray, other_index: int, index: int):
    return zone & ~_contested_losses(_distances(own), _distances(other), other_index, index)


def test_tie_row_belongs_to_the_lower_index() -> None:
    """Catches ties kept by both lines: one strip lands in two crops.

    Mirror-symmetric plains (rows 0-9 against rows 19-28) give bit-exact
    equal distances along row 14; the grown bands overlap over rows 10-18.
    """
    own = _block(29, 40, 0, 10)
    other = _block(29, 40, 19, 29)
    grown_own = _block(29, 40, 0, 19)
    grown_other = _block(29, 40, 10, 29)
    zone = grown_own & grown_other

    assert (_distances(own)[14, :] == _distances(other)[14, :]).all()
    kept_own = _kept(zone, own, other, 1, 0)
    kept_other = _kept(zone, other, own, 0, 1)
    assert not (kept_own & kept_other).any()
    assert (kept_own | kept_other == zone).all()
    assert kept_own[14, :].all()
    assert not kept_other[14, :].any()


def test_three_stacked_lines_share_no_pixels() -> None:
    """Catches tie strips between every neighbour pair of a three stack."""
    plains = [_block(47, 40, top, top + 10) for top in (0, 19, 38)]
    growns = [_block(47, 40, 0, 19), _block(47, 40, 10, 38), _block(47, 40, 29, 47)]
    kept = []
    for index in range(3):
        keep = growns[index].copy()
        for other in range(3):
            if other == index:
                continue
            zone = growns[index] & growns[other]
            if zone.any():
                keep = keep & ~_contested_losses(
                    _distances(plains[index]), _distances(plains[other]), other, index
                )
        kept.append(keep)
    for left in range(3):
        for right in range(left + 1, 3):
            assert not (kept[left] & kept[right]).any()
    # Tie row 14 belongs to line 0, tie row 33 to line 1.
    assert kept[0][14, :].all() and not kept[1][14, :].any()
    assert kept[1][33, :].all() and not kept[2][33, :].any()


def test_isolated_line_grows_by_the_full_factor() -> None:
    """Catches skipped growth: an alone line gains the whole 1.35 area."""
    outline = _rect(0.0, 0.0, 200.0, 20.0)
    quad = _rect(0.0, 0.0, 200.0, 20.0)

    (grown,) = grow_outlines_vertical([outline], [quad], factor=1.35)

    assert Polygon(grown).area == pytest.approx(Polygon(outline).area * 1.35, rel=0.03)


def test_growth_one_is_the_ungrown_polygon() -> None:
    """Catches a growth path that reshapes even at factor 1.0."""
    outline = [[0.0, 0.0], [103.0, 1.0], [200.0, 0.0], [200.0, 20.0], [100.0, 21.0], [0.0, 20.0]]
    quad = _rect(0.0, 0.0, 200.0, 20.0)

    assert grow_outlines_vertical([outline], [quad], factor=1.0)[0] == outline


def test_grown_polygons_are_valid() -> None:
    """Catches ragged output: simple, inside the page, capped, integer."""
    outlines = [
        _rect(0.0, 0.0, 200.0, 20.0),
        _rect(5.0, 30.0, 195.0, 48.0),
        _rect(0.0, 55.0, 200.0, 75.0),
    ]
    grown = grow_outlines_vertical(
        outlines, outlines, factor=1.35, page_width=200.0, page_height=80.0
    )

    assert len(grown) == 3
    for points in grown:
        assert 4 <= len(points) <= MAX_POLYGON_POINTS
        assert Polygon(points).is_valid
        for x, y in points:
            assert x == round(x)
            assert y == round(y)
            assert 0.0 <= x <= 200.0
            assert 0.0 <= y <= 80.0


def test_growth_never_moves_baselines() -> None:
    """Catches growth feeding back into baselines: they match growth 1.0."""
    quads = [_quad(0, 0, 200, 20), _quad(0, 30, 200, 50), _quad(0, 60, 200, 80)]
    layout = layout_lines(quads, direction="ltr")

    grown = refine_to_lines(
        quads, layout, box_type="poly", page_width=200, page_height=100, vertical_growth=1.35
    )
    plain = refine_to_lines(
        quads, layout, box_type="poly", page_width=200, page_height=100, vertical_growth=1.0
    )

    assert [item.members for item in grown] == [item.members for item in plain]
    for with_growth, without in zip(grown, plain, strict=True):
        assert with_growth.baseline == without.baseline


def test_quad_mode_ignores_the_new_geometry_params() -> None:
    """Catches new params leaking into the quad path, which stays 0.4.1."""
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[10:26, 8:56] = 1.0
    kwargs = {"orig_width": 64, "orig_height": 64, "ratio_h": 1.0, "ratio_w": 1.0}

    plain = detect_lines(prob, box_type="quad", **kwargs)
    tuned = detect_lines(prob, box_type="quad", contour_tolerance_px=2.0, **kwargs)
    assert [q.points for q in plain] == [q.points for q in tuned]

    quads = [_quad(0, 0, 200, 20), _quad(0, 30, 200, 50)]
    layout = layout_lines(quads, direction="ltr")
    first = refine_to_lines(quads, layout, box_type="quad")
    second = refine_to_lines(
        quads,
        layout,
        box_type="quad",
        outline_tolerance_px=2.0,
        vertical_growth=2.0,
    )
    assert [item.points for item in first] == [item.points for item in second]


def test_served_params_reject_out_of_range_values(tmp_path) -> None:
    """Catches unvalidated geometry params reaching the offsetter."""
    from nomikos_inference.architectures.ppocr_det.ppocr_det import run_ppocr_det_segment

    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    with pytest.raises(ValueError, match="contour_tolerance_px"):
        run_ppocr_det_segment(b"", model_path=model, params={"contour_tolerance_px": -1.0})
    with pytest.raises(ValueError, match="outline_tolerance_px"):
        run_ppocr_det_segment(b"", model_path=model, params={"outline_tolerance_px": 11.0})
    with pytest.raises(ValueError, match="vertical_growth"):
        run_ppocr_det_segment(b"", model_path=model, params={"vertical_growth": 1.0 - 1e-6})
    with pytest.raises(ValueError, match="vertical_growth"):
        run_ppocr_det_segment(b"", model_path=model, params={"vertical_growth": 2.5})
