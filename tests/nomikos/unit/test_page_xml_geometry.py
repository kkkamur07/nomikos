"""What ``Coords`` and ``Baseline`` come out as, for every shape a line is stored in.

The export used to write the mask polygon into ``Baseline``, which made the files
unusable as segmenter training data. These tests pin the four cases that produced
that geometry and the derived baseline that replaces it.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.annotation.application.page_xml_geometry import (
    BASELINE_HEIGHT_FRACTION,
    derive_baseline,
    quantize_baseline,
    quantize_polygon,
    resolve_line_geometry,
)
from backend.document.infrastructure.orm_models import LineGeometryKind, LineSource


def _line(*, points, mask=None, baseline=None, kind=LineGeometryKind.polygon):
    return SimpleNamespace(
        id=uuid4(),
        points=points,
        mask=mask,
        baseline=baseline,
        kind=kind,
        source=LineSource.manual,
        manual_geometry=False,
        transcriptions=[],
    )


# A twenty-pixel-tall box with a real baseline two thirds down it.
BOX = [[10.0, 100.0], [110.0, 100.0], [110.0, 120.0], [10.0, 120.0]]
BASELINE = [[12.0, 114.0], [108.0, 115.0]]


def test_mask_and_baseline_line_keeps_both_verbatim() -> None:
    geometry = resolve_line_geometry(
        _line(points=[[0.0, 0.0]], mask={"points": BOX}, baseline={"points": BASELINE})
    )

    assert geometry.coords == BOX
    assert geometry.baseline == BASELINE
    assert geometry.baseline_derived is False


def test_line_with_only_points_uses_them_as_coords_and_derives_a_baseline() -> None:
    # ``create_part_line`` stores ``baseline = {"points": points}`` when the client
    # sends none, so "only points" reaches here as a baseline that is the polygon.
    geometry = resolve_line_geometry(_line(points=BOX, baseline={"points": BOX}))

    assert geometry.coords == BOX
    assert geometry.baseline_derived is True
    assert [point[1] for point in geometry.baseline] == [114, 114]


def test_hand_drawn_box_with_no_baseline_at_all_gets_one() -> None:
    geometry = resolve_line_geometry(_line(points=BOX, baseline={}))

    assert geometry.baseline_derived is True
    assert len(geometry.baseline) >= 2


def test_a_two_point_rectangle_is_squared_off_into_a_polygon() -> None:
    geometry = resolve_line_geometry(
        _line(
            points=[[10.0, 100.0], [110.0, 120.0]], baseline=None, kind=LineGeometryKind.rectangle
        )
    )

    assert geometry.coords == [[10.0, 100.0], [110.0, 100.0], [110.0, 120.0], [10.0, 120.0]]
    assert len(geometry.baseline) >= 2


def test_a_baseline_of_one_repeated_point_is_not_a_baseline() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[5.0, 5.0], [5.0, 5.0]]})
    )

    assert geometry.baseline_derived is True


def test_a_placeholder_baseline_along_the_polygon_bottom_is_replaced() -> None:
    # What ``default_baseline_from_polygon`` leaves behind on the bulk PUT /lines
    # path: a segment on the polygon's lowest edge, a third of a line-height below
    # where a kraken baseline goes.
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[10.0, 120.0], [110.0, 120.0]]})
    )

    assert geometry.baseline_derived is True
    assert all(point[1] < 120.0 for point in geometry.baseline)


def test_a_measured_baseline_just_above_the_bottom_edge_is_kept() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[10.0, 118.0], [110.0, 118.0]]})
    )

    assert geometry.baseline_derived is False


def test_a_stored_baseline_is_reordered_left_to_right() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[108.0, 115.0], [12.0, 114.0]]})
    )

    assert geometry.baseline_derived is False
    assert [point[0] for point in geometry.baseline] == [12.0, 108.0]


def test_a_baseline_overhanging_its_polygon_is_clipped_with_its_slope_kept() -> None:
    # kraken ships baselines that run past the simplified boundary polygon: 199 of
    # 1010 lines on the 2026-09-18 Coptic snapshots.
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[0.0, 108.0], [120.0, 118.0]]})
    )

    assert geometry.baseline_derived is False
    # Cut at x=10 and x=110, with y interpolated along the original slope and
    # then rounded: 108 + 10/12 -> 109, 108 + 110/12 -> 117.
    assert geometry.baseline == [[10, 109], [110, 117]]


def test_a_baseline_below_its_polygon_is_clamped_into_it() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[12.0, 118.0], [108.0, 128.0]]})
    )

    assert [point[1] for point in geometry.baseline] == [118, 120]


def test_a_baseline_entirely_beside_its_polygon_is_replaced_not_clipped() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[900.0, 114.0], [980.0, 114.0]]})
    )

    assert geometry.baseline_derived is True


def test_malformed_geometry_falls_back_rather_than_raising() -> None:
    geometry = resolve_line_geometry(
        _line(points=BOX, mask={"points": "not-a-list"}, baseline={"points": [[1.0, 2.0, 3.0]]})
    )

    assert geometry.coords == BOX
    assert geometry.baseline_derived is True


def test_derived_baseline_sits_at_the_measured_fraction_of_the_height() -> None:
    baseline = derive_baseline(BOX)

    expected = 100.0 + BASELINE_HEIGHT_FRACTION * 20.0
    assert all(point[1] == pytest.approx(expected) for point in baseline)
    assert expected == pytest.approx(113.6)


def test_derived_baseline_follows_a_sloping_polygon() -> None:
    # A line that climbs ten pixels from left to right: a flat baseline would leave
    # the polygon at one end.
    sloped = [[0.0, 100.0], [100.0, 90.0], [100.0, 110.0], [0.0, 120.0]]

    baseline = derive_baseline(sloped)

    assert len(baseline) >= 2
    assert baseline[0][1] > baseline[-1][1]
    for x, y in baseline:
        top = 100.0 - 0.1 * x
        bottom = 120.0 - 0.1 * x
        assert top <= y <= bottom


def test_derived_baseline_x_is_strictly_increasing() -> None:
    baseline = derive_baseline([[0.0, 0.0], [50.0, 2.0], [100.0, 0.0], [100.0, 30.0], [0.0, 30.0]])

    xs = [point[0] for point in baseline]
    assert xs == sorted(xs)
    assert len(set(xs)) == len(xs)


def test_a_straight_line_collapses_back_to_two_points() -> None:
    assert len(derive_baseline(BOX)) == 2


def test_a_degenerate_polygon_still_yields_two_points() -> None:
    assert len(derive_baseline([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])) == 2


def test_a_polygon_comes_out_as_whole_pixels() -> None:
    """PAGE's PointsType is integer pairs, and kraken calls ``int()`` on every one."""
    assert quantize_polygon([[10.4, 100.4], [110.6, 100.4], [110.6, 120.6], [9.6, 120.6]]) == [
        [10, 100],
        [111, 100],
        [111, 121],
        [10, 121],
    ]


def test_rounding_a_sliver_polygon_flat_squares_it_off() -> None:
    """A one-pixel-tall boundary can round to a line, which is not a polygon any more."""
    coords = quantize_polygon([[10.1, 100.2], [110.4, 100.1], [110.2, 100.4], [10.3, 100.3]])

    assert len(coords) >= 3
    assert {point[1] for point in coords} == {100, 101}


def test_a_polygon_is_clamped_into_the_image() -> None:
    coords = quantize_polygon(
        [[-4.0, -2.0], [900.0, -2.0], [900.0, 900.0], [-4.0, 900.0]], bounds=(600, 800)
    )

    assert coords == [[0, 0], [599, 0], [599, 799], [0, 799]]


def test_a_polygon_pinned_to_the_page_edge_keeps_three_points() -> None:
    coords = quantize_polygon([[599.4, 799.2], [599.6, 799.4]], bounds=(600, 800))

    assert len(coords) >= 3
    assert all(0 <= x <= 599 and 0 <= y <= 799 for x, y in coords)


_BOX = [[10, 100], [110, 100], [110, 120], [10, 120]]


def test_baseline_vertices_that_round_onto_one_x_are_dropped() -> None:
    """Rounding must not make the polyline double back on itself."""
    baseline = quantize_baseline([[12.4, 114.0], [12.4, 115.0], [108.0, 115.0]], _BOX)

    assert baseline == [[12, 114], [108, 115]]
    assert all(
        later[0] > earlier[0] for earlier, later in zip(baseline, baseline[1:], strict=False)
    )


def test_a_baseline_rounded_outside_its_polygon_is_pulled_back_in() -> None:
    baseline = quantize_baseline([[9.6, 120.4], [110.4, 120.4]], _BOX)

    assert baseline == [[10, 120], [110, 120]]


def test_a_baseline_that_collapses_to_one_point_is_rebuilt() -> None:
    """Better a derived segment across the polygon than a one-point Baseline."""
    baseline = quantize_baseline([[12.1, 114.0], [12.2, 114.0]], _BOX)

    assert baseline == [[10, 114], [110, 114]]


def test_a_baseline_in_a_polygon_one_pixel_wide_still_has_two_points() -> None:
    baseline = quantize_baseline([[10.0, 100.0]], [[10, 100], [10, 101], [11, 101]])

    assert len(baseline) == 2
    assert baseline[1][0] > baseline[0][0]


def test_coords_follow_corrected_points_rather_than_a_stale_mask() -> None:
    # A vertex edit sends ``points`` and leaves the model's mask behind; the file has
    # to carry the outline the human left, not the one they corrected.
    model_outline = [[10.0, 100.0], [110.0, 100.0], [110.0, 130.0], [10.0, 130.0]]
    geometry = resolve_line_geometry(
        _line(points=BOX, mask={"points": model_outline}, baseline={"points": BASELINE})
    )

    assert geometry.coords == BOX


def test_a_baseline_that_rounding_collapses_is_labelled_derived() -> None:
    # Both stored x values round to 12, so no measured baseline survives rounding;
    # whatever replaces it is a guess and has to say so.
    geometry = resolve_line_geometry(
        _line(points=BOX, baseline={"points": [[12.1, 114.0], [12.2, 115.0]]})
    )

    assert geometry.baseline_derived is True
    assert len(geometry.baseline) >= 2
