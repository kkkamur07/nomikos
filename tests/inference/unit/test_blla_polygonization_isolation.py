"""Per-line isolation for BLLA polygonization.

``architectures.isolation`` says a single degenerate contour must not discard
the rest of the page. Polygonization is where BLLA produces contours, and it
raises by design, so these tests build the degenerate geometry itself rather
than stubbing the failure in.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely import geometry as geom

import nomikos_inference.architectures.blla.blla_decoder as decoder
from nomikos_inference.architectures.blla.blla_decoder import decode_blla_heatmaps
from nomikos_inference.architectures.blla.blla_decoder.polygon import (
    _intersection_ring,
    calculate_polygonal_environment,
)

PAGE_SIZE = (200, 120)

# Verified degenerate against the real polygonizer: a baseline pinned to column
# zero casts its upper ray straight off the page, so ``_calc_roi`` finds no
# boundary intersection and raises.
UNPOLYGONIZABLE_BASELINE = [[0, 20], [0, 90]]
# Vertical but one column in: every point of its seam patch shares a column, so
# the seam scale used to divide by a zero-width span.
SINGLE_COLUMN_BASELINE = [[1, 20], [1, 90]]
GOOD_BASELINES = [[[20, 30], [180, 30]], [[20, 95], [180, 95]]]


def _page_features() -> tuple[np.ndarray, np.ndarray]:
    width, height = PAGE_SIZE
    return np.zeros((height, width), dtype=np.float64), np.asarray(PAGE_SIZE, dtype=float) - 1


def _decode(monkeypatch: pytest.MonkeyPatch, baselines: list[list[list[int]]]) -> list:
    """Run the reference decode path over exactly ``baselines``.

    Only the *source* of the baselines is stubbed. The polygonizer, the seam
    solver, and the reading order are the production ones, so the failures
    these tests isolate are the failures production hits.
    """
    width, height = PAGE_SIZE
    monkeypatch.setattr(decoder, "vectorize_lines", lambda *_, **__: [list(b) for b in baselines])
    monkeypatch.setattr(decoder, "vectorize_regions", lambda *_, **__: [])
    return decode_blla_heatmaps(
        np.zeros((4, height // 4, width // 4), dtype=np.float32),
        image_size=PAGE_SIZE,
        threshold=0.17,
        raw_logits=True,
        scaled_gray=np.zeros((height, width), dtype=np.uint8),
    )


def test_undecodable_baseline_costs_its_own_line_not_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baselines = [GOOD_BASELINES[0], UNPOLYGONIZABLE_BASELINE, GOOD_BASELINES[1]]

    decoded = _decode(monkeypatch, baselines)

    assert len(decoded) == 2
    assert sorted(line.baseline[0][1] for line in decoded) == [30, 95]
    assert all(len(line.polygon) >= 4 for line in decoded)


def test_a_page_of_nothing_but_failures_reraises_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="No intersection with boundaries"):
        _decode(monkeypatch, [UNPOLYGONIZABLE_BASELINE, UNPOLYGONIZABLE_BASELINE])


def test_a_page_with_no_baselines_is_empty_rather_than_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _decode(monkeypatch, []) == []


def test_single_column_line_environment_does_not_divide_by_zero() -> None:
    features, bounds = _page_features()

    polygon = calculate_polygonal_environment(
        baseline=SINGLE_COLUMN_BASELINE,
        supplementary_objects=[],
        image_features=features,
        bounds=bounds,
    )

    assert len(polygon) >= 4


def test_single_column_line_survives_alongside_its_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = _decode(monkeypatch, [GOOD_BASELINES[0], SINGLE_COLUMN_BASELINE])

    assert len(decoded) == 2


def test_multi_part_intersection_keeps_its_largest_ring() -> None:
    # Two squares joined by a hairline: intersecting with a band that misses the
    # hairline splits the result into a MultiPolygon, whose ``.boundary`` is a
    # MultiLineString with no ``.coords``.
    left = geom.box(0, 0, 10, 10)
    right = geom.box(30, 0, 60, 10)
    roi = left.union(right).union(geom.box(10, 4, 30, 5))
    clip = geom.box(-5, -5, 65, 3)

    ring = _intersection_ring(roi, clip)

    assert ring.shape[1] == 2
    xs = ring[:, 0]
    # The 30-wide part wins over the 10-wide one.
    assert xs.min() == 30
    assert xs.max() == 60


# `test_single_part_intersection_still_returns_the_plain_ring` stood here. Its assertion was
# `np.asarray(roi.intersection(clip).boundary.coords, dtype=int).tolist() == ring.tolist()`
# -- a line-for-line restatement of `_intersection_ring`'s own single-part branch, so it
# agrees with the implementation by construction whatever that implementation does.


def test_empty_intersection_yields_no_ring_rather_than_raising() -> None:
    ring = _intersection_ring(geom.box(0, 0, 1, 1), geom.box(10, 10, 12, 12))

    assert ring.shape == (0, 2)
