"""What a PAGE ``TextLine`` carries: the polygon in ``Coords``, a baseline in ``Baseline``.

Every segmenter trainer we would hand these files to (kraken, eScriptorium,
Transkribus, Orli) reads the two elements as different things. ``Coords`` is the
closed boundary of the line, ``Baseline`` is an open polyline running left to
right under the letter bodies. A closed polygon in ``Baseline`` is not a
degraded baseline, it is a file the trainer cannot learn from, so the two are
resolved here rather than at the point where the XML is written.

Three stored shapes have to be turned back into a baseline, and none of them is a
corruption to repair in the database: each is what a write path legitimately
stores when the client supplies no baseline of its own.

* a hand-drawn line. ``LayoutService.create_part_line`` stores
  ``baseline = {"points": points}``, so the column holds the polygon the box was
  drawn as.
* a line saved through the bulk ``PUT /lines`` route, where
  ``default_baseline_from_polygon`` puts a two-point segment along the polygon's
  *bottom edge*. That is a placeholder this platform invented, not a measurement,
  and it sits about a third of a line-height below where a kraken baseline goes:
  training on it would teach a segmenter to cut descenders off.
* a line whose baseline was never populated at all.

Both get a *derived* baseline, and the line says so in its ``custom`` string so
that a trainer, or a person auditing a training set, can tell a measured
baseline from one this module invented.

``BASELINE_HEIGHT_FRACTION`` is measured, not guessed: see
``docs/inference/training-data-loop.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.annotation.application.line_geometry import baseline_matches_polygon, geometry_points

#: Where a kraken baseline sits inside its own line polygon, as a fraction of the
#: polygon's vertical extent measured downwards from the top.
#:
#: Measured over the 33 recon snapshots of the Coptic ingest (Napoli and Vatican,
#: 2026-09-18): 1010 lines that carry both a kraken-produced baseline and a
#: kraken-produced polygon give mean 0.679, median 0.680, stdev 0.061, p5 0.573,
#: p95 0.766. The two manuscripts' medians are 0.690 and 0.674, a gap of a quarter
#: of the within-manuscript stdev, which is why one constant is defensible here
#: where pixel thresholds are not: no pixel rule transfers across manuscripts, but
#: a ratio of the line's own height does. Full workings in
#: docs/inference/training-data-loop.md.
BASELINE_HEIGHT_FRACTION = 0.68

#: How many x positions a derived baseline is sampled at before collinear points are
#: dropped. Eight is enough to follow the curve of a warped line and few enough that
#: a straight line collapses back to its two endpoints.
_BASELINE_SAMPLES = 8

#: How far in from the polygon's leftmost and rightmost x the sampling starts, as a
#: fraction of its width.
_BASELINE_MARGIN = 0.01

#: A sampled point is dropped when it sits this close to the straight line between
#: its neighbours. Sub-pixel, so a genuinely curved baseline keeps its shape.
_COLLINEAR_TOLERANCE = 0.5


@dataclass(frozen=True)
class LineExportGeometry:
    """The two point lists one ``TextLine`` is written from."""

    coords: list[list[float]]
    baseline: list[list[float]]
    baseline_derived: bool


def resolve_line_geometry(line: object) -> LineExportGeometry:
    """The polygon and the baseline to export for ``line``.

    ``Coords`` prefers the mask, because that is the boundary the editor shows and
    the one a human corrects; ``line.points`` is the fallback, and a degenerate
    geometry of one or two points is squared off into its bounding rectangle so the
    element is never written with fewer vertices than a polygon has.
    """
    mask_points = geometry_points(getattr(line, "mask", None))
    raw_points = _as_points(getattr(line, "points", None))
    coords = _as_polygon(mask_points or raw_points)

    stored = geometry_points(getattr(line, "baseline", None))
    if _is_usable_baseline(stored, coords, mask_points, raw_points):
        ordered = _clip_to_polygon(_left_to_right(stored), coords)
        if len(ordered) >= 2:
            return LineExportGeometry(coords=coords, baseline=ordered, baseline_derived=False)
    return LineExportGeometry(
        coords=coords, baseline=derive_baseline(coords), baseline_derived=True
    )


def _is_usable_baseline(
    stored: list[list[float]],
    coords: list[list[float]],
    mask_points: list[list[float]],
    raw_points: list[list[float]],
) -> bool:
    """True when the stored baseline is a measurement rather than a placeholder.

    The polygon copy is checked against every polygon the row holds, because which
    one it was copied from depends on which write path produced it.
    """
    if len(stored) < 2:
        return False
    if len({(point[0], point[1]) for point in stored}) < 2:
        return False
    if any(
        baseline_matches_polygon(stored, polygon)
        for polygon in (coords, mask_points, raw_points)
        if polygon
    ):
        return False
    return not _lies_on_the_bottom_edge(stored, coords)


def _lies_on_the_bottom_edge(stored: list[list[float]], polygon: list[list[float]]) -> bool:
    """``default_baseline_from_polygon``'s output, recognised by where it sits.

    A real baseline has descender space under it, so every point of it is above the
    polygon's lowest y by more than rounding. A baseline whose every point sits on
    that lowest y, to within half a pixel, is the placeholder and not a
    measurement. The test is deliberately this strict: mistaking a measured
    baseline for a placeholder would throw away the one thing worth training on.
    """
    if not polygon:
        return False
    bottom = max(point[1] for point in polygon)
    return all(point[1] >= bottom - 0.5 for point in stored)


def _clip_to_polygon(baseline: list[list[float]], polygon: list[list[float]]) -> list[list[float]]:
    """The baseline cut back to the polygon's bounding box.

    kraken's BLLA baselines routinely run past the simplified boundary polygon it
    ships them with: on the 2026-09-18 Coptic snapshots, 199 of 1010 kraken lines
    overhang, by a median of 1px but up to 194px. The overhang is not information a
    trainer can use, because the crop it learns from is the polygon, and a
    ``Baseline`` that leaves its ``Coords`` is the first thing a schema check
    flags. Clipping keeps the two elements describing the same piece of page.

    The x ends are cut with the y interpolated at the cut, so a sloping baseline
    keeps its slope; y is clamped, which moved 23 of those 2831 lines by at most
    8 pixels.
    """
    if len(polygon) < 3 or len(baseline) < 2:
        return baseline
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    clipped: list[list[float]] = []
    for previous, current in zip([None, *baseline], baseline, strict=False):
        if previous is not None:
            for edge in (left, right):
                if min(previous[0], current[0]) < edge < max(previous[0], current[0]):
                    span = current[0] - previous[0]
                    y = previous[1] + (current[1] - previous[1]) * (edge - previous[0]) / span
                    clipped.append([edge, y])
        if left <= current[0] <= right:
            clipped.append(current)
    return [[x, min(max(y, top), bottom)] for x, y in _left_to_right(clipped)]


def derive_baseline(polygon: list[list[float]]) -> list[list[float]]:
    """A baseline for a polygon that has none, at ``BASELINE_HEIGHT_FRACTION`` of its height.

    Polygon-following rather than one straight segment: at each sampled ``x`` the
    polygon's own top and bottom edges are found and the point is placed between
    them, so a line that sags or climbs across the page gets a baseline that sags
    or climbs with it, and every point is inside the polygon by construction.
    Points that add nothing are dropped afterwards, which turns a rectangle's
    baseline back into the two-point segment it should be.
    """
    xs = [point[0] for point in polygon] or [0.0]
    ys = [point[1] for point in polygon] or [0.0]
    left, right = min(xs), max(xs)
    if len(polygon) < 3 or right - left <= 0:
        # Nothing to follow. One pixel wide rather than zero, because a Baseline
        # whose x does not increase is a Baseline kraken rejects.
        y = min(ys) + BASELINE_HEIGHT_FRACTION * (max(ys) - min(ys))
        return [[left, y], [right + 1.0 if right == left else right, y]]
    # Inset from the extreme x: at exactly the leftmost vertex the polygon has a
    # single crossing and no vertical extent to place a point inside. One percent,
    # because a kraken baseline spans very nearly its whole polygon and a baseline
    # that stops short of the first letter is a baseline that mistrains the model.
    margin = (right - left) * _BASELINE_MARGIN
    sampled: list[list[float]] = []
    for index in range(_BASELINE_SAMPLES):
        x = left + margin + index * (right - left - 2 * margin) / (_BASELINE_SAMPLES - 1)
        span = _vertical_span(polygon, x)
        if span is None:
            continue
        top, bottom = span
        sampled.append([x, top + BASELINE_HEIGHT_FRACTION * (bottom - top)])
    if len(sampled) < 2:
        y = min(ys) + BASELINE_HEIGHT_FRACTION * (max(ys) - min(ys))
        return [[left, y], [right, y]]
    return _drop_collinear(sampled)


def _vertical_span(polygon: list[list[float]], x: float) -> tuple[float, float] | None:
    """The topmost and bottommost y where the polygon boundary crosses the line ``x``."""
    crossings: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        x0, y0 = start
        x1, y1 = end
        if x0 == x1:
            if x0 == x:
                crossings.extend((y0, y1))
            continue
        if min(x0, x1) <= x <= max(x0, x1):
            crossings.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    if len(crossings) < 2:
        return None
    return min(crossings), max(crossings)


def _drop_collinear(points: list[list[float]]) -> list[list[float]]:
    kept = [points[0]]
    for previous, current, following in zip(points, points[1:], points[2:], strict=False):
        run = following[0] - previous[0]
        if run == 0:
            kept.append(current)
            continue
        expected = previous[1] + (following[1] - previous[1]) * (current[0] - previous[0]) / run
        if abs(current[1] - expected) > _COLLINEAR_TOLERANCE:
            kept.append(current)
    kept.append(points[-1])
    return kept


def _left_to_right(points: list[list[float]]) -> list[list[float]]:
    """The baseline with strictly increasing x, which is the direction PAGE asks for.

    A right-to-left script still has a left-to-right baseline: the reading direction
    is a property of the text, the baseline is a property of the geometry, and
    kraken's parser treats a baseline that doubles back on itself as a defect.

    Returns fewer than two points when the input cannot make a left-to-right
    polyline at all - a vertical baseline, say - which is the caller's signal to
    derive one instead of shipping something a trainer will choke on.
    """
    ordered = sorted(points, key=lambda point: point[0])
    deduped: list[list[float]] = []
    for point in ordered:
        if deduped and point[0] <= deduped[-1][0]:
            continue
        deduped.append(point)
    return deduped


def _as_points(raw: object) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    points: list[list[float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                points.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError):
                continue
    return points


def _as_polygon(points: list[list[float]]) -> list[list[float]]:
    """At least three vertices, by squaring a degenerate geometry off into its bbox.

    A rectangle-kind line stored as two opposite corners, or a line left with a
    single point by a bad import, would otherwise produce a ``Coords`` that no
    schema validator accepts.
    """
    if len(points) >= 3:
        return points
    if not points:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    if right == left:
        right = left + 1.0
    if bottom == top:
        bottom = top + 1.0
    return [[left, top], [right, top], [right, bottom], [left, bottom]]
