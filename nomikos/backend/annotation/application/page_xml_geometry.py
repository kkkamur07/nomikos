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

Everything leaves here as **integers**. PAGE's ``PointsType`` is a list of
integer pairs, and kraken's ``parse_page_coords`` calls ``int()`` on each
component, so a single ``330.52`` makes the whole file unreadable. Rounding is
done here rather than in the formatter because rounding is not a display choice:
it can collapse two baseline vertices onto one x, or a polygon onto a line, and
the invariants those would break are this module's to keep.
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
    """The two point lists one ``TextLine`` is written from, in integer pixels."""

    coords: list[list[int]]
    baseline: list[list[int]]
    baseline_derived: bool


def resolve_line_geometry(
    line: object, *, bounds: tuple[int, int] | None = None
) -> LineExportGeometry:
    """The polygon and the baseline to export for ``line``, rounded to whole pixels.

    ``Coords`` prefers the mask, because that is the boundary the editor shows and
    the one a human corrects; ``line.points`` is the fallback, and a degenerate
    geometry of one or two points is squared off into its bounding rectangle so the
    element is never written with fewer vertices than a polygon has.

    ``bounds`` is the page's ``(width, height)`` when it is known. Coordinates are
    clamped into it: a vertex a few pixels off the edge of the scan is a vertex no
    trainer can crop to, and the schema has nothing to say about it.
    """
    mask_points = geometry_points(getattr(line, "mask", None))
    raw_points = _as_points(getattr(line, "points", None))
    coords = quantize_polygon(_as_polygon(mask_points or raw_points), bounds=bounds)
    # The same polygon the file will carry, in the float form the geometry helpers
    # work in. Every test below is made against this one rather than the unrounded
    # input, so an invariant that holds here holds in the exported file.
    polygon = [[float(x), float(y)] for x, y in coords]

    stored = geometry_points(getattr(line, "baseline", None))
    if _is_usable_baseline(stored, polygon, mask_points, raw_points):
        # Clipped against the *quantized* polygon, because that is the one the
        # containment invariant has to hold against once both are written. The
        # length is checked before quantizing as well as after: a baseline clipped
        # away to nothing must fall through to a derived one, not be replaced by
        # the two-point fallback, which would label a guess as a measurement.
        clipped = _clip_to_polygon(_left_to_right(stored), polygon)
        if len(clipped) >= 2:
            ordered = quantize_baseline(clipped, coords)
            if len(ordered) >= 2:
                return LineExportGeometry(coords=coords, baseline=ordered, baseline_derived=False)
    return LineExportGeometry(
        coords=coords,
        baseline=quantize_baseline(derive_baseline(polygon), coords),
        baseline_derived=True,
    )


def quantize_polygon(
    points: list[list[float]], *, bounds: tuple[int, int] | None = None
) -> list[list[int]]:
    """Whole-pixel polygon vertices, still at least three of them.

    Rounding can fold neighbouring vertices onto each other, and a simplified
    kraken boundary that is only a pixel or two tall can round flat. Duplicates go
    (kraken's own parser drops consecutive ones anyway), and anything left with
    fewer than three distinct vertices is squared off into its bounding rectangle,
    one pixel wide or tall at minimum.
    """
    rounded = _dedupe([_clamp([round(x), round(y)], bounds) for x, y in points])
    if len(rounded) >= 3:
        return rounded
    return _clamped_rectangle(rounded, bounds)


def quantize_baseline(baseline: list[list[float]], polygon: list[list[int]]) -> list[list[int]]:
    """Whole-pixel baseline vertices that still read left to right, inside ``polygon``.

    Three things can go wrong at once when a float baseline is rounded: two
    vertices can land on the same x, which makes the polyline double back; a
    vertex can round a half pixel outside the polygon it was measured inside; and
    a short baseline can collapse to a single point. All three are fixed here
    rather than discovered by a trainer.
    """
    if not polygon:
        return [[round(x), round(y)] for x, y in baseline]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    kept: list[list[int]] = []
    for x, y in baseline:
        point = [min(max(round(x), left), right), min(max(round(y), top), bottom)]
        if kept and point[0] <= kept[-1][0]:
            # Two vertices on one x. The first wins: dropping the later one keeps
            # the run strictly increasing without moving anything off the line.
            continue
        kept.append(point)
    if len(kept) >= 2:
        return kept
    y = min(max(round(top + BASELINE_HEIGHT_FRACTION * (bottom - top)), top), bottom)
    return [[left, y], [right, y]] if right > left else [[left, y], [left + 1, y]]


def _dedupe(points: list[list[int]]) -> list[list[int]]:
    unique: list[list[int]] = []
    for point in points:
        if point not in unique:
            unique.append(point)
    return unique


def _clamp(point: list[int], bounds: tuple[int, int] | None) -> list[int]:
    x, y = point
    if bounds is None:
        return [x, y]
    width, height = bounds
    return [min(max(x, 0), max(width - 1, 0)), min(max(y, 0), max(height - 1, 0))]


def _clamped_rectangle(points: list[list[int]], bounds: tuple[int, int] | None) -> list[list[int]]:
    if not points:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, top = min(xs), min(ys)
    right, bottom = max(max(xs), left + 1), max(max(ys), top + 1)
    corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
    # Clamping could re-collapse the rectangle against the page edge, so a shape
    # pinned to the boundary is nudged inwards instead of outwards.
    if bounds is not None:
        width, height = bounds
        if right > width - 1:
            left, right = max(width - 2, 0), max(width - 1, 1)
        if bottom > height - 1:
            top, bottom = max(height - 2, 0), max(height - 1, 1)
        corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return corners


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
