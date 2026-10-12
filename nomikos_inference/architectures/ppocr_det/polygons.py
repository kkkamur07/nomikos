"""Polygon algebra for the PP-OCRv6 poly output mode.

Everything here runs on plain rings (lists of ``[x, y]``) with OpenCV and
pyclipper only, so the poly carriage in refinement and response never needs
a second geometry engine. The quad decision path in ``refinement.py`` is
untouched; these helpers only combine, cut, simplify and measure the
polygons that the served masks are drawn from.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pyclipper

#: Hard cap on served polygon points; the simplifier raises epsilon until
#: the ring fits.
MAX_POLYGON_POINTS = 64

#: Default serving tolerance for the page-space outline fit, in pixels.
#: 0.5 px keeps the same ink as 0.25 px for fewer points.
DEFAULT_OUTLINE_TOLERANCE_PX = 0.5

#: Default vertical growth factor for served outlines. Each polygon grows
#: across its line (never along it) about its own centre line, so ascenders
#: and descenders stay inside the transcription mask where the pitch allows.
DEFAULT_VERTICAL_GROWTH = 1.35

#: Supersampling for the growth overlap resolution. Contested pixels go to
#: the line whose ungrown polygon is nearer, which needs subpixel masks;
#: 3x keeps the boundary within a third of a pixel of the true Voronoi edge.
#: 2x was tried and dropped: half-integer grown edges round up a whole
#: pixel there, inflating every line by a pixel and leaving tie strips.
_GROWTH_SUPERSAMPLE = 3

#: Tolerance for dropping collinear baseline samples, in pixels.
BASELINE_COLLINEAR_PX = 0.5

#: Fixed scale for served-space pyclipper calls. Clipper works in integers,
#: so served coordinates (which carry fractions) are scaled up before each
#: call and back down after; detector-space unclip calls that mirror Paddle
#: stay untouched so quad mode output never moves.
CLIPPER_SCALE = 1000.0


def _to_clipper(path: object) -> list:
    """Scale one served-space ring up to integer Clipper coordinates."""
    return pyclipper.scale_to_clipper(_ring(path).tolist(), CLIPPER_SCALE)


def _from_clipper(paths: object) -> list[list[list[float]]]:
    """Scale Clipper integer output back down to served coordinates."""
    scaled = pyclipper.scale_from_clipper(paths, CLIPPER_SCALE)
    return [[[float(x), float(y)] for x, y in path] for path in scaled]


def _ring(points: object) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def ring_area(points: object) -> float:
    """Absolute area of a ring, 0 for degenerate input."""
    ring = _ring(points)
    if len(ring) < 3:
        return 0.0
    return abs(float(cv2.contourArea(ring.astype(np.float32))))


def dedup_ring(points: object) -> list[list[float]]:
    """Drop repeated consecutive points and the closing duplicate."""
    ring = _ring(points).tolist()
    out: list[list[float]] = []
    for point in ring:
        pair = [float(point[0]), float(point[1])]
        if not out or pair != out[-1]:
            out.append(pair)
    while len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def clip_ring_to_page(
    points: object, page_width: float | None, page_height: float | None
) -> list[list[float]]:
    """Clamp a ring inside the page; unclamped axes pass through."""
    ring = dedup_ring(points)
    if page_width is None and page_height is None:
        return ring
    clipped = []
    for x, y in ring:
        if page_width is not None:
            x = min(max(x, 0.0), float(page_width))
        if page_height is not None:
            y = min(max(y, 0.0), float(page_height))
        clipped.append([x, y])
    return dedup_ring(clipped)


def union_outlines(outlines: list[object]) -> list[list[float]] | None:
    """pyclipper union of member polygons; one path or ``None``.

    Touching members fuse into a single path, which the caller keeps.
    Separate fragments stay several paths (``None``) so the caller can use
    the convex hull instead.
    """
    clipper = pyclipper.Pyclipper()
    for outline in outlines:
        ring = _ring(outline)
        if len(ring) >= 3 and ring_area(ring) > 0:
            clipper.AddPath(_to_clipper(ring), pyclipper.PT_SUBJECT, True)
    try:
        solution = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    except pyclipper.ClipperException:
        return None
    solution = _from_clipper(solution)
    paths = [_ring(path) for path in solution if len(_ring(path)) >= 3 and ring_area(path) > 0]
    if len(paths) != 1:
        return None
    return dedup_ring(paths[0])


def intersect_outline(outline: object, rect: list[list[float]]) -> list[list[float]] | None:
    """Clip a polygon with a rectangle; largest piece or ``None``."""
    ring = _ring(outline)
    if len(ring) < 3 or ring_area(ring) <= 0:
        return None
    clipper = pyclipper.Pyclipper()
    clipper.AddPath(_to_clipper(ring), pyclipper.PT_SUBJECT, True)
    clipper.AddPath(_to_clipper(rect), pyclipper.PT_CLIP, True)
    try:
        solution = clipper.Execute(
            pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO
        )
    except pyclipper.ClipperException:
        return None
    solution = _from_clipper(solution)
    paths = [_ring(path) for path in solution if len(_ring(path)) >= 3 and ring_area(path) > 0]
    if not paths:
        return None
    best = max(paths, key=ring_area)
    return dedup_ring(best)


def convex_hull_of(points: object) -> list[list[float]]:
    """Convex hull of a point cloud, as a deduped ring."""
    cloud = _ring(points).astype(np.float32)
    if len(cloud) < 3:
        return dedup_ring(cloud)
    hull = cv2.convexHull(cloud, clockwise=False).reshape(-1, 2)
    return dedup_ring(hull)


def simplify_outline(
    outline: object,
    *,
    outline_tolerance_px: float = DEFAULT_OUTLINE_TOLERANCE_PX,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[list[float]] | None:
    """Douglas-Peucker a polygon for serving; ``None`` when unusable.

    Epsilon is ``outline_tolerance_px`` (0.5 px by default, which keeps the
    same ink as 0.25 px for fewer points), raised until the ring fits
    ``MAX_POLYGON_POINTS`` (with a uniform subsample fallback so the cap
    always holds). The result is deduped, clipped to the page, and checked
    with ``SimplifyPolygon`` keeping the largest piece; anything under 4
    points is ``None`` so the caller falls back to the quad.
    """
    ring = _ring(outline).astype(np.float32)
    if len(ring) < 3:
        return None
    epsilon = max(float(outline_tolerance_px), 1e-9)
    approx = ring
    for _ in range(25):
        approx = cv2.approxPolyDP(ring, epsilon, True).reshape(-1, 2)
        if len(approx) <= MAX_POLYGON_POINTS:
            break
        epsilon *= 1.5
    if len(approx) > MAX_POLYGON_POINTS:
        index = np.round(np.linspace(0, len(approx) - 1, MAX_POLYGON_POINTS)).astype(int)
        approx = approx[index]
    cleaned = clip_ring_to_page(approx, page_width, page_height)
    if len(cleaned) < 3:
        return None
    try:
        pieces = _from_clipper(pyclipper.SimplifyPolygon(_to_clipper(cleaned), True))
    except pyclipper.ClipperException:
        return None
    candidates = [
        _ring(piece) for piece in pieces if len(_ring(piece)) >= 3 and ring_area(piece) > 0
    ]
    if not candidates:
        return None
    best = dedup_ring(max(candidates, key=ring_area))
    if len(best) < 4:
        return None
    return best


def _long_axis(
    quad_ring: object,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Principal axis of a quad ring: origin, unit vector, min and max u.

    ``eigh`` returns the axis with an arbitrary sign, so ``anchor`` (the
    quad's own two-point baseline direction) pins it: the axis is flipped
    when it points against the anchor. The anchor is never reduced to an
    ``axis[0] > 0`` rule, which would break vertical lines.
    """
    ring = _ring(quad_ring)
    origin = ring.mean(axis=0)
    centred = ring - origin
    cov = (centred.T @ centred) / max(len(ring), 1)
    _, vecs = np.linalg.eigh(cov)
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0]) if norm <= 0 else axis / norm
    if anchor is not None and float(np.linalg.norm(anchor)) > 0 and float(axis @ anchor) < 0:
        axis = -axis
    proj = (ring - origin) @ axis
    return origin, axis, float(proj.min()), float(proj.max())


def _quad_baseline_anchor(
    quad_ring: object,
    baseline: object | None,
) -> np.ndarray | None:
    """Two-point quad baseline direction used to orient the long axis.

    An explicit ``baseline`` (the quad mode direction, always top left to
    top right) wins; otherwise the quad ring's own top edge stands in.
    """
    if baseline is not None:
        ends = np.asarray(baseline, dtype=np.float64).reshape(-1, 2)
        if len(ends) >= 2:
            return np.asarray(ends[-1] - ends[0], dtype=np.float64)
    corners = _ring(quad_ring)
    if len(corners) >= 2:
        return np.asarray(corners[1] - corners[0], dtype=np.float64)
    return None


def polyline_baseline(
    outline: object,
    quad_ring: object,
    fraction: float,
    *,
    samples: int | None = None,
    baseline: object | None = None,
) -> list[list[float]] | None:
    """Baseline polyline across a polygon along the quad long axis.

    ``samples`` (8 to 16) x positions span the quad axis end to end; at
    each, the polygon's upper and lower boundary give a point at
    ``fraction`` between them. ``baseline`` is the quad's own two-point
    baseline and pins the axis direction, so the polyline runs the same
    way as the quad baseline. Samples past the polygon ends reuse the
    nearest kept point at the axis end instead of dropping out, and
    collinear runs collapse, so a straight line reduces to two points on
    the quad baseline. Returns ``None`` when the polygon has no usable
    crossings.
    """
    ring = _ring(outline)
    if len(ring) < 3 or ring_area(ring) <= 0:
        return None
    anchor = _quad_baseline_anchor(quad_ring, baseline)
    origin, axis, _, _ = _long_axis(quad_ring, anchor)
    normal = np.array([-axis[1], axis[0]])
    if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
        normal = -normal
    rel = ring - origin
    outline_u = rel @ axis
    outline_v = rel @ normal
    quad_u = (_ring(quad_ring) - origin) @ axis
    low, high = float(quad_u.min()), float(quad_u.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    count = samples or min(16, max(8, int(round((high - low) / 16.0))))
    count = min(16, max(8, count))
    edges = [
        (outline_u[i], outline_v[i], outline_u[(i + 1) % len(ring)], outline_v[(i + 1) % len(ring)])
        for i in range(len(ring))
    ]
    slots: list[tuple[float, float | None]] = []
    for step in range(count):
        pos = low + (high - low) * step / (count - 1)
        crossings: list[float] = []
        for u1, v1, u2, v2 in edges:
            if (u1 <= pos < u2) or (u2 <= pos < u1):
                crossings.append(float(v1 + (pos - u1) * (v2 - v1) / (u2 - u1)))
            elif u1 == pos and u2 == pos:
                crossings.extend([float(v1), float(v2)])
        if len(crossings) < 2:
            slots.append((pos, None))
            continue
        top, bottom = min(crossings), max(crossings)
        slots.append((pos, top + float(fraction) * (bottom - top)))
    kept = [(pos, at) for pos, at in slots if at is not None]
    if len(kept) < 2:
        return None
    first_pos, first_at = kept[0]
    last_pos, last_at = kept[-1]
    filled = [
        (pos, first_at if pos < first_pos else (last_at if pos > last_pos else at))
        if at is None
        else (pos, at)
        for pos, at in slots
    ]
    # Step 0 sits at ``low`` and the last step at ``high``, so the ends are
    # always covered after the fill above; interior gaps with no crossings
    # stay out.
    series = [(pos, at) for pos, at in filled if at is not None]
    if len(series) < 2:
        return None
    points = []
    for pos, at in series:
        point = origin + pos * axis + at * normal
        points.append([float(point[0]), float(point[1])])
    return _drop_collinear(points)


def _drop_collinear(points: list[list[float]]) -> list[list[float]]:
    """Greedily drop points within half a pixel of their chord."""
    kept = [points[0]]
    for point in points[1:]:
        kept.append(point)
        while len(kept) >= 3:
            first = np.asarray(kept[-3], dtype=np.float64)
            middle = np.asarray(kept[-2], dtype=np.float64)
            last = np.asarray(kept[-1], dtype=np.float64)
            leg = last - first
            length = float(np.linalg.norm(leg))
            if length <= 0:
                kept.pop(-2)
                continue
            distance = (
                abs(float(leg[0] * (first[1] - middle[1]) - leg[1] * (first[0] - middle[0])))
                / length
            )
            if distance <= BASELINE_COLLINEAR_PX:
                kept.pop(-2)
            else:
                break
    return kept


def _growth_frame(
    quad_ring: object,
    baseline: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre line frame of one text line: origin, along axis, across normal.

    The along axis is the quad long axis pinned to the baseline direction,
    so slanted lines grow perpendicular to their own direction. The sign of
    the axis does not matter for growth, which is symmetric about the origin.
    """
    anchor = _quad_baseline_anchor(quad_ring, baseline)
    origin, axis, _, _ = _long_axis(quad_ring, anchor)
    normal = np.array([-axis[1], axis[0]])
    return origin, axis, normal


def _grow_ring(
    ring: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    normal: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Scale a ring across the line only, about its centre line."""
    rel = ring - origin
    return origin + np.outer(rel @ axis, axis) + np.outer(factor * (rel @ normal), normal)


def _raster_supersampled(
    poly: np.ndarray, x0: float, y0: float, width: int, height: int, ss: int
) -> np.ndarray:
    """Boolean mask of a polygon over a page window at ``ss`` times scale."""
    mask = np.zeros((height * ss, width * ss), dtype=np.uint8)
    cv2.fillPoly(mask, [(((poly - np.array([x0, y0])) * ss).astype(np.int32))], 255)
    return mask > 0


def _contested_losses(
    own_distance: np.ndarray, other_distance: np.ndarray, other: int, index: int
) -> np.ndarray:
    """Contested pixels the neighbour wins outright.

    Strictly nearer pixels, plus exact ties when the neighbour has the
    lower index, so every shared pixel has exactly one owner and no strip
    lands in two transcription crops. A pixel stays in both masks only when
    neither distance is strictly smaller, which for two floats means exactly
    equal, so the equality branch is complete with no epsilon. Distances
    compare exactly as the distance transform computed them.
    """
    return (other_distance < own_distance) | ((other_distance == own_distance) & (other < index))


def grow_outlines_vertical(
    outlines: list[object],
    quad_rings: list[object],
    *,
    factor: float = DEFAULT_VERTICAL_GROWTH,
    baselines: list[object | None] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
    outline_tolerance_px: float = DEFAULT_OUTLINE_TOLERANCE_PX,
) -> list[list[list[float]]]:
    """Grow served outlines across their lines and split neighbour overlaps.

    Each outline is scaled by ``factor`` perpendicular to its quad long axis
    about the quad centre (the along-line extent never moves), which covers
    ascenders and descenders where the pitch allows. Where grown outlines
    overlap, each contested pixel goes to the line whose ungrown outline is
    nearer, with exact ties kept by the lower index, so the boundary follows
    both shapes and no pixel lands in two outlines; where there is no
    neighbour the grown outline stands. Results are integer points clipped
    to the page
    and capped at ``MAX_POLYGON_POINTS``; any line whose grown contour is
    unusable keeps its ungrown outline. ``factor == 1.0`` returns the inputs
    unchanged. Quads and baselines are read only for direction, never moved.
    """
    if baselines is None:
        baselines = [None] * len(outlines)
    if factor == 1.0:
        return [outline for outline in outlines]
    rings = [_ring(outline) for outline in outlines]
    frames = [
        _growth_frame(quad, baseline) for quad, baseline in zip(quad_rings, baselines, strict=True)
    ]
    grown = [
        _grow_ring(ring, origin, axis, normal, float(factor))
        for ring, (origin, axis, normal) in zip(rings, frames, strict=True)
    ]
    boxes = [
        (float(g[:, 0].min()), float(g[:, 0].max()), float(g[:, 1].min()), float(g[:, 1].max()))
        for g in grown
    ]
    box_array = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ss = _GROWTH_SUPERSAMPLE
    approx_ss = max(float(outline_tolerance_px), 1e-9) * ss
    served: list[list[list[float]]] = []
    for index, ring in enumerate(rings):
        fallback = dedup_ring(ring)
        low_x, high_x, low_y, high_y = boxes[index]
        x0 = int(math.floor(low_x - 2))
        x1 = int(math.ceil(high_x + 2))
        y0 = int(math.floor(low_y - 2))
        y1 = int(math.ceil(high_y + 2))
        if page_width is not None:
            x0 = max(0, x0)
            x1 = min(int(math.ceil(float(page_width))), x1)
        if page_height is not None:
            y0 = max(0, y0)
            y1 = min(int(math.ceil(float(page_height))), y1)
        if x1 <= x0 or y1 <= y0:
            served.append(fallback)
            continue
        width, height = x1 - x0, y1 - y0
        grown_mask = _raster_supersampled(grown[index], x0, y0, width, height, ss)
        keep = grown_mask.copy()
        # Contenders first: only lines whose grown masks truly share pixels
        # need distance transforms, so isolated lines skip them entirely.
        hits = (
            (box_array[:, 1] >= x0)
            & (box_array[:, 0] <= x1)
            & (box_array[:, 3] >= y0)
            & (box_array[:, 2] <= y1)
        )
        hits[index] = False
        contenders: dict[int, np.ndarray] = {}
        for other in np.flatnonzero(hits).tolist():
            other_mask = _raster_supersampled(grown[other], x0, y0, width, height, ss)
            if (grown_mask & other_mask).any():
                contenders[int(other)] = other_mask
        if contenders:
            plain_mask = _raster_supersampled(ring, x0, y0, width, height, ss)
            own_distance = cv2.distanceTransform((~plain_mask).astype(np.uint8), cv2.DIST_L2, 3)
            for other, other_mask in contenders.items():
                zone = grown_mask & other_mask
                other_plain = _raster_supersampled(rings[other], x0, y0, width, height, ss)
                other_distance = cv2.distanceTransform(
                    (~other_plain).astype(np.uint8), cv2.DIST_L2, 3
                )
                # Stable ownership: the nearer outline wins, and on an exact
                # tie the lower index keeps the pixel, so no strip lands in
                # two transcription crops.
                keep[zone & _contested_losses(own_distance, other_distance, other, index)] = False
        contours, _ = cv2.findContours(
            keep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            served.append(fallback)
            continue
        biggest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(biggest, approx_ss, True).reshape(-1, 2).astype(np.float64)
        approx = approx / ss + np.array([x0, y0])
        if len(approx) < 4:
            served.append(fallback)
            continue
        rounded = [[float(round(float(x))), float(round(float(y)))] for x, y in approx]
        cleaned = clip_ring_to_page(dedup_ring(rounded), page_width, page_height)
        if len(cleaned) > MAX_POLYGON_POINTS:
            pick = np.round(np.linspace(0, len(cleaned) - 1, MAX_POLYGON_POINTS)).astype(int)
            cleaned = dedup_ring([cleaned[i] for i in pick])
        if len(cleaned) < 4:
            served.append(fallback)
            continue
        try:
            pieces = _from_clipper(pyclipper.SimplifyPolygon(_to_clipper(cleaned), True))
        except pyclipper.ClipperException:
            served.append(fallback)
            continue
        candidates = [piece for piece in pieces if len(_ring(piece)) >= 3 and ring_area(piece) > 0]
        if not candidates:
            served.append(fallback)
            continue
        best = dedup_ring(max(candidates, key=ring_area))
        served.append(best if len(best) >= 4 else fallback)
    return served


__all__ = [
    "BASELINE_COLLINEAR_PX",
    "CLIPPER_SCALE",
    "DEFAULT_OUTLINE_TOLERANCE_PX",
    "DEFAULT_VERTICAL_GROWTH",
    "MAX_POLYGON_POINTS",
    "clip_ring_to_page",
    "convex_hull_of",
    "dedup_ring",
    "grow_outlines_vertical",
    "intersect_outline",
    "polyline_baseline",
    "ring_area",
    "simplify_outline",
    "union_outlines",
]
