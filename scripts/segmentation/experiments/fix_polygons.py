#!/usr/bin/env python
"""Fix Orli line polygons with pure geometry.

Reads prediction JSON files (baseline + polygon per line, source pixels)
and writes the same format with fixed polygons. Baselines, line order and
line count are untouched. The output "variant" gets a "+polyfix" suffix.

Steps (each can be disabled for ablation, all on by default):
  1. neighbours: same-column line above and below per line.
  2. trim: clip polygons at the split line between neighbour baselines.
  3. cap: cap first/last lines of a column at pitch fractions.
  4. repair: rebuild degenerate polygons as bands from the baseline.
  5. clamp: clamp to image, round, dedupe, guarantee 3 points.

Usage:
  fix_polygons.py --pred-dir IN --out-dir OUT [--split-above 0.35]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon

MIN_OVERLAP = 0.5
FALLBACK_PITCH = 30.0
BIG = 100000.0


# --------------------------------------------------------------------------
# baseline helpers
# --------------------------------------------------------------------------


def median_y(points: list[list[float]]) -> float:
    return statistics.median(float(p[1]) for p in points)


def extent(baseline: list[list[float]]) -> tuple[float, float]:
    xs = [float(p[0]) for p in baseline]
    return min(xs), max(xs)


def baseline_overlap(a: list[list[float]], b: list[list[float]]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a0, a1 = extent(a)
    b0, b1 = extent(b)
    wa = max(a1 - a0, 1e-6)
    wb = max(b1 - b0, 1e-6)
    ov = min(a1, b1) - max(a0, b0)
    if ov <= 0:
        return 0.0
    return ov / min(wa, wb)


def sorted_by_x(baseline: list[list[float]]) -> list[tuple[float, float]]:
    pts = sorted(((float(p[0]), float(p[1])) for p in baseline), key=lambda t: t[0])
    out: list[tuple[float, float]] = []
    for x, y in pts:
        if out and out[-1][0] == x:
            out[-1] = (x, (out[-1][1] + y) / 2.0)
        else:
            out.append((x, y))
    return out


def interp_y(sorted_bl: list[tuple[float, float]], x: float) -> float:
    if not sorted_bl:
        return 0.0
    if x <= sorted_bl[0][0]:
        return sorted_bl[0][1]
    if x >= sorted_bl[-1][0]:
        return sorted_bl[-1][1]
    for (x0, y0), (x1, y1) in zip(sorted_bl, sorted_bl[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2.0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return sorted_bl[-1][1]


def to_polygon(points: list | None) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        ring = Polygon(points)
    except Exception:
        return None
    if not ring.is_valid:
        try:
            fixed = ring.buffer(0)
        except Exception:
            return None
        if fixed.is_empty or getattr(fixed, "area", 0) <= 0:
            return None
        if isinstance(fixed, Polygon):
            ring = fixed
        else:
            polys = [g for g in getattr(fixed, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return None
            ring = max(polys, key=lambda g: g.area)
    if ring.area <= 0:
        return None
    return ring


def is_degenerate(points: list | None) -> bool:
    if not points or len(points) < 3:
        return True
    try:
        ring = Polygon(points)
    except Exception:
        return True
    if not ring.is_valid:
        return True
    return ring.area <= 0


# --------------------------------------------------------------------------
# neighbours and pitch
# --------------------------------------------------------------------------


def find_neighbours(
    baselines: list[list[list[float]]],
) -> tuple[list[int | None], list[int | None]]:
    n = len(baselines)
    meds = [median_y(bl) if len(bl) >= 1 else float("inf") for bl in baselines]
    above: list[int | None] = [None] * n
    below: list[int | None] = [None] * n
    for i in range(n):
        if len(baselines[i]) < 2:
            continue
        best_up = None
        best_up_dy = float("inf")
        best_dn = None
        best_dn_dy = float("inf")
        for j in range(n):
            if j == i or len(baselines[j]) < 2:
                continue
            if baseline_overlap(baselines[i], baselines[j]) < MIN_OVERLAP:
                continue
            dy = meds[i] - meds[j]
            if dy > 0 and dy < best_up_dy:
                best_up_dy = dy
                best_up = j
            elif dy < 0 and -dy < best_dn_dy:
                best_dn_dy = -dy
                best_dn = j
        above[i] = best_up
        below[i] = best_dn
    return above, below


def page_median_pitch(baselines: list[list[list[float]]]) -> float:
    meds = sorted(median_y(bl) for bl in baselines if len(bl) >= 2)
    gaps = [b - a for a, b in zip(meds, meds[1:], strict=False) if b - a > 0]
    if not gaps:
        return FALLBACK_PITCH
    return statistics.median(gaps)


def local_pitch(
    i: int,
    baselines: list[list[list[float]]],
    page_pitch: float,
    window: int = 4,
) -> float:
    meds = [median_y(bl) if len(bl) >= 2 else None for bl in baselines]
    if meds[i] is None:
        return page_pitch if page_pitch > 0 else FALLBACK_PITCH
    mates = [i]
    for j, m in enumerate(meds):
        if j != i and m is not None and baseline_overlap(baselines[i], baselines[j]) >= MIN_OVERLAP:
            mates.append(j)
    if len(mates) < 2:
        return page_pitch if page_pitch > 0 else FALLBACK_PITCH
    mates_sorted = sorted(mates, key=lambda k: meds[k])  # type: ignore[index]
    pos = mates_sorted.index(i)
    ordered = [meds[k] for k in mates_sorted]  # type: ignore[misc]
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b - a > 0]  # type: ignore[operator]
    if not gaps:
        return page_pitch if page_pitch > 0 else FALLBACK_PITCH
    lo = max(0, pos - window // 2)
    hi = min(len(gaps), pos + window // 2)
    near = gaps[lo:hi]
    if not near:
        near = gaps
    val = statistics.median(near)
    return val if val > 0 else (page_pitch if page_pitch > 0 else FALLBACK_PITCH)


# --------------------------------------------------------------------------
# clipping
# --------------------------------------------------------------------------


def keeper_below(xs: list[float], ys: list[float]) -> Polygon | None:
    pts = [(x, y) for x, y in zip(xs, ys, strict=False)]
    big = max(ys) + BIG
    pts = pts + [(xs[-1], big), (xs[0], big)]
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if not isinstance(poly, Polygon):
            polys = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)
    return poly


def keeper_above(xs: list[float], ys: list[float]) -> Polygon | None:
    pts = [(x, y) for x, y in zip(xs, ys, strict=False)]
    small = min(ys) - BIG
    pts = pts + [(xs[-1], small), (xs[0], small)]
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if not isinstance(poly, Polygon):
            polys = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)
    return poly


def sample_xs(
    poly: Polygon, self_bl: list[tuple[float, float]], other_bl: list[tuple[float, float]]
) -> list[float]:
    x0, _, x1, _ = poly.bounds
    xs = {x0, x1}
    for x, _ in self_bl:
        if x0 - 2 <= x <= x1 + 2:
            xs.add(float(x))
    for x, _ in other_bl:
        if x0 - 2 <= x <= x1 + 2:
            xs.add(float(x))
    try:
        coords = list(poly.exterior.coords)
    except Exception:
        coords = []
    for x, _ in coords:
        xs.add(float(x))
    out = sorted(xs)
    if len(out) < 2:
        out = [x0 - 1.0, x1 + 1.0]
    return out


def pick_piece(geom, ref: tuple[float, float]) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    geoms = list(getattr(geom, "geoms", []))
    polys: list[Polygon] = []
    for g in geoms:
        if isinstance(g, Polygon) and g.area > 0:
            polys.append(g)
        elif isinstance(g, MultiPolygon):
            polys.extend([p for p in g.geoms if p.area > 0])
    if not polys:
        return geom if isinstance(geom, Polygon) else None
    from shapely.geometry import Point

    pt = Point(ref)
    contained = [p for p in polys if _contains(p, ref)]
    if contained:
        return max(contained, key=lambda p: p.area)
    try:
        return min(polys, key=lambda p: p.distance(pt))
    except Exception:
        return max(polys, key=lambda p: p.area)


def _contains(poly: Polygon, ref: tuple[float, float]) -> bool:
    from shapely.geometry import Point

    try:
        return poly.contains(Point(ref))
    except Exception:
        return False


def clip_poly(poly: Polygon, keeper: Polygon | None, ref: tuple[float, float]) -> Polygon:
    if keeper is None:
        return poly
    try:
        cut = poly.intersection(keeper)
    except Exception:
        return poly
    if cut.is_empty:
        return poly
    piece = pick_piece(cut, ref)
    if piece is None or piece.area <= 0:
        return poly
    if not piece.is_valid:
        try:
            piece = piece.buffer(0)
        except Exception:
            return poly
        if piece.is_empty:
            return poly
        piece = pick_piece(piece, ref)
        if piece is None:
            return poly
    return piece


def ref_point(baseline: list[list[float]]) -> tuple[float, float]:
    if not baseline:
        return (0.0, 0.0)
    mid = baseline[len(baseline) // 2]
    return (float(mid[0]), float(mid[1]))


def trim_top(
    poly: Polygon,
    self_sorted: list[tuple[float, float]],
    up_sorted: list[tuple[float, float]],
    split: float,
    ref: tuple[float, float],
) -> Polygon:
    xs = sample_xs(poly, self_sorted, up_sorted)
    ys = [
        interp_y(up_sorted, x) + split * (interp_y(self_sorted, x) - interp_y(up_sorted, x))
        for x in xs
    ]
    return clip_poly(poly, keeper_below(xs, ys), ref)


def trim_bottom(
    poly: Polygon,
    self_sorted: list[tuple[float, float]],
    low_sorted: list[tuple[float, float]],
    split: float,
    ref: tuple[float, float],
) -> Polygon:
    xs = sample_xs(poly, self_sorted, low_sorted)
    ys = [
        interp_y(self_sorted, x) + split * (interp_y(low_sorted, x) - interp_y(self_sorted, x))
        for x in xs
    ]
    return clip_poly(poly, keeper_above(xs, ys), ref)


def cap_top(
    poly: Polygon,
    self_sorted: list[tuple[float, float]],
    pitch: float,
    cap_above: float,
    ref: tuple[float, float],
) -> Polygon:
    xs = sample_xs(poly, self_sorted, self_sorted)
    ys = [interp_y(self_sorted, x) - cap_above * pitch for x in xs]
    return clip_poly(poly, keeper_below(xs, ys), ref)


def cap_bottom(
    poly: Polygon,
    self_sorted: list[tuple[float, float]],
    pitch: float,
    cap_below: float,
    ref: tuple[float, float],
) -> Polygon:
    xs = sample_xs(poly, self_sorted, self_sorted)
    ys = [interp_y(self_sorted, x) + cap_below * pitch for x in xs]
    return clip_poly(poly, keeper_above(xs, ys), ref)


def build_band(
    baseline: list[list[float]],
    pitch: float,
    cap_above: float,
    cap_below: float,
) -> Polygon | None:
    if len(baseline) < 1:
        return None
    if len(baseline) < 2:
        x, y = float(baseline[0][0]), float(baseline[0][1])
        half = max(pitch / 2.0, 5.0)
        baseline = [[x - half, y], [x + half, y]]
    sbl = sorted_by_x(baseline)
    top = [(x, y - cap_above * pitch) for x, y in sbl]
    bot = [(x, y + cap_below * pitch) for x, y in reversed(sbl)]
    try:
        ring = Polygon(top + bot)
    except Exception:
        return None
    if not ring.is_valid:
        ring = ring.buffer(0)
        if ring.is_empty:
            return None
        if not isinstance(ring, Polygon):
            polys = [g for g in getattr(ring, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return None
            ring = max(polys, key=lambda g: g.area)
    return ring if ring.area > 0 else None


# --------------------------------------------------------------------------
# per page fix
# --------------------------------------------------------------------------


def fix_page(
    data: dict,
    *,
    split_above: float,
    cap_above: float,
    cap_below: float,
    pitch_window: int,
    do_neighbours: bool,
    do_trim: bool,
    do_cap: bool,
    do_repair: bool,
    do_clamp: bool,
) -> dict:
    lines = data.get("lines", [])
    baselines = [ln.get("baseline") or [] for ln in lines]
    poly_pts = [ln.get("polygon") or [] for ln in lines]
    w, h = data.get("image_size", [0, 0])
    try:
        w, h = int(w), int(h)
    except Exception:
        w, h = 0, 0

    n = len(lines)
    if do_neighbours:
        above, below = find_neighbours(baselines)
    else:
        above, below = [None] * n, [None] * n

    page_pitch = page_median_pitch(baselines)
    pitches = [local_pitch(i, baselines, page_pitch, pitch_window) for i in range(n)]
    sorted_bls = [sorted_by_x(bl) if len(bl) >= 2 else [] for bl in baselines]
    refs = [ref_point(bl) if len(bl) >= 1 else (0.0, 0.0) for bl in baselines]

    new_lines = []
    for i in range(n):
        bl = baselines[i]
        pts = poly_pts[i]
        pitch = pitches[i]
        if pitch <= 0:
            pitch = page_pitch if page_pitch > 0 else FALLBACK_PITCH
        ref = refs[i]

        deg = is_degenerate(pts)
        if deg and do_repair and len(bl) >= 1:
            band = build_band(bl, pitch, cap_above, cap_below)
            poly = band if band is not None else to_polygon(pts)
            if poly is None and len(bl) >= 2:
                poly = build_band(bl, FALLBACK_PITCH, cap_above, cap_below)
        else:
            poly = to_polygon(pts)

        if poly is None:
            if len(bl) >= 1 and do_repair:
                poly = build_band(bl, pitch, cap_above, cap_below)
            if poly is None:
                new_lines.append({"baseline": bl, "polygon": _fallback_triangle(ref, w, h)})
                continue

        sbl = sorted_bls[i]
        if len(bl) >= 2 and sbl:
            if do_trim:
                up = above[i]
                if up is not None and len(baselines[up]) >= 2 and sorted_bls[up]:
                    poly = trim_top(poly, sbl, sorted_bls[up], split_above, ref)
                low = below[i]
                if low is not None and len(baselines[low]) >= 2 and sorted_bls[low]:
                    poly = trim_bottom(poly, sbl, sorted_bls[low], split_above, ref)
            if do_cap:
                if above[i] is None:
                    poly = cap_top(poly, sbl, pitch, cap_above, ref)
                if below[i] is None:
                    poly = cap_bottom(poly, sbl, pitch, cap_below, ref)

        coords = list(poly.exterior.coords)
        if do_clamp and w > 0 and h > 0:
            coords = _clamp_round_dedupe(coords, w, h)
        else:
            coords = _round_dedupe(coords)
        if _distinct_count(coords) < 3:
            coords = _fallback_triangle(ref, w, h)
        new_lines.append({"baseline": bl, "polygon": coords})

    variant = data.get("variant", "")
    if variant and not variant.endswith("+polyfix"):
        variant = variant + "+polyfix"
    elif not variant:
        variant = "+polyfix"
    out = dict(data)
    out["lines"] = new_lines
    out["variant"] = variant
    return out


def _round_dedupe(coords: list[tuple[float, float]]) -> list[list[int]]:
    rounded = [(int(round(x)), int(round(y))) for x, y in coords]
    if rounded and rounded[0] == rounded[-1]:
        rounded = rounded[:-1]
    dedup: list[tuple[int, int]] = []
    for p in rounded:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return [[x, y] for x, y in dedup]


def _clamp_round_dedupe(coords: list[tuple[float, float]], w: int, h: int) -> list[list[int]]:
    rounded = [(int(round(x)), int(round(y))) for x, y in coords]
    if rounded and rounded[0] == rounded[-1]:
        rounded = rounded[:-1]
    clamped = [(min(max(x, 0), w - 1), min(max(y, 0), h - 1)) for x, y in rounded]
    dedup: list[tuple[int, int]] = []
    for p in clamped:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup = dedup[:-1]
    return [[x, y] for x, y in dedup]


def _distinct_count(coords: list[list[int]]) -> int:
    return len({(p[0], p[1]) for p in coords})


def _fallback_triangle(ref: tuple[float, float], w: int, h: int) -> list[list[int]]:
    cx, cy = int(round(ref[0])), int(round(ref[1]))
    if w > 0:
        cx = min(max(cx, 1), w - 2)
    if h > 0:
        cy = min(max(cy, 1), h - 2)
    tri = [[cx - 2, cy - 2], [cx + 2, cy - 2], [cx, cy + 2]]
    if w > 0 and h > 0:
        tri = [[min(max(x, 0), w - 1), min(max(y, 0), h - 1)] for x, y in tri]
    if len({(x, y) for x, y in tri}) < 3:
        tri = [[cx - 2, cy], [cx + 2, cy], [cx, cy + 2]]
    return tri


def run_dir(
    pred_dir: Path,
    out_dir: Path,
    *,
    split_above: float,
    cap_above: float,
    cap_below: float,
    pitch_window: int,
    do_neighbours: bool,
    do_trim: bool,
    do_cap: bool,
    do_repair: bool,
    do_clamp: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {pred_dir}")
    for path in files:
        data = json.loads(path.read_text())
        fixed = fix_page(
            data,
            split_above=split_above,
            cap_above=cap_above,
            cap_below=cap_below,
            pitch_window=pitch_window,
            do_neighbours=do_neighbours,
            do_trim=do_trim,
            do_cap=do_cap,
            do_repair=do_repair,
            do_clamp=do_clamp,
        )
        (out_dir / path.name).write_text(json.dumps(fixed))
    print(f"wrote {len(files)} files to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    # Default 0.35, not the midpoint. Text sits above its baseline, so the boundary
    # between two lines belongs nearer the upper baseline. A reviewer's per-line ink
    # check on 872 corrected Coptic lines gave ink recall 0.939 at 0.35 against 0.851
    # at 0.50, with foreign ink 2.5% against 2.9%.
    parser.add_argument(
        "--split-above",
        type=float,
        default=0.35,
        help=(
            "where the boundary between two stacked lines sits, as a fraction of the "
            "gap from the upper baseline to this line's baseline (default: 0.35, which "
            "leaves room for ascenders; 0.5 is the midpoint and clips them)"
        ),
    )
    parser.add_argument("--cap-above", "--cap-pitch", dest="cap_above", type=float, default=1.0)
    parser.add_argument("--cap-below", dest="cap_below", type=float, default=0.45)
    parser.add_argument("--pitch-window", type=int, default=4)
    parser.add_argument("--no-neighbours", action="store_true")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-clamp", action="store_true")
    args = parser.parse_args(argv)
    return run_dir(
        args.pred_dir,
        args.out_dir,
        split_above=args.split_above,
        cap_above=args.cap_above,
        cap_below=args.cap_below,
        pitch_window=args.pitch_window,
        do_neighbours=not args.no_neighbours,
        do_trim=not args.no_trim,
        do_cap=not args.no_cap,
        do_repair=not args.no_repair,
        do_clamp=not args.no_clamp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
