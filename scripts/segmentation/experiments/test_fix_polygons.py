#!/usr/bin/env python
"""Synthetic tests for fix_polygons.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fix_polygons as fp  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402


def page(lines, w=600, h=800, variant="t"):
    return {"page": "t", "variant": variant, "image_size": [w, h], "seconds": 0.0, "lines": lines}


def overlap_area(a, b) -> float:
    pa = Polygon(a)
    pb = Polygon(b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    return pa.intersection(pb).area


def fix(lines, w=600, h=800, **kw):
    args = {
        "split_above": 0.5,
        "cap_above": 1.0,
        "cap_below": 0.45,
        "pitch_window": 4,
        "do_neighbours": True,
        "do_trim": True,
        "do_cap": True,
        "do_repair": True,
        "do_clamp": True,
    }
    args.update(kw)
    return fp.fix_page(page(lines, w, h), **args)["lines"]


def bl(x0, x1, y, n=8):
    return [[x0 + (x1 - x0) * k / (n - 1), y] for k in range(n)]


def box(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_two_stacked_lines_trimmed():
    lines = [
        {"baseline": bl(100, 500, 200), "polygon": box(100, 500, 150, 260)},
        {"baseline": bl(100, 500, 260), "polygon": box(100, 500, 210, 310)},
    ]
    out = fix(lines)
    assert len(out) == 2
    for ln in out:
        assert len(ln["polygon"]) >= 3
    inter = overlap_area(out[0]["polygon"], out[1]["polygon"])
    smaller = min(Polygon(out[0]["polygon"]).area, Polygon(out[1]["polygon"]).area)
    assert inter / smaller < 0.20


def test_two_columns_do_not_trim():
    # boxes already fit inside the no-neighbour caps, so any shrink is trim, not cap
    lines = [
        {"baseline": bl(50, 250, 200), "polygon": box(50, 250, 180, 210)},
        {"baseline": bl(350, 550, 200), "polygon": box(350, 550, 180, 210)},
    ]
    out = fix(lines)
    # side by side lines share a y but must not cut each other: areas stay intact
    assert Polygon(out[0]["polygon"]).area > 0.9 * Polygon(box(50, 250, 180, 210)).area
    assert Polygon(out[1]["polygon"]).area > 0.9 * Polygon(box(350, 550, 180, 210)).area


def test_skewed_pair():
    upper = [[100 + k * 50, 200 + k * 4] for k in range(8)]
    lower = [[100 + k * 50, 260 + k * 4] for k in range(8)]
    lines = [
        {"baseline": upper, "polygon": box(100, 450, 160, 280)},
        {"baseline": lower, "polygon": box(100, 450, 220, 320)},
    ]
    out = fix(lines)
    inter = overlap_area(out[0]["polygon"], out[1]["polygon"])
    smaller = min(Polygon(out[0]["polygon"]).area, Polygon(out[1]["polygon"]).area)
    assert inter / smaller < 0.20
    # lower line keeps room for ascenders: its top is above the flat midline
    top1 = min(p[1] for p in out[1]["polygon"])
    assert top1 < 260


def test_first_line_huge_box_capped():
    lines = [
        {"baseline": bl(100, 500, 300), "polygon": box(100, 500, 20, 340)},
        {"baseline": bl(100, 500, 360), "polygon": box(100, 500, 320, 400)},
        {"baseline": bl(100, 500, 420), "polygon": box(100, 500, 380, 460)},
    ]
    out = fix(lines)
    top = min(p[1] for p in out[0]["polygon"])
    # pitch is 60, cap above is 1.0 so top must be at or below 240
    assert top >= 300 - 60 - 1
    assert top > 20


def test_broken_polygon_repaired():
    lines = [
        {"baseline": bl(100, 500, 200), "polygon": [[100, 200], [500, 200]]},
        {"baseline": bl(100, 500, 260), "polygon": box(100, 500, 220, 300)},
    ]
    out = fix(lines)
    assert len(out[0]["polygon"]) >= 3
    assert Polygon(out[0]["polygon"]).area > 0
    assert Polygon(out[0]["polygon"]).is_valid


def test_single_line_page_capped_both_sides():
    lines = [{"baseline": bl(100, 500, 400), "polygon": box(100, 500, 50, 750)}]
    out = fix(lines, h=800)
    ys = [p[1] for p in out[0]["polygon"]]
    # fallback pitch 30: top >= 400 - 30, bottom <= 400 + 0.45*30
    assert min(ys) >= 400 - 30 - 1
    assert max(ys) <= 400 + 0.45 * 30 + 1


def test_baselines_untouched_and_variant_suffix():
    lines = [{"baseline": bl(100, 500, 200), "polygon": box(100, 500, 150, 260)}]
    d = fp.fix_page(
        page(lines, variant="orli-base"),
        split_above=0.5,
        cap_above=1.0,
        cap_below=0.45,
        pitch_window=4,
        do_neighbours=True,
        do_trim=True,
        do_cap=True,
        do_repair=True,
        do_clamp=True,
    )
    assert d["lines"][0]["baseline"] == lines[0]["baseline"]
    assert d["variant"] == "orli-base+polyfix"


def test_split_above_moves_boundary():
    def run(split):
        lines = [
            {"baseline": bl(100, 500, 200), "polygon": box(100, 500, 150, 300)},
            {"baseline": bl(100, 500, 260), "polygon": box(100, 500, 150, 320)},
        ]
        return fix(lines, split_above=split)

    small = run(0.35)
    large = run(0.65)
    top_small = min(p[1] for p in small[1]["polygon"])
    top_large = min(p[1] for p in large[1]["polygon"])
    # smaller split gives the lower line more room, so its top is higher up
    assert top_small < top_large
