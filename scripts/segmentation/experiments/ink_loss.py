#!/usr/bin/env python
"""Ink loss of fixed polygons vs original polygons.

For each page: binarise the scan with Otsu, build the union mask of the
ORIGINAL polygons and of the FIXED polygons, and report the share of dark
pixels inside the original union that is still inside the fixed union.

Usage:
  ink_loss.py --orig-dir OUT_ORIG --fixed-dir OUT_FIXED --pages JSON [--scale-check]
Prints per page retention and a corpus total (micro average).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best = 0.0
    thresh = 127
    for t in range(256):
        w_b += float(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best = between
            thresh = t
    return thresh


def union_mask(polys: list, w: int, h: int, label: str = "?") -> np.ndarray:
    """Union of the drawable polygons. A polygon PIL refuses is skipped and reported.

    Skipping is deliberate: one unusable polygon must not abort a corpus run. It is
    reported on stderr (file plus index plus the exception) so the retention numbers
    on stdout are never quietly computed over fewer polygons than the caller thinks.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for idx, pts in enumerate(polys):
        if pts and len(pts) >= 3:
            try:
                draw.polygon([tuple(p) for p in pts], fill=1)
            except Exception as exc:
                print(f"{label}: skipped polygon {idx}: {exc!r}", file=sys.stderr)
                continue
    return np.array(mask, dtype=bool)


def page_retention(image_path: Path, orig_polys: list, fixed_polys: list, size: list) -> dict:
    w, h = int(size[0]), int(size[1])
    with Image.open(image_path) as im:
        gray = np.array(im.convert("L"))
    if gray.shape[1] != w or gray.shape[0] != h:
        sy, sx = gray.shape[0] / h, gray.shape[1] / w
        orig_polys = [[[x * sx, y * sy] for x, y in p] for p in orig_polys]
        fixed_polys = [[[x * sx, y * sy] for x, y in p] for p in fixed_polys]
        w, h = gray.shape[1], gray.shape[0]
    thresh = otsu_threshold(gray)
    dark = gray < thresh
    mo = union_mask(orig_polys, w, h, label=f"{image_path.name} orig")
    mf = union_mask(fixed_polys, w, h, label=f"{image_path.name} fixed")
    dark_orig = int((dark & mo).sum())
    dark_fixed = int((dark & mf).sum())
    dark_total = int(dark.sum())
    ret = dark_fixed / dark_orig if dark_orig else 1.0
    return {
        "threshold": thresh,
        "dark_total": dark_total,
        "dark_in_orig": dark_orig,
        "dark_in_fixed": dark_fixed,
        "retention": round(ret, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-dir", required=True, type=Path)
    parser.add_argument("--fixed-dir", required=True, type=Path)
    parser.add_argument("--pages", required=True, type=Path)
    args = parser.parse_args(argv)
    pages = json.loads(args.pages.read_text())
    tot_o = tot_f = 0
    for entry in pages:
        slug = entry["slug"]
        orig = json.loads((args.orig_dir / f"{slug}.json").read_text())
        fixed = json.loads((args.fixed_dir / f"{slug}.json").read_text())
        if len(orig["lines"]) != len(fixed["lines"]):
            raise SystemExit(
                f"{slug}: line counts differ, orig has {len(orig['lines'])} and fixed has "
                f"{len(fixed['lines'])}. Retention is only meaningful when the fixer kept "
                f"every line, so the run is stopped rather than reported."
            )
        r = page_retention(
            Path(entry["path"]),
            [ln.get("polygon") or [] for ln in orig["lines"]],
            [ln.get("polygon") or [] for ln in fixed["lines"]],
            orig.get("image_size", [0, 0]),
        )
        tot_o += r["dark_in_orig"]
        tot_f += r["dark_in_fixed"]
        print(
            f"{slug}: retention {r['retention']} "
            f"(dark_in_orig {r['dark_in_orig']}, dark_in_fixed {r['dark_in_fixed']}, "
            f"otsu {r['threshold']})"
        )
    print(f"corpus retention: {round(tot_f / tot_o, 4) if tot_o else 1.0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
