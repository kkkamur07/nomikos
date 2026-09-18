#!/usr/bin/env python
"""Render before/after polygons-only side by side for polyfix.

Usage:
  render_polyfix.py --slug SLUG --image PATH --before JSON --after JSON --out JPG
Polygons only, filled translucent, no baselines.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

from PIL import Image, ImageDraw


def panel(image_path: Path, pred_path: Path, title: str) -> Image.Image:
    data = json.loads(pred_path.read_text())
    im = Image.open(image_path).convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    n = 0
    for i, ln in enumerate(data["lines"]):
        pts = [tuple(p) for p in ln.get("polygon") or []]
        if len(pts) < 3:
            continue
        n += 1
        r, g, b = [int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 0.95)]
        draw.polygon(pts, fill=(r, g, b, 70), outline=(r, g, b, 255))
        x0 = min(p[0] for p in pts)
        y = sum(p[1] for p in pts) / len(pts)
        draw.text((max(0, x0 - 18), y - 5), str(i), fill=(0, 0, 0, 255))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    bar = Image.new("RGB", (im.width, 26), "white")
    ImageDraw.Draw(bar).text((6, 6), f"{title}: {n} polygons", fill="black")
    out = Image.new("RGB", (im.width, im.height + 26), "white")
    out.paste(bar, (0, 0))
    out.paste(im, (0, 26))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    left = panel(args.image, args.before, f"{args.slug} before")
    right = panel(args.image, args.after, f"{args.slug} polyfix")
    sheet = Image.new(
        "RGB", (left.width + right.width + 12, max(left.height, right.height)), "white"
    )
    sheet.paste(left, (0, 0))
    sheet.paste(
        right,
        (left.width + 12, 0),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
