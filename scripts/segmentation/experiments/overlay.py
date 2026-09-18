#!/usr/bin/env python
"""Draw a side-by-side overlay of two variants on one page.

    overlay.py --page vat-1r --left v0 --right v3a-cols-oracle --out out/overlays/vat-1r.png

Ground truth is drawn in green, predictions in red with their baseline in blue.
The result is downscaled and JPEG-quality-tuned so it stays small enough to be
committed next to the write-up.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gtlib  # noqa: E402

MAIN = Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR")
OUT = HERE / "out"


def draw(gt: gtlib.GTPage, pred_path: Path, scale: float):
    from PIL import Image, ImageDraw

    image = Image.open(gt.image_path).convert("RGB")
    canvas = ImageDraw.Draw(image)
    for line in gt.lines:
        canvas.polygon([tuple(p) for p in line.polygon], outline=(0, 190, 0))
    pred = json.loads(pred_path.read_text())
    for line in pred["lines"]:
        canvas.polygon([tuple(p) for p in line["polygon"]], outline=(230, 0, 0))
        canvas.line([tuple(p) for p in line["baseline"]], fill=(0, 60, 255), width=2)
    size = (int(image.width * scale), int(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS), len(pred["lines"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--max-kb", type=int, default=290)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(
            "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon"
        ),
    )
    args = parser.parse_args(argv)

    from PIL import Image, ImageDraw

    pages = {
        p.slug: p
        for p in gtlib.load_all(
            gt_dir=args.gt_dir,
            manifest_path=MAIN / "scripts/coptic/manifest.json",
            repo_root=MAIN,
        )
    }
    gt = pages[args.page]
    scale = args.scale
    for _attempt in range(8):
        left, nl = draw(gt, OUT / args.left / f"{gt.slug}.json", scale)
        right, nr = draw(gt, OUT / args.right / f"{gt.slug}.json", scale)
        gap = 12
        band = 26
        sheet = Image.new(
            "RGB", (left.width + right.width + gap, left.height + band), (255, 255, 255)
        )
        sheet.paste(left, (0, band))
        sheet.paste(right, (left.width + gap, band))
        caption = ImageDraw.Draw(sheet)
        caption.text((6, 7), f"{args.left}: {nl} lines", fill=(0, 0, 0))
        caption.text((left.width + gap + 6, 7), f"{args.right}: {nr} lines", fill=(0, 0, 0))
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) <= args.max_kb * 1024:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(data)
            print(f"{args.out} {len(data) // 1024} KB at scale {scale:.2f}")
            return 0
        scale *= 0.82
    raise SystemExit("could not get the overlay under the size limit")


if __name__ == "__main__":
    raise SystemExit(main())
