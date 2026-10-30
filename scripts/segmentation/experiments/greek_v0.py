#!/usr/bin/env python
"""Run kraken production V0 over the five Grec 1360 spreads.

    greek_v0.py --out-dir out/greek-v0 [--gt-dir <recon>]

This is the same code path as ``out/v0-same10``: ``run_variants.py`` defines V0
as ``_run_pieces(gt, [(0, 0, gt.width, gt.height)])``, one whole-page call to
``segrun.segment_image``, which calls the production
``nomikos_inference.architectures.blla.run_blla_segment`` on PNG bytes with the
pinned blla-segment weights. Nothing is cropped, tiled or re-thresholded here
either. It is a separate entry point only because ``run_variants.py`` iterates
the Coptic manifest and these pages are not in it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WORKTREE = HERE.parents[2]
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "nomikos"))

import greek_gt  # noqa: E402
import gtlib  # noqa: E402
import segrun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path, default=gtlib.env_path(gtlib.GREEK_GT_ENV))
    parser.add_argument("--converted-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    gt_dir = gtlib.require_gt_dir(args.gt_dir, gtlib.GREEK_GT_ENV)

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    model_path, sha = segrun.blla_weights()
    pages = greek_gt.loader(gt_dir=gt_dir, manual_only=False, converted_dir=args.converted_dir)(
        gt_dir=gt_dir
    )
    for gt in pages:
        out = args.out_dir / f"{gt.slug}.json"
        if out.exists():
            print(f"[skip] {gt.slug}", flush=True)
            continue
        started = time.perf_counter()
        with Image.open(gt.image_path) as page:
            result = segrun.segment_image(
                page.convert("RGB"), model_path=model_path, artifact_sha256=sha
            )
        segrun.write_prediction(
            args.out_dir, "v0", gt.slug, (gt.width, gt.height), result.seconds, result.lines
        )
        print(
            f"[ok] {gt.slug} {gt.width}x{gt.height} lines={len(result.lines)} "
            f"{result.seconds:.1f}s (wall {time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    print("GREEK_V0_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
