#!/usr/bin/env python
"""Run the BLLA segmentation variants over the 33 Coptic ground-truth pages.

Every variant calls the production entry point
``nomikos_inference.architectures.blla.run_blla_segment``. What a variant
changes is only the image bytes handed to it (a crop, a tile, a padded canvas)
and, for V6, the ``heatmap_threshold`` param the production decoder already
takes. Coordinates are mapped back into source-image pixels before they are
written, so every prediction file is directly comparable.

Usage:
    run_variants.py --variants v0 v1-crop ... [--jobs 6] [--pages vat-1r c13]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAIN = Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR")
WORKTREE = HERE.parents[2]
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "nomikos"))

import gtlib  # noqa: E402
import segrun  # noqa: E402

GT_DIR = Path(
    os.environ.get(
        "COPTIC_GT_DIR",
        "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon",
    )
)
OUT = HERE / "out"

# Worker-process globals, filled by _init.
_MODEL: tuple[Path, str] | None = None
_PAGES: dict[str, gtlib.GTPage] = {}


# --------------------------------------------------------------------------
# image preparation
# --------------------------------------------------------------------------
@dataclass
class Piece:
    """One image handed to the segmenter, and where it sits in the page."""

    box: tuple[int, int, int, int]  # left, top, right, bottom in source px
    pad: tuple[int, int] = (0, 0)  # canvas padding, for the sub-1x scales


def _clamp_box(box, width, height):
    left, top, right, bottom = box
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(right))
    bottom = min(height, int(bottom))
    if right - left < 8 or bottom - top < 8:
        return (0, 0, width, height)
    return (left, top, right, bottom)


def oracle_text_box(gt: gtlib.GTPage, margin: float = 0.05) -> tuple[int, int, int, int]:
    xs = [p[0] for ln in gt.lines for p in ln.polygon]
    ys = [p[1] for ln in gt.lines for p in ln.polygon]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return _clamp_box((x0 - mx, y0 - my, x1 + mx, y1 + my), gt.width, gt.height)


def oracle_column_boxes(gt: gtlib.GTPage, margin: float = 0.05) -> list[tuple[int, int, int, int]]:
    boxes = []
    for column in range(gt.ncols):
        lines = [ln for ln in gt.lines if ln.column == column]
        if not lines:
            continue
        xs = [p[0] for ln in lines for p in ln.polygon]
        ys = [p[1] for ln in lines for p in ln.polygon]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        boxes.append(_clamp_box((x0 - mx, y0 - my, x1 + mx, y1 + my), gt.width, gt.height))
    return boxes


def health_column_boxes(
    gt: gtlib.GTPage, v0_lines: list[dict[str, Any]], margin: float = 0.05
) -> list[tuple[int, int, int, int]]:
    """Column bands computed by the platform's own ``column_bands`` from V0."""
    from backend.annotation.application.segment_health import Segment, column_bands

    segments = [
        Segment(id=str(i), points=ln["polygon"], baseline=ln["baseline"])
        for i, ln in enumerate(v0_lines)
    ]
    bands = column_bands(segments, float(gt.width))
    if not bands:
        return [(0, 0, gt.width, gt.height)]
    boxes = []
    for low, high in bands:
        inside = [
            ln
            for ln in v0_lines
            if low - 1 <= min(p[0] for p in ln["polygon"])
            and max(p[0] for p in ln["polygon"]) <= high + 1
        ]
        if inside:
            ys = [p[1] for ln in inside for p in ln["polygon"]]
            y0, y1 = min(ys), max(ys)
        else:
            y0, y1 = 0.0, float(gt.height)
        mx, my = margin * (high - low), margin * (y1 - y0)
        boxes.append(_clamp_box((low - mx, y0 - my, high + mx, y1 + my), gt.width, gt.height))
    return boxes


def horizontal_tiles(
    box: tuple[int, int, int, int], *, multiplier: float, pitch: float
) -> list[tuple[int, int, int, int]]:
    """Split ``box`` into bands so each band is ``multiplier`` times taller,
    relative to the box, once resized to the model's 1800 px.

    Bands overlap by two line pitches so a line cut by a boundary is whole in
    at least one of them.
    """
    left, top, right, bottom = box
    height = bottom - top
    n = max(1, int(round(multiplier)))
    if multiplier <= 1.0:
        return [box]
    band = height / multiplier
    overlap = 2.0 * pitch
    if n == 1:
        n = 2
    starts = []
    if n == 1:
        starts = [top]
    else:
        stride = (height - band) / (n - 1) if n > 1 else 0.0
        starts = [top + i * stride for i in range(n)]
    tiles = []
    for start in starts:
        t = max(top, start - overlap / 2)
        b = min(bottom, start + band + overlap / 2)
        tiles.append((left, int(t), right, int(b)))
    return tiles


def neutral_colour(image) -> tuple[int, int, int]:
    """Median colour of the image's outer frame, used for the padded canvases."""
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    edge = np.concatenate(
        [
            arr[:8].reshape(-1, 3),
            arr[-8:].reshape(-1, 3),
            arr[:, :8].reshape(-1, 3),
            arr[:, -8:].reshape(-1, 3),
        ]
    )
    return tuple(int(v) for v in np.median(edge, axis=0))


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------
def _read_v0(slug: str) -> list[dict[str, Any]]:
    import json

    path = OUT / "v0" / f"{slug}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run v0 first")
    return json.loads(path.read_text())["lines"]


def _run_pieces(
    gt: gtlib.GTPage,
    boxes: list[tuple[int, int, int, int]],
    *,
    threshold: float | None = None,
    dedupe: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    from PIL import Image

    model_path, sha = _MODEL  # type: ignore[misc]
    lines: list[dict[str, Any]] = []
    seconds = 0.0
    with Image.open(gt.image_path) as page:
        page = page.convert("RGB")
        for box in boxes:
            crop = page.crop(box)
            result = segrun.segment_image(
                crop,
                model_path=model_path,
                artifact_sha256=sha,
                heatmap_threshold=threshold,
                offset=(float(box[0]), float(box[1])),
            )
            lines.extend(result.lines)
            seconds += result.seconds
    if dedupe:
        lines = segrun.dedupe_lines(lines, pitch=gt.pitch_for(0))
    return lines, seconds


def _run_canvas(
    gt: gtlib.GTPage, multiplier: float, *, threshold: float | None = None
) -> tuple[list[dict[str, Any]], float]:
    """Paste the page on a larger neutral canvas so the effective scale drops."""
    from PIL import Image

    model_path, sha = _MODEL  # type: ignore[misc]
    factor = 1.0 / multiplier
    with Image.open(gt.image_path) as page:
        page = page.convert("RGB")
        cw, ch = int(gt.width * factor), int(gt.height * factor)
        canvas = Image.new("RGB", (cw, ch), neutral_colour(page))
        ox, oy = (cw - gt.width) // 2, (ch - gt.height) // 2
        canvas.paste(page, (ox, oy))
        result = segrun.segment_image(
            canvas,
            model_path=model_path,
            artifact_sha256=sha,
            heatmap_threshold=threshold,
            offset=(-float(ox), -float(oy)),
        )
    return result.lines, result.seconds


VariantFn = Callable[[gtlib.GTPage], tuple[list[dict[str, Any]], float]]


def build_variants() -> dict[str, VariantFn]:
    v: dict[str, VariantFn] = {}

    v["v0"] = lambda gt: _run_pieces(gt, [(0, 0, gt.width, gt.height)])
    v["v1-crop"] = lambda gt: _run_pieces(gt, [oracle_text_box(gt)])

    for n in (2, 3):
        def tiles(gt, n=n):
            return _run_pieces(
                gt,
                horizontal_tiles(
                    (0, 0, gt.width, gt.height), multiplier=n, pitch=gt.pitch_for(0)
                ),
                dedupe=True,
            )

        v[f"v2-tiles{n}"] = tiles

    v["v3a-cols-oracle"] = lambda gt: _run_pieces(gt, oracle_column_boxes(gt))
    v["v3b-cols-health"] = lambda gt: _run_pieces(
        gt, health_column_boxes(gt, _read_v0(gt.slug))
    )

    def folio(gt):
        """Napoli only: cut the opening into two folios, then column bands."""
        if gt.manuscript != "napoli" or gt.ncols < 4:
            return _run_pieces(gt, health_column_boxes(gt, _read_v0(gt.slug)))
        v0 = _read_v0(gt.slug)
        from backend.annotation.application.segment_health import Segment, column_bands

        segments = [
            Segment(id=str(i), points=ln["polygon"], baseline=ln["baseline"])
            for i, ln in enumerate(v0)
        ]
        bands = column_bands(segments, float(gt.width))
        cut = (bands[1][1] + bands[2][0]) / 2.0 if len(bands) >= 4 else gt.width / 2.0
        boxes = []
        for lo, hi in ((0.0, cut), (cut, float(gt.width))):
            inside = [
                ln
                for ln in v0
                if lo <= sum(p[0] for p in ln["baseline"]) / len(ln["baseline"]) < hi
            ]
            if not inside:
                continue
            xs = [p[0] for ln in inside for p in ln["polygon"]]
            ys = [p[1] for ln in inside for p in ln["polygon"]]
            x0, x1, y0, y1 = max(lo, min(xs)), min(hi, max(xs)), min(ys), max(ys)
            mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
            boxes.append(_clamp_box((x0 - mx, y0 - my, x1 + mx, y1 + my), gt.width, gt.height))
        return _run_pieces(gt, boxes or [(0, 0, gt.width, gt.height)])

    v["v3b-folio"] = folio

    def col_tiles(gt):
        boxes = []
        for box in oracle_column_boxes(gt):
            boxes.extend(horizontal_tiles(box, multiplier=2, pitch=gt.pitch_for(0)))
        return _run_pieces(gt, boxes, dedupe=True)

    v["v4-col-tiles"] = col_tiles

    for m in (0.5, 0.75):
        v[f"v5-s{int(m * 100):03d}"] = lambda gt, m=m: _run_canvas(gt, m)

    def s150(gt):
        return _run_pieces(
            gt,
            horizontal_tiles((0, 0, gt.width, gt.height), multiplier=1.5, pitch=gt.pitch_for(0)),
            dedupe=True,
        )

    v["v5-s150"] = s150

    for t in (0.10, 0.25, 0.32, 0.40):
        v[f"v6-v0-t{int(t * 100):03d}"] = lambda gt, t=t: _run_pieces(
            gt, [(0, 0, gt.width, gt.height)], threshold=t
        )
        v[f"v6-best-t{int(t * 100):03d}"] = lambda gt, t=t: _run_pieces(
            gt, oracle_column_boxes(gt), threshold=t
        )

    return v


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _init() -> None:
    global _MODEL, _PAGES
    _MODEL = segrun.blla_weights()
    pages = gtlib.load_all(
        gt_dir=GT_DIR, manifest_path=MAIN / "scripts/coptic/manifest.json", repo_root=MAIN
    )
    _PAGES = {p.slug: p for p in pages}


def _task(job: tuple[str, str]) -> tuple[str, str, int, float]:
    variant, slug = job
    if _MODEL is None:
        _init()
    gt = _PAGES[slug]
    fn = build_variants()[variant]
    started = time.perf_counter()
    lines, seconds = fn(gt)
    wall = time.perf_counter() - started
    segrun.write_prediction(
        OUT / variant, variant, slug, (gt.width, gt.height), seconds, lines
    )
    return variant, slug, len(lines), wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--pages", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(sorted(build_variants())))
        return 0

    _init()
    slugs = args.pages or sorted(_PAGES)
    known = build_variants()
    jobs = [
        (variant, slug)
        for variant in args.variants
        for slug in slugs
        if variant in known and slug in _PAGES
    ]
    unknown = [v for v in args.variants if v not in known]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    started = time.perf_counter()
    if args.jobs <= 1:
        for job in jobs:
            print(_task(job), flush=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs, initializer=_init) as pool:
            for done, result in enumerate(pool.imap_unordered(_task, jobs), 1):
                print(f"[{done}/{len(jobs)}] {result}", flush=True)
    print(f"total wall {time.perf_counter() - started:.1f}s for {len(jobs)} page-variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
