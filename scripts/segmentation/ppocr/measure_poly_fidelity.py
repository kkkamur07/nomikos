#!/usr/bin/env python3
"""Served-polygon fidelity numbers for one ppocr-det code tree.

Runs the production entry point with default params (``box_type=poly``)
on the given pages and reports per page: lines, mean served points per
line, mean adjacent-line overlap of the served polygons, zone ink kept
inside the transcription-style mask (integer rounded polygon on white,
the mask transcription crops with), share of lines with over 10 percent
of zone ink outside the polygon, and seconds per page.

Run once with this tree on ``PYTHONPATH`` and once with the baseline
tree; pass the baseline JSON as ``--zones-from`` both times so ink zones
(and therefore ink numbers) compare. The ink helpers mirror the throwaway
prototype (compare2.py ``ink_page`` and ``line_recall_cut_kept``); the
polygons always come from the production entry point.

Example::

    PYTHONPATH=<tree> <venv-python> \\
        scripts/segmentation/ppocr/measure_poly_fidelity.py \\
        --model <detector.onnx> --pages <page1.jpg> [<page2.jpg> ...] \\
        --output numbers-new.json --zones-from numbers-base.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as SPoly


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Detector ONNX artifact")
    parser.add_argument(
        "--pages",
        type=Path,
        nargs="+",
        required=True,
        help="Page images in order; the report names each by file stem",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where the JSON numbers go")
    parser.add_argument(
        "--zones-from",
        type=Path,
        default=None,
        help="Baseline JSON whose polygons define the ink zones (default: own polygons)",
    )
    parser.add_argument(
        "--artifact-sha256",
        default=None,
        help="Expected SHA-256 of the model artifact (default: unchecked)",
    )
    return parser.parse_args()


def _raster(poly: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [((poly - np.array([x0, y0])).astype(np.int32))], 255)
    return mask > 0


def _ink_page(grey: np.ndarray, polys_a: list[np.ndarray]) -> tuple[np.ndarray, float]:
    heights = [float(p[:, 1].max() - p[:, 1].min()) for p in polys_a if len(p) >= 3]
    med_h = max(float(np.median(heights)) if heights else 12.0, 4.0)
    size = max(3, int(round(2 * med_h)))
    if size % 2 == 0:
        size += 1
    while size >= min(grey.shape[:2]) and size > 3:
        size -= 2
    ink = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, size, 10
    )
    return ink > 0, med_h


def _kept_cut(
    polys_cfg: list[np.ndarray], polys_a: list[np.ndarray], ink: np.ndarray, med_h: float
) -> tuple[float, float]:
    """Mean transcription-mask kept ink and cut-line share, zones from A."""
    height, width = ink.shape[:2]
    dilate = max(1, int(round(0.5 * med_h)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    kept, cuts = [], []
    for i, poly in enumerate(polys_cfg):
        anchor = polys_a[i] if i < len(polys_a) else poly
        margin = 0.5 * med_h + 2
        x0 = max(0, int(math.floor(anchor[:, 0].min() - margin)))
        x1 = min(width, int(math.ceil(anchor[:, 0].max() + margin)))
        y0 = max(0, int(math.floor(anchor[:, 1].min() - margin)))
        y1 = min(height, int(math.ceil(anchor[:, 1].max() + margin)))
        if x1 <= x0 or y1 <= y0:
            continue
        w, h = x1 - x0, y1 - y0
        base = np.zeros((h, w), np.uint8)
        cv2.fillPoly(base, [((anchor - np.array([x0, y0])).astype(np.int32))], 255)
        zone = cv2.dilate(base, kernel) > 0
        for j, other in enumerate(polys_a):
            if j == i:
                continue
            if other[:, 0].max() < x0 or other[:, 0].min() > x1:
                continue
            if other[:, 1].max() < y0 or other[:, 1].min() > y1:
                continue
            zone[_raster(other, x0, y0, w, h)] = False
        zone_ink = int((ink[y0:y1, x0:x1] & zone).sum())
        if zone_ink == 0:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(
            mask,
            [
                np.array(
                    [[int(round(float(x) - x0)), int(round(float(y) - y0))] for x, y in poly],
                    dtype=np.int32,
                )
            ],
            255,
        )
        ratio = float((ink[y0:y1, x0:x1] & zone & (mask > 0)).sum() / zone_ink)
        kept.append(ratio)
        cuts.append(1.0 if (1 - ratio) > 0.10 else 0.0)
    return (
        float(np.mean(kept)) if kept else 1.0,
        float(np.mean(cuts)) if cuts else 0.0,
    )


def _adjacent_overlap(polys: list[np.ndarray]) -> float:
    if len(polys) < 2:
        return 0.0
    order = sorted(range(len(polys)), key=lambda i: float(polys[i][:, 1].mean()))
    ratios = []
    for a, b in zip(order, order[1:], strict=False):
        try:
            pa, pb = SPoly(polys[a]), SPoly(polys[b])
            if not pa.is_valid:
                pa = pa.buffer(0)
            if not pb.is_valid:
                pb = pb.buffer(0)
            small = min(pa.area, pb.area)
            ratios.append(pa.intersection(pb).area / small if small > 0 else 0.0)
        except Exception:
            ratios.append(0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def main() -> int:
    from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment

    args = _parse_args()
    zone_polys: dict[str, list[np.ndarray]] = {}
    if args.zones_from is not None:
        payload = json.loads(args.zones_from.read_text(encoding="utf-8"))
        for name, entry in payload["pages"].items():
            zone_polys[name] = [np.asarray(p, dtype=np.float64) for p in entry["polygons"]]
    rows: dict[str, dict] = {}
    print("page lines pts/line adj_overlap crop_kept ink_cut sec/page", flush=True)
    # File stems name the pages; underscores read as hyphens so fixture
    # names like segment_page.jpeg match baseline keys like segment-page.
    pages = [(path.stem.replace("_", "-"), path) for path in args.pages]
    # Warm-up untimed so the session load lands outside the per-page times.
    run_ppocr_det_segment(
        pages[0][1].read_bytes(),
        model_path=args.model,
        artifact_sha256=args.artifact_sha256,
        params={"box_type": "poly"},
    )
    for name, path in pages:
        data = path.read_bytes()
        grey = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        started = time.perf_counter()
        response = run_ppocr_det_segment(
            data,
            model_path=args.model,
            artifact_sha256=args.artifact_sha256,
            params={"box_type": "poly"},
        )
        seconds = time.perf_counter() - started
        polys = [
            np.asarray(line.points, dtype=np.float64).reshape(-1, 2) for line in response.lines
        ]
        anchors = zone_polys.get(name, polys)
        ink, med_h = _ink_page(grey, anchors)
        kept, cut = _kept_cut(polys, anchors, ink, med_h)
        rows[name] = {
            "lines": len(polys),
            "pts_per_line": float(np.mean([len(p) for p in polys])) if polys else 0.0,
            "adj_overlap": _adjacent_overlap(polys),
            "crop_kept": kept,
            "ink_cut": cut,
            "sec_per_page": seconds,
            "polygons": [np.asarray(p).tolist() for p in polys],
        }
        row = rows[name]
        print(
            f"{name} {row['lines']} {row['pts_per_line']:.1f} {row['adj_overlap']:.3f} "
            f"{row['crop_kept']:.3f} {row['ink_cut']:.3f} {row['sec_per_page']:.1f}",
            flush=True,
        )
    args.output.write_text(json.dumps({"pages": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
