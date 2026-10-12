#!/usr/bin/env python3
"""Prove the ppocr-det serving adapter matches the Paddle reference on 14 pages.

The ONNX graph itself was already shown to reproduce Paddle exactly
(`docs/inference/ppocrv6-onnx-parity-2026-09-19.md`); what nobody had measured
is the serving adapter (`run_ppocr_det_segment`: PIL decode, numpy/OpenCV
preprocessing, pyclipper DB postprocess) on real weights. This script is that
measurement. For each of the 14 reference fixtures (produced by the full
`paddleocr.TextDetection` pipeline at cap 1920) it reads the page BYTES, calls
the production entry point with refinement switched off (`merge_fragments`,
`resolve_overlaps` and `noise_policy=off`, so the gate measures the detector
and nothing else) in quad mode (`box_type=quad`, pinning the quad path now
that the serving default is polygons), and compares the
returned line quads with the fixture boxes: counts, then one-to-one nearest
matching by quad centre with corner-SET distances (corner order may differ
between the two, so for each corner only its nearest corner in the matched
box counts). It also renders one overlay per page so a human can check the
reading order.

Example::

    PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /Users/krishuagarwal/Desktop/Programming/python/greekOCR/.venv/bin/python \\
        scripts/segmentation/ppocr/verify_adapter.py \\
        --fixtures /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/fixtures \\
        --images /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env/server/images \\
        --grec-images /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/adapter-e2e \\
        --onnx /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/pp-ocrv6-medium-det.onnx \\
        --output-dir /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/adapter-e2e

The exit code is the gate: 0 when every page has identical counts, no
unmatched boxes on either side, and max corner distance at most 0.5 px.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

PAGES = (
    "vat-1r",
    "vat-1v",
    "vat-2r",
    "vat-2v",
    "vat-3r",
    "vat-7v",
    "c10",
    "c11",
    "c12",
    "c13",
    "c14",
    "c21",
    "grec-p1",
    "grec-p4",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
DEFAULT_ARTIFACT_SHA256 = "09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3"
MAX_CORNER_PX = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, required=True, help="Directory of <page>.json fixtures"
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="Directory of the 12 Coptic page images"
    )
    parser.add_argument(
        "--grec-images",
        type=Path,
        required=True,
        help="Directory holding grec-p1 and grec-p4",
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Adapter ONNX artifact")
    parser.add_argument(
        "--artifact-sha256",
        default=DEFAULT_ARTIFACT_SHA256,
        help="Expected SHA-256 of the ONNX artifact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where results.json and <page>.overlay.jpg go",
    )
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_image_bytes(page: str, fixture: dict, args: argparse.Namespace) -> bytes:
    """Read the page image, refusing it unless its SHA-256 matches the fixture."""
    expected_name = fixture["image"]
    expected_sha = fixture["image_sha256"]
    for directory in (args.images, args.grec_images):
        for name in (expected_name, f"{page}{Path(expected_name).suffix}"):
            candidate = directory / name
            if candidate.is_file():
                data = candidate.read_bytes()
                if _sha256(data) != expected_sha:
                    raise ValueError(f"{candidate} SHA-256 does not match fixture {page}.json")
                return data
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{page}{suffix}"
            if candidate.is_file() and candidate.name != expected_name:
                data = candidate.read_bytes()
                if _sha256(data) == expected_sha:
                    return data
    raise FileNotFoundError(f"no image matching fixture {page}.json (sha {expected_sha[:12]}…)")


def _centre(quad: np.ndarray) -> np.ndarray:
    return quad.mean(axis=0)


def _match_by_centre(
    fixture_boxes: list[np.ndarray], adapter_quads: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching: repeatedly take the closest unmatched pair."""
    pairs: list[tuple[int, int]] = []
    unmatched_f = set(range(len(fixture_boxes)))
    unmatched_a = set(range(len(adapter_quads)))
    while unmatched_f and unmatched_a:
        best: tuple[float, int, int] | None = None
        for i in unmatched_f:
            for j in unmatched_a:
                dist = float(np.linalg.norm(_centre(fixture_boxes[i]) - _centre(adapter_quads[j])))
                if best is None or dist < best[0]:
                    best = (dist, i, j)
        if best is None:
            raise RuntimeError("greedy matching found no pair with unmatched boxes left")
        _, i, j = best
        pairs.append((i, j))
        unmatched_f.discard(i)
        unmatched_a.discard(j)
    return pairs


def _corner_set_distances(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Directed nearest-corner distances both ways: 8 values per matched pair."""
    distances = []
    for corner in a:
        distances.append(float(np.linalg.norm(b - corner, axis=1).min()))
    for corner in b:
        distances.append(float(np.linalg.norm(a - corner, axis=1).min()))
    return distances


def _draw_overlays(
    page: str,
    image_bytes: bytes,
    quads: list[np.ndarray],
    baselines: list[np.ndarray],
    fixture_boxes: list[np.ndarray],
    unmatched_fixture: set[int],
    output_dir: Path,
) -> None:
    """Write <page>.overlay.jpg: green quads with order numbers, yellow baselines."""
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    canvas = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(f"could not decode image for page {page}")
    height, width = canvas.shape[:2]
    scale = max(width, height) / 1400.0
    thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.7, 1.1 * scale)
    font_thickness = max(1, int(round(2 * scale)))

    for position, quad in enumerate(quads):
        pts = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 200, 0), thickness, cv2.LINE_AA)
        left_x = float(quad[:, 0].min())
        mid_y = float(quad[:, 1].mean())
        label = str(position + 1)
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_x = int(min(max(left_x - label_w - 6 * scale, 0), width - label_w - 1))
        text_y = int(min(max(mid_y + label_h / 2, label_h), height - 1))
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    for baseline in baselines:
        ends = baseline.astype(np.int32)
        cv2.line(
            canvas,
            (int(ends[0][0]), int(ends[0][1])),
            (int(ends[1][0]), int(ends[1][1])),
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    for i in unmatched_fixture:
        pts = fixture_boxes[i].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), thickness, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output_dir / f"{page}.overlay.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError(f"could not write overlay for page {page}")


def main() -> int:
    from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment

    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[str] = []
    for page in PAGES:
        fixture = json.loads((args.fixtures / f"{page}.json").read_text(encoding="utf-8"))
        image_bytes = _resolve_image_bytes(page, fixture, args)
        response = run_ppocr_det_segment(
            image_bytes,
            model_path=args.onnx,
            artifact_sha256=args.artifact_sha256,
            params={
                "merge_fragments": False,
                "resolve_overlaps": False,
                "noise_policy": "off",
                "box_type": "quad",
            },
        )
        fixture_boxes = [
            np.asarray(box, dtype=np.float64).reshape(4, 2) for box in fixture["boxes"]
        ]
        adapter_quads = [
            np.asarray(line.points, dtype=np.float64).reshape(4, 2) for line in response.lines
        ]
        baselines = [
            np.asarray(line.baseline["points"], dtype=np.float64).reshape(2, 2)
            for line in response.lines
        ]
        pairs = _match_by_centre(fixture_boxes, adapter_quads)
        matched_f = {i for i, _ in pairs}
        matched_a = {j for _, j in pairs}
        unmatched_f = set(range(len(fixture_boxes))) - matched_f
        unmatched_a = set(range(len(adapter_quads))) - matched_a
        distances: list[float] = []
        for i, j in pairs:
            distances.extend(_corner_set_distances(fixture_boxes[i], adapter_quads[j]))
        max_px = max(distances) if distances else 0.0
        mean_px = float(sum(distances) / len(distances)) if distances else 0.0
        rows.append(
            {
                "page": page,
                "fixture_boxes": len(fixture_boxes),
                "adapter_boxes": len(adapter_quads),
                "fixture_unmatched": sorted(unmatched_f),
                "adapter_unmatched": sorted(unmatched_a),
                "max_corner_px": max_px,
                "mean_corner_px": mean_px,
            }
        )
        print(
            f"{page}: fixture={len(fixture_boxes)} adapter={len(adapter_quads)} "
            f"unmatched_f={len(unmatched_f)} unmatched_a={len(unmatched_a)} "
            f"max={max_px:.3f}px mean={mean_px:.3f}px",
            flush=True,
        )
        if len(fixture_boxes) != len(adapter_quads):
            failures.append(
                f"{page}: box count differs "
                f"(fixture={len(fixture_boxes)} adapter={len(adapter_quads)})"
            )
        if unmatched_f or unmatched_a:
            failures.append(
                f"{page}: unmatched boxes f={sorted(unmatched_f)} a={sorted(unmatched_a)}"
            )
        if not math.isfinite(max_px) or max_px > MAX_CORNER_PX:
            failures.append(f"{page}: max corner distance {max_px:.3f}px exceeds {MAX_CORNER_PX}px")
        _draw_overlays(
            page, image_bytes, adapter_quads, baselines, fixture_boxes, unmatched_f, args.output_dir
        )
    report = {
        "onnx": str(args.onnx),
        "artifact_sha256": args.artifact_sha256,
        "params": {
            "merge_fragments": False,
            "resolve_overlaps": False,
            "noise_policy": "off",
            "box_type": "quad",
        },
        "gate": {
            "identical_counts": True,
            "no_unmatched": True,
            "max_corner_px": MAX_CORNER_PX,
        },
        "pages": rows,
        "failures": failures,
        "passed": not failures,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"passed={report['passed']} failures={failures}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
