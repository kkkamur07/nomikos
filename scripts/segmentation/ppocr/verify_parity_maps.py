#!/usr/bin/env python3
"""The ``maps`` parity stage: Paddle vs ONNX probability maps, page by page.

Stage library for ``verify_parity.py``: it builds the input tensor with
PaddleX 3.7.0's own preprocessing classes, runs the SAME tensor through the
Paddle static predictor (mkldnn) and onnxruntime, compares the raw
probability maps, and runs PaddleX's own DB postprocess on both maps to
compare the boxes. All 14 pages, caps 1920 and 960.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.segmentation.ppocr._fixtures import (
    CAPS,
    MAP_PAGES,
    PAGES,
    STAGE_THREADS,
    as_boxes,
    build_tensor,
    config_float,
    make_ort_session,
    make_paddle_predictor,
    make_post_op,
    max_corner_distance,
    norm_kwargs,
    resolve_image_file,
    run_onnx,
    run_paddle,
)


def _score_maps_cell(
    page: str,
    cap: int,
    *,
    bgr: np.ndarray,
    normaliser: dict[str, object],
    post_op,
    thresh: float,
    predictor,
    session,
) -> tuple[dict[str, object], list[str], tuple[np.ndarray, np.ndarray]]:
    """Compare Paddle and ONNX maps for one page and cap.

    Returns the report row, its failures, and the (paddle, onnx) maps for the
    .npy snapshots the driver writes at cap 1920.
    """
    tensor, shape = build_tensor(bgr, cap=cap, normaliser=normaliser)
    paddle_map = run_paddle(predictor, tensor).astype(np.float64)
    onnx_map = run_onnx(session, tensor).astype(np.float64)
    paddle_prob = paddle_map.reshape(paddle_map.shape[-2], paddle_map.shape[-1])
    onnx_prob = onnx_map.reshape(onnx_map.shape[-2], onnx_map.shape[-1])
    abs_diff = np.abs(paddle_prob - onnx_prob)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    bin_diff = int(np.count_nonzero((paddle_prob > thresh) != (onnx_prob > thresh)))

    paddle_boxes_raw, _ = post_op([paddle_map.astype(np.float32)], [shape])
    onnx_boxes_raw, _ = post_op([onnx_map.astype(np.float32)], [shape])
    paddle_boxes = as_boxes(paddle_boxes_raw)
    onnx_boxes = as_boxes(onnx_boxes_raw)
    box_delta = int(len(paddle_boxes) - len(onnx_boxes))
    corner = (
        max_corner_distance(paddle_boxes, onnx_boxes) if box_delta == 0 else float("inf")
    )
    row: dict[str, object] = {
        "page": page,
        "cap": cap,
        "tensor_shape": [int(v) for v in tensor.shape],
        "paddle_boxes": int(len(paddle_boxes)),
        "onnx_boxes": int(len(onnx_boxes)),
        "box_delta": box_delta,
        "max_corner_px": corner,
        "max_abs_prob_diff": max_abs,
        "mean_abs_prob_diff": mean_abs,
        "binarisation_diff_px": bin_diff,
    }
    failures = []
    if box_delta != 0:
        failures.append(f"{page} cap={cap}: box count delta {box_delta}")
    elif corner > 1.0:
        failures.append(f"{page} cap={cap}: corner distance {corner:.3f}px")
    if max_abs > 1e-2:
        failures.append(f"{page} cap={cap}: max abs prob diff {max_abs:.2e}")
    return row, failures, (paddle_map, onnx_map)


def cmd_maps(args: argparse.Namespace) -> int:
    """Run the ``maps`` stage; returns the stage exit code."""
    import cv2

    cv2.setNumThreads(1)
    normaliser, post_cfg = norm_kwargs(args.model_dir)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    args.workdir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.workdir / "maps"
    maps_dir.mkdir(exist_ok=True)

    predictor = make_paddle_predictor(args.model_dir, STAGE_THREADS)
    session = make_ort_session(args.onnx, STAGE_THREADS)
    per_page: list[dict[str, object]] = []
    failures: list[str] = []
    for page in PAGES:
        image_path = resolve_image_file(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        for cap in CAPS:
            row, cell_failures, (paddle_map, onnx_map) = _score_maps_cell(
                page,
                cap,
                bgr=bgr,
                normaliser=normaliser,
                post_op=post_op,
                thresh=thresh,
                predictor=predictor,
                session=session,
            )
            per_page.append(row)
            failures.extend(cell_failures)
            print(
                f"{page} cap={cap} boxes paddle={row['paddle_boxes']} onnx={row['onnx_boxes']} "
                f"delta={row['box_delta']} corner={row['max_corner_px']:.3f}px "
                f"maxabs={row['max_abs_prob_diff']:.2e} "
                f"meanabs={row['mean_abs_prob_diff']:.2e} bindiff={row['binarisation_diff_px']}",
                flush=True,
            )
            if cap == 1920 and page in MAP_PAGES:
                np.save(maps_dir / f"{page}.paddle.npy", paddle_map.astype(np.float32))
                np.save(maps_dir / f"{page}.onnx.npy", onnx_map.astype(np.float32))
    (args.workdir / "maps.json").write_text(
        json.dumps({"per_page": per_page, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"maps done: failures={failures}", flush=True)
    return 0 if not failures else 1
