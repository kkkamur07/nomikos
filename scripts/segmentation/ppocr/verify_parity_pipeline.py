#!/usr/bin/env python3
"""The ``pipeline`` parity stage: hand-built tensor path vs the Paddle pipeline.

Stage library for ``verify_parity.py``: it runs the full
``paddleocr.TextDetection`` pipeline from the raw image to prove the
hand-built tensor path is faithful, and saves the reference fixtures for the
serving adapter.
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
    PAGES,
    STAGE_THREADS,
    as_boxes,
    build_tensor,
    config_float,
    make_paddle_predictor,
    make_post_op,
    max_corner_distance,
    norm_kwargs,
    resolve_image_file,
    run_paddle,
    sha256_file,
)


def _score_pipeline_cell(
    page: str,
    cap: int,
    *,
    image_path: Path,
    bgr: np.ndarray,
    normaliser: dict[str, object],
    post_op,
    predictor,
    pipeline,
) -> tuple[dict[str, object], list[str], dict[str, object] | None]:
    """Hand tensor path vs pipeline for one page and cap.

    Returns the check row, its failures, and the fixture payload (only at
    cap 1920, else None).
    """
    results = pipeline.predict(bgr, batch_size=1, limit_side_len=cap, limit_type="max")
    if len(results) != 1:
        raise RuntimeError(f"expected one result for {page} cap={cap}")
    pipe_boxes = as_boxes(np.asarray(results[0]["dt_polys"], dtype=np.float64))
    pipe_scores = np.asarray(results[0]["dt_scores"], dtype=np.float64)
    tensor, shape = build_tensor(bgr, cap=cap, normaliser=normaliser)
    hand_map = run_paddle(predictor, tensor)
    hand_boxes_raw, _ = post_op([hand_map.astype(np.float32)], [shape])
    hand_boxes = as_boxes(hand_boxes_raw)
    delta = int(len(hand_boxes) - len(pipe_boxes))
    corner = max_corner_distance(hand_boxes, pipe_boxes) if delta == 0 else float("inf")
    row: dict[str, object] = {
        "page": page,
        "cap": cap,
        "pipeline_boxes": int(len(pipe_boxes)),
        "hand_delta": delta,
        "hand_corner_px": corner,
    }
    failures = []
    if delta != 0 or corner > 1.0:
        failures.append(f"{page} cap={cap}: hand tensor path differs from pipeline")
    fixture: dict[str, object] | None = None
    if cap == 1920:
        fixture = {
            "image": image_path.name,
            "image_sha256": sha256_file(image_path),
            "original_height": int(bgr.shape[0]),
            "original_width": int(bgr.shape[1]),
            "tensor_shape": [int(v) for v in tensor.shape],
            "boxes": pipe_boxes.tolist(),
            "scores": [float(v) for v in pipe_scores.tolist()],
        }
    return row, failures, fixture


def _open_pipeline(
    args: argparse.Namespace, post_cfg: dict[str, object], thresh: float, box_thresh: float
):
    """The reference Paddle pipeline with the model's own thresholds."""
    from paddleocr import TextDetection

    return TextDetection(
        model_name="PP-OCRv6_medium_det",
        model_dir=str(args.model_dir),
        device="cpu",
        engine="paddle_static",
        engine_config={
            "run_mode": "mkldnn",
            "cpu_threads": STAGE_THREADS,
            "mkldnn_cache_capacity": 10,
            "enable_cinn": False,
        },
        enable_hpi=False,
        limit_type="max",
        thresh=thresh,
        box_thresh=box_thresh,
        unclip_ratio=config_float(post_cfg, "unclip_ratio", 1.4),
    )


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the ``pipeline`` stage; returns the stage exit code."""
    import cv2

    cv2.setNumThreads(1)

    normaliser, post_cfg = norm_kwargs(args.model_dir)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    fixtures_dir = args.workdir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    predictor = make_paddle_predictor(args.model_dir, STAGE_THREADS)
    pipeline = _open_pipeline(args, post_cfg, thresh, box_thresh)
    pipeline_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for page in PAGES:
        image_path = resolve_image_file(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        for cap in CAPS:
            row, cell_failures, fixture = _score_pipeline_cell(
                page,
                cap,
                image_path=image_path,
                bgr=bgr,
                normaliser=normaliser,
                post_op=post_op,
                predictor=predictor,
                pipeline=pipeline,
            )
            pipeline_rows.append(row)
            failures.extend(cell_failures)
            print(
                f"pipeline {page} cap={cap}: boxes={row['pipeline_boxes']} "
                f"hand_delta={row['hand_delta']} hand_corner={row['hand_corner_px']:.3f}px",
                flush=True,
            )
            if fixture is not None:
                (fixtures_dir / f"{page}.json").write_text(
                    json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
                )
    pipeline.close()
    (args.workdir / "pipeline.json").write_text(
        json.dumps({"pipeline_check": pipeline_rows, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"pipeline done: failures={failures}", flush=True)
    return 0 if not failures else 1
