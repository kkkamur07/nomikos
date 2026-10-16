#!/usr/bin/env python3
"""Prove the medium-det ONNX reproduces Paddle static inference on 14 pages.

ADR 0006 serves models through onnxruntime on CPU, so this script is the
measurement the export stands on. It runs as one stage at a time, each stage
writing its own JSON under the workdir so stages can be run and re-run alone:

* ``maps``: build the input tensor with PaddleX 3.7.0's own preprocessing
  classes, run the SAME tensor through the Paddle static predictor (mkldnn)
  and onnxruntime, compare the raw probability maps, and run PaddleX's own DB
  postprocess on both maps to compare the boxes. All 14 pages, caps 1920 and
  960.
* ``shapes``: five extra input sizes on synthetic tensors, so a frozen axis
  fails loudly instead of passing silently.
* ``pipeline``: run the full ``paddleocr.TextDetection`` pipeline from the raw
  image to prove the hand-built tensor path is faithful, and save the
  reference fixtures for the serving adapter.
* ``timing``: detector plus postprocess per page at cap 1920 on 3 pages, at 8
  threads and at 2 threads.
* ``report``: merge the stage JSON files, apply the pass gate, write the
  report, and exit non-zero on failure.

Every stage except ``timing`` runs both runtimes at 4 threads, so production
keeps half the box. Thread count does not change the numerics measured here.

Example (one line per stage; each appends to its own log file)::

    BOX=/root/ppocrv6-onnx-parity-20260919
    LIBS=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 LD_LIBRARY_PATH=$LIBS
    cd $BOX
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run maps > run/maps.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run shapes > run/shapes.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run pipeline > run/pipeline.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run timing > run/timing.log 2>&1
    ./venv/bin/python verify_parity.py --model-dir ... --onnx ... \\
        --images /root/orli-bench/images \\
        --grec-images /root/ppocrv6-bench-20260919/images \\
        --workdir $BOX/run report \\
        --report-json $BOX/parity-report.json > run/report.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.segmentation.ppocr._fixtures import (
    SHAPE_CASES,
    STAGE_THREADS,
    TIMING_PAGES,
    build_tensor,
    config_float,
    make_ort_session,
    make_paddle_predictor,
    make_post_op,
    norm_kwargs,
    resolve_image_file,
    run_onnx,
    run_paddle,
    runtime_versions,
)
from scripts.segmentation.ppocr.verify_parity_maps import cmd_maps
from scripts.segmentation.ppocr.verify_parity_pipeline import cmd_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="PaddleX model directory")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported ONNX file")
    parser.add_argument("--images", type=Path, required=True, help="Directory of manuscript pages")
    parser.add_argument("--grec-images", type=Path, required=True, help="Directory with grec pages")
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Scratch dir for stage JSON and maps"
    )
    parser.add_argument("--report-json", type=Path, help="Where the report stage writes the report")
    parser.add_argument(
        "stage", choices=("maps", "shapes", "pipeline", "timing", "report"), help="Stage to run"
    )
    return parser.parse_args()


def cmd_shapes(args: argparse.Namespace) -> int:
    predictor = make_paddle_predictor(args.model_dir, STAGE_THREADS)
    session = make_ort_session(args.onnx, STAGE_THREADS)
    rng = np.random.default_rng(0)
    shape_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for label, height, width in SHAPE_CASES:
        tensor = rng.normal(0.0, 1.0, size=(1, 3, height, width)).astype(np.float32)
        paddle_map = run_paddle(predictor, tensor).astype(np.float64)
        onnx_map = run_onnx(session, tensor).astype(np.float64)
        max_abs = float(np.abs(paddle_map - onnx_map).max())
        ok_shape = tuple(int(v) for v in onnx_map.shape) == (1, 1, height, width)
        shape_rows.append(
            {
                "label": label,
                "input": [1, 3, height, width],
                "max_abs_prob_diff": max_abs,
                "onnx_shape_ok": ok_shape,
            }
        )
        print(f"shape {label}: maxabs={max_abs:.2e} shape_ok={ok_shape}", flush=True)
        if not ok_shape:
            failures.append(f"shape {label}: onnx output shape {tuple(onnx_map.shape)}")
        if max_abs > 1e-2:
            failures.append(f"shape {label}: max abs prob diff {max_abs:.2e}")
    (args.workdir / "shapes.json").write_text(
        json.dumps({"shapes": shape_rows, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"shapes done: failures={failures}", flush=True)
    return 0 if not failures else 1


def cmd_timing(args: argparse.Namespace) -> int:
    import cv2

    cv2.setNumThreads(1)
    normaliser, post_cfg = norm_kwargs(args.model_dir)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    timing_rows: list[dict[str, object]] = []
    for threads in (8, 2):
        predictor = make_paddle_predictor(args.model_dir, threads)
        session = make_ort_session(args.onnx, threads)
        for page in TIMING_PAGES:
            image_path = resolve_image_file(page, args.images, args.grec_images)
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not decode {image_path}")
            tensor, shape = build_tensor(bgr, cap=1920, normaliser=normaliser)
            paddle_samples, onnx_samples = [], []
            for repeat in range(4):
                started = time.perf_counter()
                paddle_map = run_paddle(predictor, tensor)
                post_op([paddle_map.astype(np.float32)], [shape])
                elapsed_paddle = time.perf_counter() - started
                started = time.perf_counter()
                onnx_map = run_onnx(session, tensor)
                post_op([onnx_map.astype(np.float32)], [shape])
                elapsed_onnx = time.perf_counter() - started
                if repeat >= 1:
                    paddle_samples.append(elapsed_paddle)
                    onnx_samples.append(elapsed_onnx)
            row: dict[str, object] = {
                "page": page,
                "threads": threads,
                "paddle_median_s": float(statistics.median(paddle_samples)),
                "onnx_median_s": float(statistics.median(onnx_samples)),
            }
            timing_rows.append(row)
            print(
                f"timing {page} threads={threads}: "
                f"paddle={row['paddle_median_s']:.3f}s onnx={row['onnx_median_s']:.3f}s",
                flush=True,
            )
        del predictor, session
    (args.workdir / "timing.json").write_text(
        json.dumps({"timing": timing_rows, "failures": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("timing done", flush=True)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.report_json is None:
        raise ValueError("report stage needs --report-json")
    maps = json.loads((args.workdir / "maps.json").read_text())
    shapes = json.loads((args.workdir / "shapes.json").read_text())
    pipeline = json.loads((args.workdir / "pipeline.json").read_text())
    timing = json.loads((args.workdir / "timing.json").read_text())
    failures: list[str] = (
        list(maps["failures"])
        + list(shapes["failures"])
        + list(pipeline["failures"])
        + list(timing["failures"])
    )
    report = {
        "versions": runtime_versions(),
        "model_dir": str(args.model_dir),
        "onnx": str(args.onnx),
        "per_page": maps["per_page"],
        "shapes": shapes["shapes"],
        "timing": timing["timing"],
        "pipeline_check": pipeline["pipeline_check"],
        "failures": failures,
        "passed": not failures,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in maps["per_page"]:
        print(
            f"{row['page']} cap={row['cap']} boxes "
            f"paddle={row['paddle_boxes']} onnx={row['onnx_boxes']} "
            f"delta={row['box_delta']} corner={row['max_corner_px']:.3f}px "
            f"maxabs={row['max_abs_prob_diff']:.2e} "
            f"meanabs={row['mean_abs_prob_diff']:.2e} "
            f"bindiff={row['binarisation_diff_px']}",
            flush=True,
        )
    for row in shapes["shapes"]:
        print(
            f"shape {row['label']}: maxabs={row['max_abs_prob_diff']:.2e} "
            f"shape_ok={row['onnx_shape_ok']}",
            flush=True,
        )
    for row in timing["timing"]:
        print(
            f"timing {row['page']} threads={row['threads']}: "
            f"paddle={row['paddle_median_s']:.3f}s onnx={row['onnx_median_s']:.3f}s",
            flush=True,
        )
    for row in pipeline["pipeline_check"]:
        print(
            f"pipeline {row['page']} cap={row['cap']}: boxes={row['pipeline_boxes']} "
            f"hand_delta={row['hand_delta']} hand_corner={row['hand_corner_px']:.3f}px",
            flush=True,
        )
    print(f"passed={report['passed']} failures={failures}", flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    args = _parse_args()
    if args.stage == "maps":
        return cmd_maps(args)
    if args.stage == "shapes":
        return cmd_shapes(args)
    if args.stage == "pipeline":
        return cmd_pipeline(args)
    if args.stage == "timing":
        return cmd_timing(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
