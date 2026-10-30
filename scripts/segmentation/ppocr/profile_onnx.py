#!/usr/bin/env python3
"""Profile the PP-OCRv6 medium-det ONNX and test cheap speedups.

ADR 0006 serves models through onnxruntime on CPU, and the accepted export
runs about 4x slower than Paddle mkldnn on the same tensors. This script
finds out why and tests the cheap fixes. It runs as one subcommand at a
time, each writing its own log under the box `perf/` directory, with at
most 4 threads unless a subcommand says otherwise:

* ``counts``: count node types in the graph (paddle2onnx costs show up here
  as decomposed activation chains, Casts, and Shape/Gather/Concat chains).
* ``profile``: run onnxruntime with profiling on c13 at 4 threads and
  report the top 10 node types by total time and the top 10 single nodes.
* ``sessions``: cheap session experiments on the accepted file on c13
  (median of 3): graph optimisation level, execution mode, thread count,
  spinning, denormal-as-zero. Split across runs with ``--only`` so no
  single command runs longer than 3 minutes.
* ``candidate``: check one candidate file (onnx.checker, load in
  onnxruntime 1.30.0), run the 5 shape cases, and time it on c13
  (1 warmup and 3 timed runs at 4 threads, detector plus postprocess).

Example::

    BOX=/root/ppocrv6-onnx-parity-20260919
    LIBS=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu
    cd $BOX
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LD_LIBRARY_PATH=$LIBS \\
        ./venv/bin/python scripts/segmentation/ppocr/profile_onnx.py \\
        --onnx pp-ocrv6-medium-det.onnx --model-dir ... --images ... \\
        --grec-images ... --perfdir perf profile > perf/profile.log 2>&1
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.segmentation.ppocr._fixtures import (
    SHAPE_CASES,
    build_tensor,
    config_float,
    make_paddle_predictor,
    make_post_op,
    norm_kwargs,
    resolve_image_file,
    run_onnx,
    run_paddle,
)

ORT_VERSION_PIN = "1.30.0"
PROFILE_PAGE = "c13"
PROFILE_CAP = 1920


class _SessionOptions(TypedDict, total=False):
    """One sessions-subset row: _make_session overrides for a labelled run."""

    threads: int
    opt_level: str
    exec_mode: str
    extra_entries: dict[str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX file under test")
    parser.add_argument("--model-dir", type=Path, required=True, help="PaddleX model directory")
    parser.add_argument("--images", type=Path, required=True, help="Directory of manuscript pages")
    parser.add_argument("--grec-images", type=Path, required=True, help="Directory with grec pages")
    parser.add_argument("--perfdir", type=Path, required=True, help="Directory for profiles")
    parser.add_argument(
        "stage",
        choices=("counts", "profile", "sessions", "candidate", "compare", "final"),
        help="Step to run",
    )
    parser.add_argument(
        "--only",
        choices=("opt1", "opt2", "t1", "t4", "misc1", "misc2"),
        help="sessions subset, kept small so no command runs over 3 minutes",
    )
    return parser.parse_args()


def _load_model(path: Path):
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    return model


def _node_counts(model) -> collections.Counter[str]:
    return collections.Counter(node.op_type for node in model.graph.node)


def cmd_counts(args: argparse.Namespace) -> int:
    model = _load_model(args.onnx)
    counts = _node_counts(model)
    print(f"nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")
    for op_type, count in counts.most_common():
        print(f"{op_type}: {count}")
    return 0


def _c13_tensor(args: argparse.Namespace):
    import cv2

    cv2.setNumThreads(1)
    normaliser, post_cfg = norm_kwargs(args.model_dir)
    image_path = resolve_image_file(PROFILE_PAGE, args.images, args.grec_images)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not decode {image_path}")
    tensor, shape = build_tensor(bgr, cap=PROFILE_CAP, normaliser=normaliser)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    return tensor, shape, post_op


def _make_session(
    onnx_path: Path,
    threads: int,
    *,
    opt_level: str = "ALL",
    exec_mode: str = "SEQUENTIAL",
    extra_entries: dict[str, str] | None = None,
    profiling: Path | None = None,
):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = {
        "DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[opt_level]
    options.execution_mode = getattr(ort.ExecutionMode, "ORT_" + exec_mode)
    for key, value in (extra_entries or {}).items():
        options.add_session_config_entry(key, value)
    if profiling is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profiling)
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def cmd_profile(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    if ort.__version__ != ORT_VERSION_PIN:
        raise RuntimeError(f"expected onnxruntime {ORT_VERSION_PIN}, got {ort.__version__}")
    tensor, _, _ = _c13_tensor(args)
    args.perfdir.mkdir(parents=True, exist_ok=True)
    session = _make_session(args.onnx, 4, profiling=args.perfdir / "ort-profile")
    run_onnx(session, tensor)
    run_onnx(session, tensor)
    profile_path = Path(session.end_profiling())
    print(f"profile={profile_path}")
    events = json.loads(profile_path.read_text()) if profile_path.is_file() else []
    nodes = [event for event in events if event.get("cat") == "Node"]
    print(f"node_events={len(nodes)}")
    by_type: dict[str, float] = collections.defaultdict(float)
    by_type_count: dict[str, int] = collections.Counter()
    for event in nodes:
        op_name = str(event.get("args", {}).get("op_name", "?"))
        dur = float(event.get("dur", 0.0))
        by_type[op_name] += dur
        by_type_count[op_name] += 1
    print("top node types by total time (microseconds):")
    for op_name, total in sorted(by_type.items(), key=lambda item: -item[1])[:10]:
        print(f"  {op_name}: total={total:.0f} count={by_type_count[op_name]}")
    print("top single nodes by time (microseconds):")
    ranked = sorted(nodes, key=lambda event: -float(event.get("dur", 0.0)))[:10]
    for event in ranked:
        event_args = event.get("args", {})
        print(
            f"  {event.get('name')}: op={event_args.get('op_name')} "
            f"dur={float(event.get('dur', 0.0)):.0f}"
        )
    return 0


def _median_of_3(session, tensor: np.ndarray, shape: np.ndarray, post_op) -> float:
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        prob = run_onnx(session, tensor)
        post_op([prob.astype(np.float32)], [shape])
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def _pop_extra_entries(kwargs: _SessionOptions) -> dict[str, str] | None:
    """The session config entries for one subset row, if it sets any."""
    entries = kwargs.pop("extra_entries", None)
    if entries is None:
        return None
    return dict(entries)


def cmd_sessions(args: argparse.Namespace) -> int:
    tensor, shape, post_op = _c13_tensor(args)
    _run_warmup = _make_session(args.onnx, 4)
    run_onnx(_run_warmup, tensor)
    del _run_warmup
    subsets: dict[str, list[tuple[str, _SessionOptions]]] = {
        "opt1": [
            ("opt=ALL", {}),
            ("opt=EXTENDED", {"opt_level": "EXTENDED"}),
        ],
        "opt2": [("opt=BASIC", {"opt_level": "BASIC"})],
        "t1": [("threads=1", {"threads": 1})],
        "t4": [("threads=4", {"threads": 4})],
        "misc1": [
            ("exec=PARALLEL", {"exec_mode": "PARALLEL"}),
            ("spinning=off", {"extra_entries": {"session.intra_op.allow_spinning": "0"}}),
        ],
        "misc2": [
            ("denormal_as_zero", {"extra_entries": {"session.set_denormal_as_zero": "1"}}),
        ],
    }
    only = args.only or "opt1"
    print("c13, detector plus postprocess, median of 3 after 1 warmup:")
    for label, kwargs in subsets[only]:
        threads = kwargs.pop("threads", 4)
        session = _make_session(
            args.onnx,
            threads,
            opt_level=str(kwargs.pop("opt_level", "ALL")),
            exec_mode=str(kwargs.pop("exec_mode", "SEQUENTIAL")),
            extra_entries=_pop_extra_entries(kwargs),
        )
        run_onnx(session, tensor)
        median = _median_of_3(session, tensor, shape, post_op)
        print(f"{label}: median={median:.3f}s", flush=True)
        del session
    return 0


def cmd_candidate(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    if ort.__version__ != ORT_VERSION_PIN:
        raise RuntimeError(f"expected onnxruntime {ORT_VERSION_PIN}, got {ort.__version__}")
    model = _load_model(args.onnx)
    counts = _node_counts(model)
    print(f"checker=ok nodes={len(model.graph.node)}")
    session = _make_session(args.onnx, 4)
    print(f"load=ok inputs={session.get_inputs()[0].name}")
    rng = np.random.default_rng(0)
    for label, height, width in SHAPE_CASES:
        tensor = rng.normal(0.0, 1.0, size=(1, 3, height, width)).astype(np.float32)
        out = run_onnx(session, tensor)
        ok = tuple(int(v) for v in out.shape) == (1, 1, height, width)
        print(f"shape {label}: ok={ok}")
        if not ok:
            return 1
    tensor, shape, post_op = _c13_tensor(args)
    run_onnx(session, tensor)
    median = _median_of_3(session, tensor, shape, post_op)
    print(f"c13 median={median:.3f}s")
    print(f"top node types: {counts.most_common(8)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Check EXTENDED and denormal_as_zero change nothing but speed (3 pages)."""
    import cv2

    cv2.setNumThreads(1)
    normaliser, post_cfg = norm_kwargs(args.model_dir)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    ref = _make_session(args.onnx, 4)
    fast = _make_session(args.onnx, 4, opt_level="EXTENDED")
    dz = _make_session(args.onnx, 4, extra_entries={"session.set_denormal_as_zero": "1"})
    for page in ("c13", "vat-1r", "grec-p1"):
        image_path = resolve_image_file(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        tensor, shape = build_tensor(bgr, cap=PROFILE_CAP, normaliser=normaliser)
        ref_map = run_onnx(ref, tensor).astype(np.float64)
        ref_boxes, _ = post_op([ref_map.astype(np.float32)], [shape])
        ref_boxes = np.asarray(ref_boxes, dtype=np.float64).reshape(-1, 4, 2)
        for label, session in (("extended", fast), ("denorm", dz)):
            other_map = run_onnx(session, tensor).astype(np.float64)
            diff = float(np.abs(ref_map - other_map).max())
            other_boxes, _ = post_op([other_map.astype(np.float32)], [shape])
            other_boxes = np.asarray(other_boxes, dtype=np.float64).reshape(-1, 4, 2)
            delta = int(len(ref_boxes) - len(other_boxes))
            corner = (
                float(np.linalg.norm(ref_boxes - other_boxes, axis=2).max())
                if delta == 0 and len(ref_boxes) > 0
                else (0.0 if delta == 0 else float("inf"))
            )
            print(
                f"{page} {label}: maxabs={diff:.2e} boxes={len(other_boxes)} "
                f"delta={delta} corner={corner:.3f}px",
                flush=True,
            )
    return 0


def cmd_final(args: argparse.Namespace) -> int:
    """Final timing: accepted file at EXTENDED on 3 pages plus the big tensor."""
    import cv2

    cv2.setNumThreads(1)

    normaliser, post_cfg = norm_kwargs(args.model_dir)
    thresh = config_float(post_cfg, "thresh", 0.2)
    box_thresh = config_float(post_cfg, "box_thresh", 0.45)
    post_op = make_post_op(post_cfg, thresh, box_thresh)
    sugars: list[tuple[str, np.ndarray, np.ndarray]] = []
    for page in ("c13", "vat-1r", "grec-p1"):
        image_path = resolve_image_file(page, args.images, args.grec_images)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {image_path}")
        tensor, shape = build_tensor(bgr, cap=PROFILE_CAP, normaliser=normaliser)
        sugars.append((page, tensor, shape))
    vat_path = resolve_image_file("vat-1r", args.images, args.grec_images)
    vat = cv2.imread(str(vat_path), cv2.IMREAD_COLOR)
    if vat is None:
        raise RuntimeError(f"could not decode {vat_path}")
    big_bgr = cv2.resize(vat, (1280, 1920), interpolation=cv2.INTER_LINEAR)
    big_tensor, big_shape = build_tensor(big_bgr, cap=PROFILE_CAP, normaliser=normaliser)
    print(f"big tensor shape={[int(v) for v in big_tensor.shape]}", flush=True)
    sugars.append(("big-1920x1280", big_tensor, big_shape))
    for threads in (8, 2):
        session = _make_session(args.onnx, threads, opt_level="EXTENDED")
        for page, tensor, shape in sugars:
            run_onnx(session, tensor)
            median = _median_of_3(session, tensor, shape, post_op)
            print(f"onnx-extended {page} threads={threads}: {median:.3f}s", flush=True)
        del session
    predictor = make_paddle_predictor(args.model_dir, 8)
    run_paddle(predictor, big_tensor)
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        prob = run_paddle(predictor, big_tensor)
        post_op([prob.astype(np.float32)], [big_shape])
        samples.append(time.perf_counter() - started)
    print(f"paddle big-1920x1280 threads=8: {float(statistics.median(samples)):.3f}s", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.stage == "counts":
        return cmd_counts(args)
    if args.stage == "profile":
        return cmd_profile(args)
    if args.stage == "sessions":
        return cmd_sessions(args)
    if args.stage == "compare":
        return cmd_compare(args)
    if args.stage == "final":
        return cmd_final(args)
    return cmd_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
