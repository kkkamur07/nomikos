#!/usr/bin/env python3
"""Shared loaders and overlay helpers for the PP-OCR segmentation scripts.

``verify_parity.py`` (export-vs-Paddle measurement), ``verify_adapter.py``
(serving-adapter measurement) and ``evaluate_refinement.py`` (refinement
scoring) walk the same 14 reference pages and draw the same reading-order
overlays. This module owns the page list, the fixture/image loading with
SHA-256 checks, and the overlay canvas setup so the three scripts cannot
drift apart. It is a library, not a script: no argparse, no ``main``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shapely.geometry import Polygon

COPTIC_PAGES = (
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
)
PAGES = COPTIC_PAGES + ("grec-p1", "grec-p4")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
CAPS = (1920, 960)
MAP_PAGES = ("c13", "grec-p1")
TIMING_PAGES = ("c13", "vat-1r", "grec-p1")
# (label, height, width): small, very wide, tall, and two odd multiples of 32.
SHAPE_CASES = (
    ("320x320", 320, 320),
    ("wide-1920x320", 320, 1920),
    ("tall-640x1920", 1920, 640),
    ("odd-480x736", 736, 480),
    ("odd-1056x864", 864, 1056),
)
# Stages other than timing share the box with production at 4 threads.
STAGE_THREADS = 4


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of in-memory bytes (fixtures, page images)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, chunked so multi-hundred-megabyte artifacts fit."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fixture(fixtures: Path, page: str) -> dict:
    """Parse one ``<page>.json`` reference fixture."""
    return json.loads((fixtures / f"{page}.json").read_text(encoding="utf-8"))


def resolve_image_file(name: str, images: Path, grec_images: Path) -> Path:
    """First existing ``<name><suffix>`` under the manuscript image roots."""
    for directory in (images, grec_images):
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no image for page {name} under {images} or {grec_images}")


def load_image_bytes(page: str, fixtures: Path, images: Path) -> bytes:
    """Read the page image, refusing it unless its SHA-256 matches the fixture."""
    fixture = read_fixture(fixtures, page)
    data = (images / fixture["image"]).read_bytes()
    if sha256_bytes(data) != fixture["image_sha256"]:
        raise ValueError(f"image SHA-256 does not match fixture {page}.json")
    return data


def load_targets(gt_dir: Path, page: str) -> list[dict]:
    """Care and ignore target polygons, as in the benchmark's capture_inputs."""
    doc = json.loads((gt_dir / f"{page}-latest.json").read_text(encoding="utf-8"))
    paired = {
        text["paired_line_id"]
        for text in doc["pairing"]["text_lines"]
        if text.get("paired_line_id")
    }
    targets = []
    for line in doc["lines"]:
        polygon = Polygon(line["points"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        targets.append({"id": line["id"], "polygon": polygon, "care": line["id"] in paired})
    return targets


def resolve_verified_image(page: str, fixture: dict, images: Path, grec_images: Path) -> bytes:
    """Read the page image, refusing it unless its SHA-256 matches the fixture."""
    expected_name = fixture["image"]
    expected_sha = fixture["image_sha256"]
    directories = (images, grec_images)
    candidate = _find_named_image(page, expected_name, directories)
    if candidate is not None:
        data = candidate.read_bytes()
        if sha256_bytes(data) != expected_sha:
            raise ValueError(f"{candidate} SHA-256 does not match fixture {page}.json")
        return data
    for directory in directories:
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{page}{suffix}"
            if candidate.is_file() and candidate.name != expected_name:
                data = candidate.read_bytes()
                if sha256_bytes(data) == expected_sha:
                    return data
    raise FileNotFoundError(f"no image matching fixture {page}.json (sha {expected_sha[:12]}…)")


def _find_named_image(page: str, expected_name: str, directories: tuple[Path, Path]) -> Path | None:
    """The fixture's exact image name, or the page stem with the same suffix."""
    for directory in directories:
        for name in (expected_name, f"{page}{Path(expected_name).suffix}"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def decode_overlay_canvas(image_bytes: bytes, page: str):
    """Decode page bytes to a BGR canvas plus the shared scale metrics."""
    import cv2
    import numpy as np

    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    canvas = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError(f"could not decode image for page {page}")
    height, width = canvas.shape[:2]
    scale = max(width, height) / 1400.0
    return canvas, scale, width, height


def put_order_label(
    canvas,
    number: int,
    left_x: float,
    mid_y: float,
    scale: float,
    width: int,
    height: int,
) -> None:
    """Black-outlined white order number beside the quad's left edge."""
    import cv2

    font_scale = max(0.7, 1.1 * scale)
    font_thickness = max(1, int(round(2 * scale)))
    label = str(number)
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


def config_float(post_cfg: dict[str, object], key: str, default: float) -> float:
    """A postprocess config number, validated: a wrong type is refused, not cast."""
    value = post_cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"unexpected postprocess config: {key}={value!r}")
    return float(value)


def config_int(post_cfg: dict[str, object], key: str, default: int) -> int:
    """A postprocess config integer, validated like config_float."""
    value = post_cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"unexpected postprocess config: {key}={value!r}")
    return int(value)


def read_prepost(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """PaddleX pre/postprocess configs: normalised op map plus the DB postprocess."""
    import yaml

    config = yaml.safe_load((model_dir / "inference.yml").read_text())
    pre_ops: dict[str, object] = {}
    for entry in config["PreProcess"]["transform_ops"]:
        key = next(iter(entry))
        pre_ops[key] = entry[key] or {}
    return pre_ops, config["PostProcess"]


def norm_kwargs(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Normaliser kwargs with the PaddleX defaults, plus the postprocess config."""
    pre_ops, post_cfg = read_prepost(model_dir)
    norm_entry = pre_ops["NormalizeImage"]
    if not isinstance(norm_entry, dict):
        raise ValueError(f"unexpected NormalizeImage config: {norm_entry!r}")
    return (
        {
            "scale": norm_entry.get("scale", 1.0 / 255.0),
            "mean": norm_entry.get("mean", [0.485, 0.456, 0.406]),
            "std": norm_entry.get("std", [0.229, 0.224, 0.225]),
            "order": norm_entry.get("order", "hwc"),
        },
        post_cfg,
    )


def build_tensor(bgr, *, cap: int, normaliser: dict[str, object]):
    """Run PaddleX's own preprocessing classes over one BGR image."""
    import numpy as np
    from paddlex.inference.models.common import ToBatch, ToCHWImage
    from paddlex.inference.models.text_detection.processors import (
        DetResizeForTest,
        NormalizeImage,
    )

    resizer = DetResizeForTest(limit_side_len=cap, limit_type="max", input_shape=None)
    resized, shapes = resizer(imgs=[bgr])
    normalizer = NormalizeImage(
        scale=normaliser["scale"],
        mean=normaliser["mean"],
        std=normaliser["std"],
        order=normaliser["order"],
    )
    normalized = normalizer(imgs=resized)
    chw = ToCHWImage()(imgs=normalized)
    batch = ToBatch()(imgs=chw)[0]
    return batch.astype(np.float32, copy=False), np.asarray(shapes[0])


def make_post_op(post_cfg: dict[str, object], thresh: float, box_thresh: float):
    """The reference DB postprocess, with the model's own thresholds."""
    from paddlex.inference.models.text_detection.processors import DBPostProcess

    if post_cfg.get("name") != "DBPostProcess":
        raise ValueError(f"unexpected postprocess: {post_cfg.get('name')}")
    return DBPostProcess(
        thresh=thresh,
        box_thresh=box_thresh,
        max_candidates=config_int(post_cfg, "max_candidates", 1000),
        unclip_ratio=config_float(post_cfg, "unclip_ratio", 1.4),
        use_dilation=bool(post_cfg.get("use_dilation", False)),
        score_mode=str(post_cfg.get("score_mode", "fast")),
        box_type=str(post_cfg.get("box_type", "quad")),
    )


def make_paddle_predictor(model_dir: Path, threads: int):
    """Mirror the PaddleX CPU recipe used by the reference benchmark."""
    from paddle import inference as paddle_inference

    config = paddle_inference.Config(
        str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams")
    )
    config.disable_gpu()
    config.enable_mkldnn()
    config.set_mkldnn_cache_capacity(10)
    config.set_cpu_math_library_num_threads(threads)
    if hasattr(config, "enable_new_ir"):
        config.enable_new_ir(True)
    if hasattr(config, "enable_new_executor"):
        config.enable_new_executor()
    config.set_optimization_level(3)
    config.enable_memory_optim()
    config.disable_glog_info()
    return paddle_inference.create_predictor(config)


def run_paddle(predictor, tensor):
    """Run one tensor through the Paddle predictor; returns the probability map."""
    import numpy as np

    input_names = predictor.get_input_names()
    handle = predictor.get_input_handle(input_names[0])
    handle.reshape(list(tensor.shape))
    handle.copy_from_cpu(tensor)
    predictor.run()
    output_names = predictor.get_output_names()
    return np.asarray(predictor.get_output_handle(output_names[0]).copy_to_cpu())


def make_ort_session(onnx_path: Path, threads: int):
    """ONNX session with the stage thread recipe (all compute intra-op)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_onnx(session, tensor):
    """Run one tensor through onnxruntime; returns the probability map."""
    import numpy as np

    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: tensor})[0])


def as_boxes(raw: object):
    """DB postprocess output to an (N, 4, 2) float box array, empty-safe."""
    import numpy as np

    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float64)
    return array.reshape(-1, 4, 2)


def max_corner_distance(a, b) -> float:
    """Greedy-bijection max corner distance between two equal-size box sets."""
    import numpy as np

    if len(a) == 0 and len(b) == 0:
        return 0.0
    remaining_b = list(range(len(b)))
    worst = 0.0
    for box_a in a:
        best_j, best_cost = -1, float("inf")
        for j in remaining_b:
            cost = float(np.linalg.norm(box_a - b[j], axis=1).max())
            if cost < best_cost:
                best_j, best_cost = j, cost
        remaining_b.remove(best_j)
        worst = max(worst, best_cost)
    return worst


def runtime_versions() -> dict[str, str]:
    """Pinned runtime versions for parity reports."""
    import importlib.metadata as metadata

    return {
        name: metadata.version(name)
        for name in ("paddlepaddle", "paddleocr", "paddlex", "onnxruntime", "numpy")
    }
