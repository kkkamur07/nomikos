"""PP-OCRv6 detection segmentation on the ONNX Runtime CPU runtime (ADR 0006).

Runs a PP-OCRv6 text detection graph (a DB detector: normalised image in,
probability map out) and returns the same segment contract the kraken blla
segmenter returns. ``resolve_artifact`` verifies the **artifact SHA-256**
before the file is opened.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from nomikos_inference.admission import open_image_bytes
from nomikos_inference.architectures.artifact import ArtifactHandle, resolve_artifact
from nomikos_inference.architectures.ppocr_det.polygons import (
    DEFAULT_OUTLINE_TOLERANCE_PX,
    DEFAULT_VERTICAL_GROWTH,
)
from nomikos_inference.architectures.ppocr_det.postprocessing import (
    DEFAULT_CONTOUR_TOLERANCE_PX,
    detect_lines,
)
from nomikos_inference.architectures.ppocr_det.preprocessing import preprocess_ppocr_det_image
from nomikos_inference.architectures.ppocr_det.reading_order import layout_lines
from nomikos_inference.architectures.ppocr_det.refinement import (
    DEFAULT_MERGE_GAP_RATIO,
    DEFAULT_MERGE_MAX_HEIGHT_RATIO,
    DEFAULT_MERGE_MAX_OVERLAP_RATIO,
    DEFAULT_OVERLAP_CUT_THRESHOLD,
    refine_to_lines,
)
from nomikos_inference.architectures.ppocr_det.response import (
    DEFAULT_BASELINE_FRACTION,
    build_ppocr_det_response,
    build_refined_ppocr_det_response,
)
from nomikos_inference.contracts.segment import SegmentRunResponse

PPOCR_DET_ARTIFACT_SUFFIXES = frozenset({".onnx"})

DEFAULT_LIMIT_SIDE_LEN = 1920
MIN_LIMIT_SIDE_LEN = 320
MAX_LIMIT_SIDE_LEN = 4000
DEFAULT_THRESH = 0.2
DEFAULT_BOX_THRESH = 0.45
DEFAULT_UNCLIP_RATIO = 1.4
DEFAULT_MAX_CANDIDATES = 3000
# Inference workers share one box: four processes on 8 cores. EXTENDED is
# about 2x faster than the ALL default with identical boxes, and on c13 it
# measured 5.85 s at 4 threads against 11.0 s at 2 threads and 3.6 s at 8
# (docs/inference/ppocrv6-onnx-performance-2026-09-20.md).
DEFAULT_PPOCR_DET_THREADS = 4
MAX_PPOCR_DET_THREADS = 64


class PPOCRDetUnavailableError(RuntimeError):
    """Raised when a PP-OCRv6 detection runtime artifact cannot be used."""


def _resolve_ppocr_det_artifact(
    model_path: Path,
    artifact_sha256: str | None = None,
) -> ArtifactHandle:
    return resolve_artifact(
        model_path,
        label="PP-OCRv6 det model",
        allowed_suffixes=PPOCR_DET_ARTIFACT_SUFFIXES,
        unusable_error=PPOCRDetUnavailableError,
        unusable_message=f"PP-OCRv6 det runtime requires an .onnx model: {model_path}",
        artifact_sha256=artifact_sha256,
    )


def _session_threads() -> int:
    """Intra-op threads for the detector session, from the environment."""
    raw = os.environ.get("NOMIKOS_PPOCR_DET_THREADS")
    if raw is None or raw == "":
        return DEFAULT_PPOCR_DET_THREADS
    try:
        threads = int(raw)
    except ValueError:
        raise PPOCRDetUnavailableError(
            "NOMIKOS_PPOCR_DET_THREADS must be an integer between 1 and "
            f"{MAX_PPOCR_DET_THREADS}, got {raw!r}"
        ) from None
    if not 1 <= threads <= MAX_PPOCR_DET_THREADS:
        raise PPOCRDetUnavailableError(
            "NOMIKOS_PPOCR_DET_THREADS must be an integer between 1 and "
            f"{MAX_PPOCR_DET_THREADS}, got {raw!r}"
        )
    return threads


@lru_cache(maxsize=4)
def _load_ppocr_det_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[Any, str, str]:
    """Open a session and read the tensor names this decoder needs.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = _session_threads()
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must have one input and output")
        if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
            raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must use 4D tensors")
        return session, inputs[0].name, outputs[0].name
    except PPOCRDetUnavailableError:
        raise
    except ImportError as error:
        raise PPOCRDetUnavailableError(
            "onnxruntime is required for the PP-OCRv6 det runtime"
        ) from error
    except Exception as error:
        raise PPOCRDetUnavailableError("unable to load PP-OCRv6 det ONNX model") from error


def _bounded_float_param(
    params: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    """Parse a caller-supplied number inside documented bounds.

    Bools, non-numeric values, non-finite values and out-of-range values
    raise the same ``ValueError`` the other bounds checks raise, naming the
    param and the bound, instead of failing the page silently later (``inf``
    thresholds empty the page, ``inf`` unclip ratios crash the offsetter).
    """
    value = params.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number between {minimum} and {maximum}")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number between {minimum} and {maximum}") from None
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{key} must be a number between {minimum} and {maximum}")
    return parsed


def _limit_side_len(params: Mapping[str, Any]) -> int:
    value = params.get("limit_side_len", DEFAULT_LIMIT_SIDE_LEN)
    if isinstance(value, bool):
        raise ValueError("limit_side_len must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("limit_side_len must be an integer") from None
    if isinstance(value, float) and float(parsed) != value:
        raise ValueError("limit_side_len must be an integer")
    if parsed < MIN_LIMIT_SIDE_LEN or parsed > MAX_LIMIT_SIDE_LEN:
        raise ValueError(
            f"limit_side_len must be between {MIN_LIMIT_SIDE_LEN} and {MAX_LIMIT_SIDE_LEN}"
        )
    return parsed


def _max_candidates(params: Mapping[str, Any]) -> int:
    value = params.get("max_candidates", DEFAULT_MAX_CANDIDATES)
    if isinstance(value, bool):
        raise ValueError("max_candidates must be an integer between 1 and 10000")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_candidates must be an integer between 1 and 10000") from None
    if isinstance(value, float) and float(parsed) != value:
        raise ValueError("max_candidates must be an integer between 1 and 10000")
    if not 1 <= parsed <= 10000:
        raise ValueError("max_candidates must be an integer between 1 and 10000")
    return parsed


def _baseline_fraction(params: Mapping[str, Any]) -> float:
    value = params.get("baseline_fraction", DEFAULT_BASELINE_FRACTION)
    if isinstance(value, bool):
        raise ValueError("baseline_fraction must be a number between 0 and 1")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("baseline_fraction must be a number between 0 and 1") from None
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError("baseline_fraction must be a number between 0 and 1")
    return parsed


def _reading_direction(params: Mapping[str, Any]) -> str:
    direction = params.get("reading_direction", "ltr")
    if direction not in ("ltr", "rtl"):
        raise ValueError('reading_direction must be "ltr" or "rtl"')
    return direction


#: Served geometry when the caller names none: line polygons. The library
#: helpers (``detect_lines``, ``refine_to_lines``, both response builders)
#: all default to ``"quad"``; this is the served default only.
DEFAULT_BOX_TYPE = "poly"


def _box_type(params: Mapping[str, Any], default: str = "quad") -> str:
    box_type = params.get("box_type", default)
    if box_type not in ("poly", "quad"):
        raise ValueError('box_type must be "poly" or "quad"')
    return box_type


def _bool_param(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be true or false")


def _noise_policy(params: Mapping[str, Any]) -> str:
    policy = params.get("noise_policy", "flag")
    if policy in ("flag", "drop", "off"):
        return policy
    raise ValueError('noise_policy must be "flag", "drop" or "off"')


def run_ppocr_det_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> SegmentRunResponse:
    """Run PP-OCRv6 detection and return the segment contract in reading order."""

    # Resolved once here so a missing, mis-suffixed or tampered artifact fails
    # before the resize is paid for; the session loader re-resolves from a
    # memoized digest, which costs a ``stat``.
    _resolve_ppocr_det_artifact(model_path, artifact_sha256)
    resolved = params or {}
    limit = _limit_side_len(resolved)
    thresh = _bounded_float_param(resolved, "thresh", DEFAULT_THRESH, 0, 1)
    box_thresh = _bounded_float_param(resolved, "box_thresh", DEFAULT_BOX_THRESH, 0, 1)
    unclip_ratio = _bounded_float_param(resolved, "unclip_ratio", DEFAULT_UNCLIP_RATIO, 0, 5)
    max_candidates = _max_candidates(resolved)
    fraction = _baseline_fraction(resolved)
    direction = _reading_direction(resolved)
    box_type = _box_type(resolved, DEFAULT_BOX_TYPE)
    merge_fragments = _bool_param(resolved, "merge_fragments", True)
    resolve_overlaps = _bool_param(resolved, "resolve_overlaps", True)
    noise_policy = _noise_policy(resolved)
    merge_gap_ratio = _bounded_float_param(
        resolved, "merge_gap_ratio", DEFAULT_MERGE_GAP_RATIO, 0, 10
    )
    merge_max_overlap_ratio = _bounded_float_param(
        resolved, "merge_max_overlap_ratio", DEFAULT_MERGE_MAX_OVERLAP_RATIO, 0, 2
    )
    merge_max_height_ratio = _bounded_float_param(
        resolved, "merge_max_height_ratio", DEFAULT_MERGE_MAX_HEIGHT_RATIO, 1, 10
    )
    overlap_cut_threshold = _bounded_float_param(
        resolved, "overlap_cut_threshold", DEFAULT_OVERLAP_CUT_THRESHOLD, 0.05, 1
    )
    contour_tolerance_px = _bounded_float_param(
        resolved, "contour_tolerance_px", DEFAULT_CONTOUR_TOLERANCE_PX, 0, 10
    )
    outline_tolerance_px = _bounded_float_param(
        resolved, "outline_tolerance_px", DEFAULT_OUTLINE_TOLERANCE_PX, 0, 10
    )
    vertical_growth = _bounded_float_param(
        resolved, "vertical_growth", DEFAULT_VERTICAL_GROWTH, 1.0, 2.0
    )

    with open_image_bytes(image_bytes) as image:
        image = image.convert("RGB")
        width, height = image.size
        tensor, meta = preprocess_ppocr_det_image(image, limit_side_len=limit)
        _, _, resized_height, resized_width = tensor.shape

        handle = _resolve_ppocr_det_artifact(model_path, artifact_sha256)
        session, input_name, output_name = _load_ppocr_det_session(handle.path, handle.fingerprint)
        outputs = session.run([output_name], {input_name: np.ascontiguousarray(tensor)})
        prob = np.asarray(outputs[0], dtype=np.float32)
        if prob.shape != (1, 1, resized_height, resized_width):
            raise PPOCRDetUnavailableError(
                "PP-OCRv6 det ONNX graph must return [1, 1, H, W] matching the input"
            )
        quads = detect_lines(
            prob[0, 0],
            orig_width=width,
            orig_height=height,
            ratio_h=meta.ratio_h,
            ratio_w=meta.ratio_w,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates,
            box_type=box_type,
            contour_tolerance_px=contour_tolerance_px,
        )
        if not merge_fragments and not resolve_overlaps and noise_policy == "off":
            return build_ppocr_det_response(
                width,
                height,
                quads,
                baseline_fraction=fraction,
                reading_direction=direction,
                box_type=box_type,
                outline_tolerance_px=outline_tolerance_px,
                vertical_growth=vertical_growth,
            )
        layout = layout_lines(quads, direction=direction)
        items = refine_to_lines(
            quads,
            layout,
            baseline_fraction=fraction,
            merge=merge_fragments,
            resolve=resolve_overlaps,
            classify=noise_policy != "off",
            merge_gap_ratio=merge_gap_ratio,
            merge_max_overlap_ratio=merge_max_overlap_ratio,
            merge_max_height_ratio=merge_max_height_ratio,
            overlap_cut_threshold=overlap_cut_threshold,
            box_type=box_type,
            page_width=width,
            page_height=height,
            outline_tolerance_px=outline_tolerance_px,
            vertical_growth=vertical_growth,
        )
        return build_refined_ppocr_det_response(
            width,
            height,
            quads,
            items,
            layout,
            baseline_fraction=fraction,
            reading_direction=direction,
            noise_policy=noise_policy,
            box_type=box_type,
        )


__all__ = [
    "PPOCRDetUnavailableError",
    "run_ppocr_det_segment",
]
