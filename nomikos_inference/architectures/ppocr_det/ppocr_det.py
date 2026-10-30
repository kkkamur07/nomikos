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
from types import ModuleType
from typing import SupportsFloat, SupportsIndex, SupportsInt

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
from nomikos_inference.architectures.shared.onnx_session import OnnxSession, open_checked_session
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


def _session_options(ort: ModuleType) -> object:
    """Sequential EXTENDED-optimised session, one inter-op thread."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = _session_threads()
    return options


def _checked_ppocr_det_session(session: OnnxSession) -> tuple[OnnxSession, str, str]:
    """The input and output names of a graph this decoder can read."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must have one input and output")
    if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
        raise PPOCRDetUnavailableError("PP-OCRv6 det ONNX graph must use 4D tensors")
    return session, inputs[0].name, outputs[0].name


@lru_cache(maxsize=4)
def _load_ppocr_det_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[OnnxSession, str, str]:
    """Open a session and read the tensor names this decoder needs.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    return open_checked_session(
        model_path,
        unavailable=PPOCRDetUnavailableError,
        missing_runtime_message="onnxruntime is required for the PP-OCRv6 det runtime",
        load_failure_message="unable to load PP-OCRv6 det ONNX model",
        check=_checked_ppocr_det_session,
        session_options=_session_options,
    )


def _bounded_float_param(
    params: Mapping[str, object], key: str, default: float, minimum: float, maximum: float
) -> float:
    """Parse a caller-supplied number inside documented bounds.

    Bools, non-numeric values, non-finite values and out-of-range values
    raise the same ``ValueError`` the other bounds checks raise, naming the
    param and the bound, instead of failing the page silently later (``inf``
    thresholds empty the page, ``inf`` unclip ratios crash the offsetter).
    """
    message = f"{key} must be a number between {minimum} and {maximum}"
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(
        value, str | bytes | SupportsFloat | SupportsIndex
    ):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(message)
    return parsed


def _bounded_int_param(
    params: Mapping[str, object],
    key: str,
    default: int,
    bounds: tuple[int, int],
    *,
    type_message: str,
    range_message: str,
) -> int:
    """Parse a caller-supplied integer inside ``bounds``, inclusive.

    A bool, a non-numeric value or a float with a fractional part raises
    ``type_message``; an integer outside the bounds raises ``range_message``.
    """
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, str | bytes | SupportsInt | SupportsIndex):
        raise ValueError(type_message)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(type_message) from None
    if isinstance(value, float) and float(parsed) != value:
        raise ValueError(type_message)
    if not bounds[0] <= parsed <= bounds[1]:
        raise ValueError(range_message)
    return parsed


def _limit_side_len(params: Mapping[str, object]) -> int:
    return _bounded_int_param(
        params,
        "limit_side_len",
        DEFAULT_LIMIT_SIDE_LEN,
        (MIN_LIMIT_SIDE_LEN, MAX_LIMIT_SIDE_LEN),
        type_message="limit_side_len must be an integer",
        range_message=(
            f"limit_side_len must be between {MIN_LIMIT_SIDE_LEN} and {MAX_LIMIT_SIDE_LEN}"
        ),
    )


def _max_candidates(params: Mapping[str, object]) -> int:
    message = "max_candidates must be an integer between 1 and 10000"
    return _bounded_int_param(
        params,
        "max_candidates",
        DEFAULT_MAX_CANDIDATES,
        (1, 10000),
        type_message=message,
        range_message=message,
    )


def _choice_param(
    params: Mapping[str, object], key: str, default: str, choices: tuple[str, ...], message: str
) -> str:
    value = params.get(key, default)
    if isinstance(value, str) and value in choices:
        return value
    raise ValueError(message)


#: Served geometry when the caller names none: line polygons. The library
#: helpers (``detect_lines``, ``refine_to_lines``, both response builders)
#: all default to ``"quad"``; this is the served default only.
DEFAULT_BOX_TYPE = "poly"


def _box_type(params: Mapping[str, object], default: str = "quad") -> str:
    return _choice_param(
        params, "box_type", default, ("poly", "quad"), 'box_type must be "poly" or "quad"'
    )


def _bool_param(params: Mapping[str, object], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be true or false")


def run_ppocr_det_segment(
    image_bytes: bytes,
    *,
    model_path: Path,
    artifact_sha256: str | None = None,
    params: Mapping[str, object] | None = None,
) -> SegmentRunResponse:
    """Run PP-OCRv6 detection and return the segment contract in reading order."""

    # Resolved once, up front, so a missing, mis-suffixed or tampered artifact
    # fails before the resize is paid for; the same handle opens the session.
    handle = _resolve_ppocr_det_artifact(model_path, artifact_sha256)
    resolved = params or {}
    limit = _limit_side_len(resolved)
    thresh = _bounded_float_param(resolved, "thresh", DEFAULT_THRESH, 0, 1)
    box_thresh = _bounded_float_param(resolved, "box_thresh", DEFAULT_BOX_THRESH, 0, 1)
    unclip_ratio = _bounded_float_param(resolved, "unclip_ratio", DEFAULT_UNCLIP_RATIO, 0, 5)
    max_candidates = _max_candidates(resolved)
    fraction = _bounded_float_param(resolved, "baseline_fraction", DEFAULT_BASELINE_FRACTION, 0, 1)
    direction = _choice_param(
        resolved,
        "reading_direction",
        "ltr",
        ("ltr", "rtl"),
        'reading_direction must be "ltr" or "rtl"',
    )
    box_type = _box_type(resolved, DEFAULT_BOX_TYPE)
    merge_fragments = _bool_param(resolved, "merge_fragments", True)
    resolve_overlaps = _bool_param(resolved, "resolve_overlaps", True)
    noise_policy = _choice_param(
        resolved,
        "noise_policy",
        "flag",
        ("flag", "drop", "off"),
        'noise_policy must be "flag", "drop" or "off"',
    )
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
