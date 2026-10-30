"""Shared admission limits and validation for untrusted inference requests."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from pydantic import Field
from pydantic_settings import BaseSettings

CLIENT_INPUT_ERROR = "Invalid inference request"

# Upper bounds for the BLLA segmentation knobs. They cannot live in the platform
# API alone: ``run_model`` is reached from both sides of a **claim** - the
# platform's own submission path and an **inference agent** running the model on
# a laptop - so the seam that bounds a value has to sit next to the runtime, not
# next to one caller. Each bound is the value at which the parameter stops
# meaning anything. The lower bound is exclusive: non-positive values are
# rejected, and an absent key means "use the default".
SEGMENT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # Decoder emission gate. The decoder requires a threshold strictly between
    # zero and one, and the runtime clamps to 0.99, so the admission ceiling
    # matches that clamp rather than leaving a caller to trigger the decoder's
    # own ``ValueError``.
    "heatmap_threshold": (0.0, 0.99),
}


class AdmissionSettings(BaseSettings):
    """Environment-configurable, process-local inference admission limits."""

    inference_max_encoded_image_bytes: int = Field(
        default=160 * 1024 * 1024, ge=1024, alias="INFERENCE_MAX_ENCODED_IMAGE_BYTES"
    )
    inference_max_decoded_image_bytes: int = Field(
        default=100 * 1024 * 1024, ge=1024, alias="INFERENCE_MAX_DECODED_IMAGE_BYTES"
    )
    inference_max_image_pixels: int = Field(
        default=200_000_000, ge=1, alias="INFERENCE_MAX_IMAGE_PIXELS"
    )
    inference_allowed_image_formats: str = Field(
        default="JPEG,PNG,TIFF,WEBP", alias="INFERENCE_ALLOWED_IMAGE_FORMATS"
    )
    inference_max_params_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1024,
        alias="INFERENCE_MAX_PARAMS_BYTES",
    )
    inference_max_params_depth: int = Field(default=8, ge=1, alias="INFERENCE_MAX_PARAMS_DEPTH")
    inference_max_params_items: int = Field(
        # 10,000 lines with 256 two-coordinate points require roughly 7.7M
        # visited values once their nested lists are counted.
        default=8_000_000,
        ge=1,
        alias="INFERENCE_MAX_PARAMS_ITEMS",
    )
    inference_max_transcribe_lines: int = Field(
        default=10_000,
        ge=1,
        alias="INFERENCE_MAX_TRANSCRIBE_LINES",
    )
    inference_max_geometry_points: int = Field(
        default=256, ge=2, alias="INFERENCE_MAX_GEOMETRY_POINTS"
    )
    inference_max_job_payload_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1024,
        alias="INFERENCE_MAX_JOB_PAYLOAD_BYTES",
    )


def allowed_image_formats(settings: AdmissionSettings) -> frozenset[str]:
    formats = frozenset(
        value.strip().upper()
        for value in settings.inference_allowed_image_formats.split(",")
        if value.strip()
    )
    if not formats:
        raise ValueError(CLIENT_INPUT_ERROR)
    return formats


def validate_encoded_image(value: object, settings: AdmissionSettings) -> bytes:
    """Reject oversized base64 before allocating decoded image bytes."""
    if isinstance(value, bytes):
        if len(value) > settings.inference_max_decoded_image_bytes:
            raise ValueError(CLIENT_INPUT_ERROR)
        return value

    if not isinstance(value, str):
        raise TypeError("image_bytes must be bytes or base64 string")

    if len(value) > settings.inference_max_encoded_image_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)

    encoded_length = sum(not character.isspace() for character in value)
    if encoded_length > settings.inference_max_encoded_image_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)
    # Base64 decoding cannot produce more than three bytes per four characters.
    if (encoded_length // 4) * 3 > settings.inference_max_decoded_image_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)
    return b""


def validate_image_bytes(image_bytes: bytes, settings: AdmissionSettings) -> None:
    """Validate image headers with Pillow before model loading or array conversion."""
    if len(image_bytes) > settings.inference_max_decoded_image_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = (image.format or "").upper()
                if image_format not in allowed_image_formats(settings):
                    raise ValueError(CLIENT_INPUT_ERROR)
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > settings.inference_max_image_pixels
                ):
                    raise ValueError(CLIENT_INPUT_ERROR)
                image.verify()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
    ):
        raise ValueError(CLIENT_INPUT_ERROR) from None


def open_image_bytes(image_bytes: bytes) -> Image.Image:
    """Open encoded image bytes with the decompression-bomb warning promoted.

    ``validate_image_bytes`` is the admission gate every entry point passes
    through, but architecture paths reopen the same bytes (line crops, the BLLA
    page) and should still refuse a decompression bomb if one is ever reached
    without it. Returns the lazily-open image; the caller closes it.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            return Image.open(BytesIO(image_bytes))
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(CLIENT_INPUT_ERROR) from exc


def _validate_param_leaf(current: object) -> None:
    # Two ways a leaf fails, both of them "this cannot be serialised":
    # a type the wire format does not carry, or a float JSON cannot
    # represent (NaN, +-Infinity).
    unsupported_type = not isinstance(current, str | int | float | bool | type(None))
    non_finite_float = isinstance(current, float) and not math.isfinite(current)
    if unsupported_type or non_finite_float:
        raise ValueError(CLIENT_INPUT_ERROR)


def _enqueue_param_children(current: object, depth: int, pending: list[tuple[object, int]]) -> None:
    if isinstance(current, Mapping):
        for key, nested in current.items():
            if not isinstance(key, str):
                raise ValueError(CLIENT_INPUT_ERROR)
            pending.append((nested, depth + 1))
    elif isinstance(current, list):
        for nested in current:
            pending.append((nested, depth + 1))
    else:
        _validate_param_leaf(current)


def _validate_param_structure(value: object, settings: AdmissionSettings) -> int:
    item_count = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > settings.inference_max_params_depth:
            raise ValueError(CLIENT_INPUT_ERROR)
        item_count += 1
        if item_count > settings.inference_max_params_items:
            raise ValueError(CLIENT_INPUT_ERROR)
        _enqueue_param_children(current, depth, pending)
    return item_count


def serialized_params_size(params: Mapping[str, object], settings: AdmissionSettings) -> int:
    _validate_param_structure(params, settings)
    try:
        size = len(
            json.dumps(
                params,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
    except (TypeError, ValueError):
        raise ValueError(CLIENT_INPUT_ERROR) from None
    if size > settings.inference_max_params_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)
    return size


def validate_transcribe_params(params: Mapping[str, object], settings: AdmissionSettings) -> None:
    """Bound line-region input before the runner crops and fans out work."""
    lines = params.get("lines")
    if lines is None:
        return
    if not isinstance(lines, list) or len(lines) > settings.inference_max_transcribe_lines:
        raise ValueError(CLIENT_INPUT_ERROR)
    for line in lines:
        if not isinstance(line, dict):
            raise ValueError(CLIENT_INPUT_ERROR)
        points = line.get("points")
        if points is not None and (
            not isinstance(points, list) or len(points) > settings.inference_max_geometry_points
        ):
            raise ValueError(CLIENT_INPUT_ERROR)


def _validate_bounded_param(
    params: Mapping[str, object],
    key: str,
    *,
    maximum: float,
    minimum: float = 0.0,
) -> None:
    if key not in params:
        return
    raw = params[key]
    if not isinstance(raw, (str, int, float)):
        return
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        # Unparseable values already fall back to the runtime default (see
        # ``blla_runtime._positive_float_param``); only the numeric range is this
        # seam's business, so leave the existing tolerance alone.
        return
    if not math.isfinite(parsed) or parsed <= minimum or parsed > maximum:
        raise ValueError(CLIENT_INPUT_ERROR)


def validate_segment_params(params: Mapping[str, object]) -> None:
    """Bound the BLLA segmentation knobs before they reach the decoder."""
    for key, (minimum, maximum) in SEGMENT_PARAM_BOUNDS.items():
        _validate_bounded_param(params, key, minimum=minimum, maximum=maximum)


def validate_request_params(params: Mapping[str, object], settings: AdmissionSettings) -> int:
    size = serialized_params_size(params, settings)
    validate_transcribe_params(params, settings)
    validate_segment_params(params)
    return size


def validate_job_payload(
    image_bytes: bytes, params: Mapping[str, object], settings: AdmissionSettings
) -> None:
    params_size = validate_request_params(params, settings)
    if len(image_bytes) + params_size > settings.inference_max_job_payload_bytes:
        raise ValueError(CLIENT_INPUT_ERROR)
