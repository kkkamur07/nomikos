"""Shared schemas and helpers."""

from __future__ import annotations

import base64
import binascii
from enum import StrEnum
from typing import Annotated

from pydantic import PlainSerializer, PlainValidator

from nomikos_inference.admission import CLIENT_INPUT_ERROR, validate_encoded_image
from nomikos_inference.settings import get_inference_settings


class InferenceTask(StrEnum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"


class InferenceJobStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RegistryArchitecture(StrEnum):
    calamari = "calamari"
    ppocr_rec = "ppocr_rec"
    blla = "blla"
    blla_segment = "blla-segment"
    ppocr_det = "ppocr-det"


class LineCrop(StrEnum):
    """How a transcribe model's line crops were cut when it was trained.

    ``polygon_white`` is the recipe in
    ``src/preprocessing_data/syriac/xml_to_data.py::crop_polygon``: the polygon's
    bounding box, widened by the model's ``line_crop_padding`` and clamped to the
    page, with every pixel outside the polygon painted white on the crop.

    Every registry model uses it. What differs between them is the padding, not
    the function, which is why that lives in its own registry field rather than
    in another enum value here. Measured 2026-09-07 on the pages each model was
    trained on, same ONNX in every run:

    * ``greek-calamari-v1`` on the whole Grec1360 corpus (204 lines) reads 0/204
      exact at padding 12 (CER 0.304) and 125/204 exact at padding 0 (CER 0.050).
      Its finetuning crops were exported with no padding.
    * ``armenian-calamari-v1`` on the full MS_UCLA_MS document reads 664/819
      exact at padding 12.
    * ``coptic-calamari-v1`` on 2823 platform Coptic project lines through
      ``best.pt`` at batch size 1 reads CER 0.0177 at padding 12 against 0.1890
      at padding 0, minimum flat across 10 to 12, measured 2026-09-18 on the
      platform's Coptic project lines; full curve in
      ``docs/inference/models-and-datasets.md``.

    The enum stays a single-value enum rather than collapsing to a bool because
    the field then still names the convention, and a second convention (should a
    future model arrive with one) is an added member rather than a schema change.
    """

    polygon_white = "polygon-white"


class ComputeDevice(StrEnum):
    cpu = "cpu"
    cuda = "cuda"
    any = "any"


class HostEligibility(StrEnum):
    local = "local"
    remote = "remote"
    any = "any"


# Wire-format bounds for segment and transcribe responses. These mirror
# ``AdmissionSettings.inference_max_geometry_points`` (256) and the platform's
# ``MAX_LINE_GEOMETRY_POINTS`` / ``MAX_LINE_TEXT_CHARS``; the response contract
# enforces what admission promises, so a denser polygon or a longer text can
# never be produced and written back to the platform.
MAX_GEOMETRY_POINTS = 256
# The Kraken ceiling is the raw pre-simplification decoder polygon, legitimately
# denser than the simplified ``points`` (which is clamped to MAX_GEOMETRY_POINTS):
# real lines reach a few hundred vertices. It still needs an upper bound so a
# degenerate or hostile decode cannot ship an unbounded array; this generous cap
# admits real ceilings while rejecting pathological ones.
MAX_KRAKEN_CEILING_POINTS = 4_096
MAX_LINE_TEXT_CHARS = 10_000
MAX_TRANSCRIBE_LINES = 10_000
MAX_SEGMENT_LINES = 10_000
MAX_SEGMENT_BLOCKS = 1_000


def _coerce_image_bytes(value: object) -> bytes:
    settings = get_inference_settings()
    raw_bytes = validate_encoded_image(value, settings)
    if isinstance(value, bytes):
        return raw_bytes

    if isinstance(value, str):
        normalized = "".join(value.split())

        try:
            decoded = base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(CLIENT_INPUT_ERROR) from exc
        if len(decoded) > settings.inference_max_decoded_image_bytes:
            raise ValueError(CLIENT_INPUT_ERROR)
        return decoded

    raise TypeError("image_bytes must be bytes or base64 string")


def _serialize_image_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode()


ImageBytes = Annotated[
    bytes,
    PlainValidator(_coerce_image_bytes),
    PlainSerializer(_serialize_image_bytes),
]

__all__ = [
    "ComputeDevice",
    "HostEligibility",
    "ImageBytes",
    "InferenceJobStatus",
    "InferenceTask",
    "LineCrop",
    "MAX_GEOMETRY_POINTS",
    "MAX_KRAKEN_CEILING_POINTS",
    "MAX_LINE_TEXT_CHARS",
    "MAX_SEGMENT_BLOCKS",
    "MAX_SEGMENT_LINES",
    "MAX_TRANSCRIBE_LINES",
    "RegistryArchitecture",
]
