"""Model execution for the synchronous run path."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from nomikos_inference.admission import validate_image_bytes, validate_request_params
from nomikos_inference.architectures.blla import run_blla_segment
from nomikos_inference.architectures.calamari import (
    run_calamari_transcribe,
    run_calamari_transcribe_many,
)
from nomikos_inference.architectures.calamari.preprocessing import (
    TRAINING_CROP_PADDING,
    Page,
    crop_line,
    crop_line_rgb,
    page_pixels,
)
from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment
from nomikos_inference.architectures.ppocr_rec.adapter import (
    run_ppocr_rec_transcribe,
    run_ppocr_rec_transcribe_many,
)
from nomikos_inference.contracts.common import InferenceTask, LineCrop, RegistryArchitecture
from nomikos_inference.contracts.segment import SegmentRunResponse
from nomikos_inference.contracts.transcribe import (
    TRANSCRIBE_LINE_ERROR,
    TranscribeBatchLineResult,
    TranscribeBatchRunResponse,
    TranscribeLineRegion,
    TranscribeRunResponse,
)
from nomikos_inference.registry.resolve import resolve_registry_entry
from nomikos_inference.settings import get_inference_settings
from nomikos_inference.weights import resolve_weights_source

logger = logging.getLogger(__name__)


#: ``save_crop`` in ``src/preprocessing_data/syriac/xml_to_data.py`` wrote every
#: training crop as a grayscale JPEG with exactly these options, and the
#: trainer's loader read that file back. The lossy round trip is therefore part
#: of the picture the model was fitted on, not an artifact of storage, so serving
#: reproduces it rather than handing over a cleaner PNG.
TRAINING_CROP_JPEG_QUALITY = 82


class _LineFailure(Protocol):
    """A batch line an adapter could not transcribe, with its cause."""

    @property
    def error(self) -> BaseException: ...


class _LineCropper(Protocol):
    def __call__(
        self,
        image: Page,
        image_bytes: bytes,
        points: list[list[float]] | None,
        line_crop: LineCrop = ...,
        line_crop_padding: int = ...,
    ) -> bytes: ...


class _RunOne(Protocol):
    def __call__(
        self, image_bytes: bytes, *, checkpoint_path: Path, artifact_sha256: str | None = None
    ) -> TranscribeRunResponse: ...


class _RunMany(Protocol):
    def __call__(
        self,
        line_images: list[bytes],
        *,
        checkpoint_path: Path,
        artifact_sha256: str | None = None,
    ) -> Sequence[TranscribeRunResponse | _LineFailure]: ...


def _crop_line_image(
    image: Page,
    image_bytes: bytes,
    points: list[list[float]] | None,
    line_crop: LineCrop = LineCrop.polygon_white,
    line_crop_padding: int = TRAINING_CROP_PADDING,
) -> bytes:
    """Cut one line out of a page the way *this model's* training crops were cut.

    One crop function serves every model: the polygon's box widened by
    ``line_crop_padding``, clamped to the page, with everything outside the
    polygon painted white, which is what
    ``src/preprocessing_data/syriac/xml_to_data.py::crop_polygon`` wrote. Only
    the padding differs, and it differs enough to matter: Greek was exported at 0
    and reads 125 of its 204 corpus lines exactly there against 0 of 204 at 12.
    So the registry states both per model and they are threaded down to here.

    Takes the open ``image`` (or its RGB array, see ``page_pixels``) so a page
    with N lines is decoded and converted once, not N times:
    ``Image.crop`` forces a full decode of the source on every call, so re-opening
    the multi-megapixel scan per line was O(N) full decodes of the same bytes.
    ``image_bytes`` is still returned verbatim for the whole-page fallback so the
    downstream model sees the original encoding, not a re-encode.

    The crop goes on as a mode ``L`` JPEG at the exporter's own quality, not as a
    lossless PNG: the model was trained on crops that had been through that
    encoder, so a clean PNG would hand it pixels slightly unlike any it ever saw.
    A ``ValueError`` from the crop (degenerate geometry off the page) propagates:
    ``_transcribe_batch`` isolates it to its own line.
    """
    if not points:
        return image_bytes

    polygon = [point for point in points if len(point) == 2]
    if not polygon:
        return image_bytes

    cropped = crop_line(image, polygon, line_crop, padding=line_crop_padding)
    output = BytesIO()
    Image.fromarray(cropped, mode="L").save(
        output, format="JPEG", quality=TRAINING_CROP_JPEG_QUALITY, optimize=True
    )
    return output.getvalue()


def _crop_ppocr_rec_line_image(
    image: Page,
    image_bytes: bytes,
    points: list[list[float]] | None,
    line_crop: LineCrop = LineCrop.polygon_white,
    line_crop_padding: int = 0,
) -> bytes:
    """Cut one line for the PP-OCR recognizer: RGB, lossless.

    The kraken model was trained on ``extract_polygons`` crops of the RGB page
    with no JPEG round trip (docs/inference/ppocr-rec-onnx-parity-2026-09-20.md),
    so the Calamari exporter's grayscale JPEG would hand it pixels it never saw.
    The geometry is the registry's, as for Calamari; only the colour and the
    encoding differ. The recognizer's own preprocessing pads and resizes.
    """
    if not points:
        return image_bytes

    polygon = [point for point in points if len(point) == 2]
    if not polygon:
        return image_bytes

    cropped = crop_line_rgb(image, polygon, line_crop, padding=line_crop_padding)
    output = BytesIO()
    Image.fromarray(cropped, mode="RGB").save(output, format="PNG")
    return output.getvalue()


@dataclass(frozen=True)
class _Transcriber:
    """One transcribe architecture's entry points and line crop."""

    run_one: _RunOne
    run_many: _RunMany
    crop: _LineCropper


def _transcriber_for(architecture: RegistryArchitecture) -> _Transcriber | None:
    # Built per call from the module globals, not as a module-level table, so
    # tests can stub the entry points: a table would bind the originals at
    # import time and ignore the stub.
    if architecture == RegistryArchitecture.calamari:
        return _Transcriber(run_calamari_transcribe, run_calamari_transcribe_many, _crop_line_image)
    if architecture == RegistryArchitecture.ppocr_rec:
        return _Transcriber(
            run_ppocr_rec_transcribe, run_ppocr_rec_transcribe_many, _crop_ppocr_rec_line_image
        )
    return None


_Segmenter = Callable[..., SegmentRunResponse]


def _segmenter_for(architecture: RegistryArchitecture) -> _Segmenter | None:
    if architecture in {RegistryArchitecture.blla, RegistryArchitecture.blla_segment}:
        return run_blla_segment
    if architecture == RegistryArchitecture.ppocr_det:
        return run_ppocr_det_segment
    return None


def _line_regions_from_params(params: dict[str, object] | None) -> list[TranscribeLineRegion]:
    raw_lines = (params or {}).get("lines")
    if raw_lines is None:
        return []
    if not isinstance(raw_lines, list):
        raise ValueError("transcribe params.lines must be a list")
    return [TranscribeLineRegion.model_validate(line) for line in raw_lines]


def _transcribe_batch(
    image_bytes: bytes,
    line_regions: list[TranscribeLineRegion],
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None,
    line_crop: LineCrop,
    line_crop_padding: int,
    run_many: _RunMany,
    crop: _LineCropper,
) -> TranscribeBatchRunResponse:
    """Transcribe every requested line, keeping per-line failures per-line.

    A malformed polygon or an undecodable crop costs its own line only: the
    other lines of the page still come back with their text, and the bad one
    carries ``error`` instead of ``output``. The failure is logged here rather
    than deeper down because this is the only layer that knows which document
    line the caller meant.
    """
    crops: list[bytes] = []
    cropped_positions: list[int] = []
    errors: dict[int, str] = {}

    # Decode and convert the page once and crop every line from it. Re-opening
    # the scan per line re-decoded the whole multi-megapixel image N times for
    # an N-line page, and converting it per line copied it N times.
    with Image.open(BytesIO(image_bytes)) as page_image:
        page_image.load()
        pixels: np.ndarray = page_pixels(page_image)
        for position, region in enumerate(line_regions):
            try:
                line_image = crop(
                    pixels,
                    image_bytes,
                    region.points,
                    line_crop=line_crop,
                    line_crop_padding=line_crop_padding,
                )
            except Exception as error:  # noqa: BLE001 - one bad region is not a bad page
                logger.warning(
                    "transcribe line crop failed (line_index=%s, line_id=%s)",
                    region.line_index,
                    region.line_id,
                    exc_info=error,
                )
                errors[position] = TRANSCRIBE_LINE_ERROR
                continue
            crops.append(line_image)
            cropped_positions.append(position)

    if not crops:
        # No line even reached the model. Nothing here is worth returning as a
        # partial success, and the cause is the caller's geometry.
        raise ValueError("no transcribable line regions in request")

    outcomes = run_many(
        crops,
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )

    outputs: dict[int, TranscribeRunResponse] = {}
    for position, outcome in zip(cropped_positions, outcomes, strict=True):
        region = line_regions[position]
        if isinstance(outcome, TranscribeRunResponse):
            outputs[position] = outcome
            continue
        logger.warning(
            "transcribe line failed (line_index=%s, line_id=%s)",
            region.line_index,
            region.line_id,
            exc_info=outcome.error,
        )
        errors[position] = TRANSCRIBE_LINE_ERROR

    # Both adapters' ``run_*_transcribe_many`` re-raise when every line they
    # were given failed, so at least one output survives here and the response
    # can never be an all-error batch dressed up as a success.
    return TranscribeBatchRunResponse(
        lines=[
            TranscribeBatchLineResult(
                line_id=region.line_id,
                line_index=region.line_index,
                output=outputs.get(position),
                error=errors.get(position),
            )
            for position, region in enumerate(line_regions)
        ]
    )


def _crop_settings(entry_line_crop: LineCrop | None, padding: int | None) -> tuple[LineCrop, int]:
    """The registry's crop settings, which every transcribe entry must carry.

    The loader requires both; an entry built outside it with either missing
    has no safe default, since the padding alone moves Greek from CER 0.050 to
    0.304 (``registry/__init__.py``).
    """
    if entry_line_crop is None or padding is None:
        raise ValueError("transcribe entry must set line_crop and line_crop_padding")
    return entry_line_crop, padding


def _run_transcribe(
    transcriber: _Transcriber,
    image_bytes: bytes,
    params: dict[str, object] | None,
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None,
    line_crop: LineCrop | None,
    line_crop_padding: int | None,
) -> TranscribeRunResponse | TranscribeBatchRunResponse:
    line_regions = _line_regions_from_params(params)
    if not line_regions:
        return transcriber.run_one(
            image_bytes, checkpoint_path=checkpoint_path, artifact_sha256=artifact_sha256
        )
    crop_kind, padding = _crop_settings(line_crop, line_crop_padding)
    return _transcribe_batch(
        image_bytes,
        line_regions,
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
        line_crop=crop_kind,
        line_crop_padding=padding,
        run_many=transcriber.run_many,
        crop=transcriber.crop,
    )


def run_model(
    *,
    task: InferenceTask,
    registry_model_id: str,
    registry_tag: str,
    image_bytes: bytes,
    params: dict[str, object] | None = None,
) -> SegmentRunResponse | TranscribeRunResponse | TranscribeBatchRunResponse:
    settings = get_inference_settings()
    validate_image_bytes(image_bytes, settings)
    validate_request_params(params or {}, settings)
    entry = resolve_registry_entry(
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        task=task,
        registry_path=settings.inference_registry_path,
    )

    version = entry.versions[registry_tag]
    weights_path = resolve_weights_source(
        version.weights_source,
        registry_model_id=registry_model_id,
        registry_tag=registry_tag,
        hub_revision=version.hub_revision,
        artifact_sha256=version.artifact_sha256,
        architecture=entry.architecture.value,
    )

    if task == InferenceTask.segment:
        segmenter = _segmenter_for(entry.architecture)
        if segmenter is None:
            raise ValueError(f"unsupported segment architecture: {entry.architecture.value}")
        return segmenter(
            image_bytes,
            model_path=weights_path,
            artifact_sha256=version.artifact_sha256,
            params=params,
        )

    if task == InferenceTask.transcribe:
        transcriber = _transcriber_for(entry.architecture)
        if transcriber is None:
            raise ValueError(f"unsupported transcribe architecture: {entry.architecture.value}")
        return _run_transcribe(
            transcriber,
            image_bytes,
            params,
            checkpoint_path=weights_path,
            artifact_sha256=version.artifact_sha256,
            line_crop=entry.line_crop,
            line_crop_padding=entry.line_crop_padding,
        )

    raise ValueError(f"unsupported ML task for runner: {task.value}")
