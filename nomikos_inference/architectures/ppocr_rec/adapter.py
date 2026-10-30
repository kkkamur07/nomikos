"""PP-OCR recognition transcription on the ONNX Runtime CPU runtime.

The Hub artifact is ``model.onnx``: the graph carries its own codec, line
height, padding, temperature and blank index in ``metadata_props``, so the
runtime needs neither the ``.safetensors`` checkpoint nor a sidecar to decode.
``resolve_artifact`` verifies the digest before the artifact is opened;
``reraise_if_none_survived`` treats an all-failed batch as a failed run rather
than a page of per-line errors.

The graph is the kraken PP-OCR recognition recipe exported to ONNX. Serving
reproduces kraken's inference exactly: the kraken input transforms (RGB,
fixed-height resize, white side padding, scale, invert), ``softmax(logits /
temperature)`` over classes, greedy CTC (argmax per frame, drop blank 0,
collapse repeats), then kraken's own bidi reorder from display order (left to
right on the image) to logical reading order, with the confidences permuted by
the same map.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort

import numpy as np

from nomikos_inference.architectures.artifact import ArtifactHandle, resolve_artifact
from nomikos_inference.architectures.isolation import reraise_if_none_survived
from nomikos_inference.architectures.ppocr_rec.bidi import get_display_map
from nomikos_inference.architectures.ppocr_rec.onnx_session import open_onnx_session
from nomikos_inference.architectures.ppocr_rec.preprocessing import (
    preprocess_line_image_bytes_to_ppocr_rec_tensor,
)
from nomikos_inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse

# The PP-OCR recognition **Hub artifact** is the self-contained ONNX graph.
# There is one runtime format per architecture; ``find_hub_artifact`` enforces
# the same rule on the cache directory so a repo holding both formats cannot
# silently decide which runtime ran.
PPOCR_REC_ARTIFACT_SUFFIXES = frozenset({".onnx"})


class PPOCRRecUnavailableError(RuntimeError):
    """Raised when a PP-OCR recognition runtime artifact cannot be used."""


@dataclass(frozen=True)
class TranscribeLineFailure:
    """One line of a batch that could not be transcribed.

    Returned in place of that line's output instead of raised, so a single
    unusable crop degrades to a per-line error rather than discarding the whole
    page. The original exception rides along because an all-failed batch has to
    re-raise it: the run-error mapping distinguishes a broken artifact (503)
    from a bad request (422), and both would collapse into a generic 500 if the
    cause were flattened to a string here.
    """

    index: int
    error: Exception


@dataclass(frozen=True)
class PPOCRRecGraphConfig:
    """The decode parameters the graph stamps in its ``metadata_props``.

    Metadata is the source of truth: line height, padding, temperature, blank
    index, codec and tensor names all come from the graph, so a republished
    artifact changes serving without a code change.
    """

    charset: tuple[str, ...]
    line_height: int
    pad: int
    pad_fill: int
    temperature: float
    input_name: str
    output_names: tuple[str, ...]


def _metadata_int(metadata: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise PPOCRRecUnavailableError(
            f"PP-OCR recognition ONNX metadata has invalid {key}"
        ) from error
    if value < minimum:
        raise PPOCRRecUnavailableError(f"PP-OCR recognition ONNX metadata has invalid {key}")
    return value


def _temperature_from_metadata(metadata: Mapping[str, str]) -> float:
    # Unlike Calamari, where the exporter bakes the temperature into the
    # graph, the PP-OCR graph emits raw logits and serving applies
    # ``softmax(logits / temperature)`` itself, exactly as kraken's
    # ``_rec_predict`` divides before its softmax.
    try:
        temperature = float(metadata["temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has invalid temperature metadata"
        ) from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has invalid temperature metadata"
        )
    return temperature


def _charset_from_metadata(metadata: Mapping[str, str], classes: int) -> tuple[str, ...]:
    charset_value = metadata.get("charset")
    if charset_value is None:
        raise PPOCRRecUnavailableError("PP-OCR recognition ONNX artifact has no codec metadata")
    charset = json.loads(charset_value)
    if (
        not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
        or charset[0] != ""
    ):
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has invalid codec metadata"
        )
    return tuple(charset)


def _tensor_names_from_metadata(metadata: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    input_name = metadata.get("input_name")
    if not input_name:
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has no input name metadata"
        )
    try:
        output_names = json.loads(metadata.get("output_names", ""))
    except ValueError as error:
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has invalid output names metadata"
        ) from error
    if (
        not isinstance(output_names, list)
        or not output_names
        or not all(isinstance(name, str) for name in output_names)
    ):
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has invalid output names metadata"
        )
    return input_name, tuple(output_names)


def _config_from_metadata(metadata: Mapping[str, str]) -> PPOCRRecGraphConfig:
    if metadata.get("format") != "ppocr-rec-onnx-v1":
        raise PPOCRRecUnavailableError("unsupported PP-OCR recognition ONNX artifact format")
    if metadata.get("blank_index") != "0":
        raise PPOCRRecUnavailableError(
            "PP-OCR recognition ONNX artifact has an unsupported blank index"
        )
    classes = _metadata_int(metadata, "classes", minimum=2)
    line_height = _metadata_int(metadata, "line_height", minimum=1)
    pad = _metadata_int(metadata, "pad", minimum=0)
    pad_fill = _metadata_int(metadata, "pad_fill", minimum=0)
    if pad_fill > 255:
        raise PPOCRRecUnavailableError("PP-OCR recognition ONNX metadata has invalid pad_fill")
    input_name, output_names = _tensor_names_from_metadata(metadata)
    return PPOCRRecGraphConfig(
        charset=_charset_from_metadata(metadata, classes),
        line_height=line_height,
        pad=pad,
        pad_fill=pad_fill,
        temperature=_temperature_from_metadata(metadata),
        input_name=input_name,
        output_names=output_names,
    )


def _check_session_io(session: ort.InferenceSession, config: PPOCRRecGraphConfig) -> None:
    input_names = {input_.name for input_ in session.get_inputs()}
    if config.input_name not in input_names:
        raise PPOCRRecUnavailableError("PP-OCR recognition ONNX artifact has incompatible inputs")
    output_name_set = {output.name for output in session.get_outputs()}
    if not set(config.output_names).issubset(output_name_set):
        raise PPOCRRecUnavailableError("PP-OCR recognition ONNX artifact has incompatible outputs")


@lru_cache(maxsize=4)
def _load_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[ort.InferenceSession, PPOCRRecGraphConfig]:
    """Open a session and read the codec the graph carries with it.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    session = open_onnx_session(
        model_path,
        PPOCRRecUnavailableError,
        "unable to load PP-OCR recognition ONNX artifact",
        "onnxruntime is required for the PP-OCR recognition runtime",
    )
    try:
        config = _config_from_metadata(session.get_modelmeta().custom_metadata_map)
        _check_session_io(session, config)
        return session, config
    except PPOCRRecUnavailableError:
        raise
    except Exception as error:
        raise PPOCRRecUnavailableError("unable to load PP-OCR recognition ONNX artifact") from error


def _decode_greedy(
    softmax: np.ndarray,
    *,
    charset: tuple[str, ...] | list[str],
) -> tuple[str, list[float]]:
    """Collapse one softmax matrix to display-order text and confidences.

    Greedy CTC in display order (left to right on the image): argmax per
    frame, drop blank 0, collapse repeats keeping the max frame confidence per
    emitted character. A grapheme spanning several codepoints emits one
    confidence per codepoint, duplicated, which is what kraken's
    ``PytorchCodec.decode`` does and what the response contract needs (one
    ``CharacterConfidence`` per character of the text).
    """
    labels = np.argmax(softmax, axis=1)
    text_parts: list[str] = []
    confidences: list[float] = []
    last_label = 0

    for index, label in enumerate(labels):
        label = int(label)
        if label == 0:
            last_label = label
            continue
        if label != last_label:
            grapheme = charset[label] if label < len(charset) else ""
            for char in grapheme:
                text_parts.append(char)
                confidences.append(float(softmax[index, label]))
        elif confidences:
            confidences[-1] = max(confidences[-1], float(softmax[index, label]))
        last_label = label

    return "".join(text_parts), confidences


def _reorder_to_logical(text: str, confidences: list[float]) -> tuple[str, list[float]]:
    """Reorder display-order text to logical reading order, kraken's way.

    The network emits characters left to right on the image; kraken's
    ``BaselineOCRRecord.logical_order`` runs that string through its bidi
    ``get_display_map`` (vendored here verbatim) and permutes the confidences
    with the same map. Syriac is right-to-left, so skipping this returns the
    line backwards. ``base_dir`` stays ``None``: kraken resolves the paragraph
    direction from the first strong character.
    """
    logical_text, order = get_display_map(text, None)
    return logical_text, [confidences[index] for index in order]


def _response_from_decoded(text: str, confidences: list[float]) -> TranscribeRunResponse:
    if len(confidences) != len(text):
        confidences = [float(np.mean(confidences)) if confidences else 0.0 for _ in text]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return TranscribeRunResponse(
        text=text,
        confidence=max(0.0, min(1.0, confidence)),
        character_confidences=[
            CharacterConfidence(char=char, confidence=max(0.0, min(1.0, confidence)))
            for char, confidence in zip(text, confidences, strict=True)
        ],
    )


def _reject_fully_failed_batch(
    results: list[TranscribeRunResponse | TranscribeLineFailure],
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    """Let partial results through, but never an all-failed batch.

    Isolating per-line failures is only safe while at least one line survived.
    If none did, the cause is almost certainly the artifact or the runtime, and
    re-raising the first line's original exception keeps its HTTP mapping (503
    for an unusable runtime, 422 for an unusable request) instead of handing the
    caller a page of identical per-line errors that looks like a successful run.

    The rule itself lives in ``architectures.isolation`` because BLLA has to
    reach the same verdict from a differently shaped loop.
    """
    failures = [result for result in results if isinstance(result, TranscribeLineFailure)]
    reraise_if_none_survived(
        survivors=len(results) - len(failures),
        first_failure=failures[0].error if failures else None,
    )
    return results


def run_ppocr_rec_transcribe_many(
    line_images: list[bytes],
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    # Checked before the artifact: an empty batch is a client error (422)
    # regardless of the weights on disk, and running the artifact preflight
    # first would report a missing artifact (503) for a request that was
    # never runnable to begin with. See ``architectures.artifact`` for why its
    # own failures are ordered the same way.
    if not line_images:
        raise ValueError("at least one line image is required")

    handle = resolve_artifact(
        checkpoint_path,
        label="PP-OCR recognition model",
        allowed_suffixes=PPOCR_REC_ARTIFACT_SUFFIXES,
        unusable_error=PPOCRRecUnavailableError,
        unusable_message=(f"PP-OCR recognition runtime requires an .onnx model: {checkpoint_path}"),
        artifact_sha256=artifact_sha256,
    )
    return _reject_fully_failed_batch(_run_onnx_transcribe_many(line_images, handle=handle))


def _run_onnx_transcribe_many(
    line_images: list[bytes],
    *,
    handle: ArtifactHandle,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    session, config = _load_session(handle.path, handle.fingerprint)
    if not config.charset:
        raise PPOCRRecUnavailableError(f"PP-OCR recognition artifact has no codec: {handle.path}")

    responses: list[TranscribeRunResponse | TranscribeLineFailure] = []
    for index, image_bytes in enumerate(line_images):
        # Per-line isolation: the caller decides what a failed line means,
        # and one of them must not end the batch.
        try:
            image = preprocess_line_image_bytes_to_ppocr_rec_tensor(
                image_bytes,
                line_height=config.line_height,
                pad=config.pad,
                pad_fill=config.pad_fill,
            ).astype(np.float32, copy=False)
            outputs = session.run(
                list(config.output_names),
                {config.input_name: image},
            )
            # Softmax in NumPy rather than in the graph: the graph emits raw
            # logits and kraken divides by the temperature before its own
            # softmax, so the division stays visible here.
            logits = np.asarray(outputs[0], dtype=np.float32)[0] / config.temperature
            logits -= np.max(logits, axis=-1, keepdims=True)
            softmax = np.exp(logits)
            softmax /= np.sum(softmax, axis=-1, keepdims=True)
            display_text, display_confidences = _decode_greedy(softmax, charset=config.charset)
            text, confidences = _reorder_to_logical(display_text, display_confidences)
            responses.append(_response_from_decoded(text, confidences))
        except Exception as error:  # noqa: BLE001 - per-line isolation is the point
            responses.append(TranscribeLineFailure(index=index, error=error))
    return responses


def run_ppocr_rec_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> TranscribeRunResponse:
    result = run_ppocr_rec_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )[0]
    if isinstance(result, TranscribeLineFailure):
        # Unreachable: a one-line batch that failed already re-raised above.
        raise result.error
    return result


__all__ = [
    "PPOCRRecUnavailableError",
    "TranscribeLineFailure",
    "run_ppocr_rec_transcribe",
    "run_ppocr_rec_transcribe_many",
]
