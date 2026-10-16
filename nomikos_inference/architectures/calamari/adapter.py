"""Calamari transcription on the ONNX Runtime CPU runtime (ADR 0006).

The Hub artifact is ``best.onnx``: the graph carries its own codec, line
height and blank index in ``metadata_props``, so the runtime needs neither the
``.pt`` checkpoint nor a sidecar to decode. ``resolve_artifact`` verifies the
digest before the artifact is opened; ``reraise_if_none_survived`` treats an
all-failed batch as a failed run rather than a page of per-line errors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from nomikos_inference.architectures.artifact import ArtifactHandle, resolve_artifact
from nomikos_inference.architectures.calamari.preprocessing import (
    preprocess_line_image_bytes_to_calamari_tensor,
)
from nomikos_inference.architectures.shared.onnx_session import OnnxSession, open_checked_session
from nomikos_inference.architectures.shared.transcribe import (
    TranscribeLineFailure,
    reject_fully_failed_batch,
    response_from_decoded,
)
from nomikos_inference.contracts.transcribe import TranscribeRunResponse

# The Calamari **Hub artifact** is the self-contained ONNX graph. There is one
# runtime format per architecture; ``find_hub_artifact`` enforces the same rule
# on the cache directory so a repo holding both formats cannot silently decide
# which runtime ran.
CALAMARI_ARTIFACT_SUFFIXES = frozenset({".onnx"})


class CalamariUnavailableError(RuntimeError):
    """Raised when a Calamari runtime artifact cannot be used."""


def _metadata_int(metadata: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as error:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}") from error
    if value < minimum:
        raise CalamariUnavailableError(f"Calamari ONNX metadata has invalid {key}")
    return value


def _checked_temperature(metadata: Mapping[str, str]) -> None:
    """Reject a missing or non-finite temperature.

    The exporter bakes any positive temperature into the graph itself
    (``CalamariTorchModel.forward`` divides the logits before tracing), so the
    session output is already temperature-scaled. The metadata value is
    validated only to reject corrupted artifacts; the runtime must NOT re-apply
    it to the logits.
    """
    try:
        temperature = float(metadata["temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise CalamariUnavailableError(
            "Calamari ONNX artifact has invalid temperature metadata"
        ) from error
    if not math.isfinite(temperature):
        raise CalamariUnavailableError("Calamari ONNX artifact has invalid temperature metadata")


def _checked_charset(metadata: Mapping[str, str], classes: int) -> list[str]:
    charset_value = metadata.get("charset")
    if charset_value is None:
        raise CalamariUnavailableError("Calamari ONNX artifact has no codec metadata")
    charset = json.loads(charset_value)
    if (
        not isinstance(charset, list)
        or len(charset) != classes
        or not all(isinstance(character, str) for character in charset)
    ):
        raise CalamariUnavailableError("Calamari ONNX artifact has invalid codec metadata")
    return charset


def _checked_calamari_session(session: OnnxSession) -> tuple[OnnxSession, list[str], int]:
    """The codec and line height of a graph this runtime can decode."""
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("format") != "calamari-onnx-v1":
        raise CalamariUnavailableError("unsupported Calamari ONNX artifact format")
    classes = _metadata_int(metadata, "classes", minimum=2)
    line_height = _metadata_int(metadata, "line_height", minimum=1)
    if metadata.get("blank_index") != "0":
        raise CalamariUnavailableError("Calamari ONNX artifact has an unsupported blank index")
    _checked_temperature(metadata)
    charset = _checked_charset(metadata, classes)
    if not {"image", "image_lengths"}.issubset({input_.name for input_ in session.get_inputs()}):
        raise CalamariUnavailableError("Calamari ONNX artifact has incompatible inputs")
    if not {"logits", "out_len"}.issubset({output.name for output in session.get_outputs()}):
        raise CalamariUnavailableError("Calamari ONNX artifact has incompatible outputs")
    return session, charset, line_height


@lru_cache(maxsize=4)
def _load_session(
    model_path: str,
    fingerprint: tuple[int, int] | None = None,
) -> tuple[OnnxSession, list[str], int]:
    """Open a session and read the codec the graph carries with it.

    ``fingerprint`` is part of the cache key rather than an argument the loader
    reads: it is what makes a *replaced* artifact file miss the cache instead of
    serving the previous model for the life of the process.
    """
    return open_checked_session(
        model_path,
        unavailable=CalamariUnavailableError,
        missing_runtime_message="onnxruntime is required for the Calamari runtime",
        load_failure_message="unable to load Calamari ONNX artifact",
        check=_checked_calamari_session,
    )


def _decode_greedy(
    softmax: np.ndarray,
    *,
    charset: list[str],
) -> tuple[str, list[float]]:
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
            char = charset[label] if label < len(charset) else ""
            if char:
                text_parts.append(char)
                confidences.append(float(softmax[index, label]))
        elif confidences:
            confidences[-1] = max(confidences[-1], float(softmax[index, label]))
        last_label = label

    # No edge-whitespace trim. The trainer's ``CharacterCodec.decode_ctc`` in
    # ``src/models/calamari/codec.py`` returns the collapsed labels as they are,
    # and its loader reads ``.gt.txt`` files without stripping, so a model that
    # learned to emit a leading or trailing space emits it on both sides. The
    # trim that used to live here was the last thing keeping serving from being
    # byte-identical to the trainer (see ADR 0007).
    return "".join(text_parts), confidences


def run_calamari_transcribe_many(
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
        label="Calamari model",
        allowed_suffixes=CALAMARI_ARTIFACT_SUFFIXES,
        unusable_error=CalamariUnavailableError,
        unusable_message=(f"Calamari runtime requires an .onnx model: {checkpoint_path}"),
        artifact_sha256=artifact_sha256,
    )
    return reject_fully_failed_batch(_run_onnx_transcribe_many(line_images, handle=handle))


def _run_onnx_transcribe_many(
    line_images: list[bytes],
    *,
    handle: ArtifactHandle,
) -> list[TranscribeRunResponse | TranscribeLineFailure]:
    session, charset, line_height = _load_session(handle.path, handle.fingerprint)
    if not charset:
        raise CalamariUnavailableError(f"Calamari artifact has no codec metadata: {handle.path}")

    responses: list[TranscribeRunResponse | TranscribeLineFailure] = []
    for index, image_bytes in enumerate(line_images):
        # Per-line isolation: the caller decides what a failed line means,
        # and one of them must not end the batch.
        try:
            image = preprocess_line_image_bytes_to_calamari_tensor(
                image_bytes,
                line_height=line_height,
            ).astype(np.float32, copy=False)
            outputs = session.run(
                ["logits", "out_len"],
                {
                    "image": image,
                    "image_lengths": np.asarray([image.shape[1]], dtype=np.int64),
                },
            )
            # Softmax in NumPy rather than in the graph: the exporter traces the
            # logits so the temperature it baked in stays visible to anything
            # comparing against the reference forward.
            logits = np.asarray(outputs[0], dtype=np.float32)[0]
            logits -= np.max(logits, axis=-1, keepdims=True)
            softmax = np.exp(logits)
            softmax /= np.sum(softmax, axis=-1, keepdims=True)
            text, confidences = _decode_greedy(softmax, charset=charset)
            responses.append(response_from_decoded(text, confidences))
        except Exception as error:  # noqa: BLE001 - per-line isolation is the point
            responses.append(TranscribeLineFailure(index=index, error=error))
    return responses


def run_calamari_transcribe(
    image_bytes: bytes,
    *,
    checkpoint_path: Path,
    artifact_sha256: str | None = None,
) -> TranscribeRunResponse:
    result = run_calamari_transcribe_many(
        [image_bytes],
        checkpoint_path=checkpoint_path,
        artifact_sha256=artifact_sha256,
    )[0]
    if isinstance(result, TranscribeLineFailure):
        # Unreachable: a one-line batch that failed already re-raised above.
        raise result.error
    return result


__all__ = [
    "CalamariUnavailableError",
    "TranscribeLineFailure",
    "run_calamari_transcribe",
    "run_calamari_transcribe_many",
]
