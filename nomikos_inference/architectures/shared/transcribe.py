"""Batch isolation and response shaping shared by the transcribe adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nomikos_inference.architectures.isolation import reraise_if_none_survived
from nomikos_inference.contracts.transcribe import CharacterConfidence, TranscribeRunResponse


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


def response_from_decoded(text: str, confidences: list[float]) -> TranscribeRunResponse:
    """One response per decoded line, confidences clamped to [0, 1].

    When the decoder's confidences do not line up with the characters, every
    character carries their mean instead.
    """
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


def reject_fully_failed_batch(
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


__all__ = [
    "TranscribeLineFailure",
    "reject_fully_failed_batch",
    "response_from_decoded",
]
