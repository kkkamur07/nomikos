"""Which text a line exports as when the export is for reading, not for training.

A reader wants the whole page: the approved (ground truth) text where there is one,
and otherwise the model's output, marked as not approved yet. A page mid-review used
to come out of the Word export partial or empty, which reads as "this page cannot be
downloaded" rather than "this page is not reviewed yet".

This is the rule the transcription PDF applies in ``TranscriptionPdfService._pdf_lines``,
restated once here so the Word export does not depend on the PDF service; the PDF can
switch to it later. The plain text and PAGE XML exports stay approved only (the PAGE XML
is training data), and keep using ``document_export_text._ground_truth_text``.

Pure, like ``regions``: anything shaped like an ORM ``Line`` qualifies.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.document.application.document_export_text import _ground_truth_text
from backend.document.infrastructure.orm_models import Line, LineTranscription, TranscriptionKind


@dataclass(frozen=True)
class LineText:
    """A line's exported text and whether a person approved it."""

    text: str
    approved: bool = True


def reading_text(line: Line) -> LineText | None:
    """The approved text if the line has any, else its newest model text, else None.

    Approved text is taken verbatim by the text export's rule. A ground-truth layer
    holding only whitespace counts as no approved text, as in the PDF, so the model's
    output shows through instead of a blank line.
    """
    approved = _ground_truth_text(line)
    if approved.strip():
        return LineText(approved, approved=True)
    model = _model_text(line)
    if model is not None:
        return LineText(model, approved=False)
    return None


def _model_text(line: Line) -> str | None:
    """The newest non-empty model transcription on ``line``, or None."""
    candidates = [
        transcription
        for transcription in line.transcriptions
        if transcription.transcription.kind == TranscriptionKind.model
        and transcription.text.strip()
    ]
    if not candidates:
        return None
    return max(candidates, key=_layer_age).text.strip("\n")


def _layer_age(transcription: LineTranscription) -> float:
    created_at = transcription.transcription.created_at
    return created_at.timestamp() if created_at is not None else 0.0
