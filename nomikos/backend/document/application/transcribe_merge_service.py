"""Transcribe merge - persist model transcription layer from ML output."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from nomikos_inference.contracts.transcribe import TranscribeRunResponse

logger = logging.getLogger(__name__)


class TranscribeJobHandlerError(Exception):
    """Raised for user-actionable transcribe job failures."""


def character_confidences_for_storage(output: TranscribeRunResponse) -> list[float] | None:
    """Per-code-point scores for ``output.text``, rounded for storage.

    Returns null when the scores do not describe the text (length mismatch):
    a row must never carry scores aligned to some other string. The caller logs
    that case; it is the worker's data, not the job's failure.
    """
    scores = [round(entry.confidence, 3) for entry in output.character_confidences]
    if len(scores) != len(output.text):
        return None
    return scores


class TranscribeMergeService:
    """Create a model transcription layer from per-line ML transcribe output."""

    @staticmethod
    def load_lines(session: Session, part_id: UUID) -> list[Line]:
        lines = list(
            session.execute(
                select(Line).where(Line.part_id == part_id).order_by(Line.order, Line.created_at)
            )
            .scalars()
            .all()
        )
        if not lines:
            raise TranscribeJobHandlerError(
                "Cannot transcribe a document part without layout lines"
            )
        return lines

    @staticmethod
    def lock_part_sync(session: Session, *, document_id: UUID, part_id: UUID) -> DocumentPart:
        """Hold one part still before reading the lines a callback is about to write.

        Take this before the caller reads its lines, not after. Segment health can
        delete a line between an unlocked read and this lock, leaving the callback
        holding a stale ``Line`` whose row is gone; the insert then fails on the
        foreign key and takes the whole model output down with it.
        ``segment_merge_service._load_part`` locks in the same order for the same
        reason.
        """
        part = session.execute(
            select(DocumentPart).where(DocumentPart.id == part_id).with_for_update()
        ).scalar_one_or_none()
        if part is None or part.document_id != document_id:
            raise TranscribeJobHandlerError("Document part not found")
        return part

    def apply_sync(
        self,
        session: Session,
        *,
        document_id: UUID,
        part_id: UUID,
        job_id: UUID | None,
        lines_with_output: list[tuple[Line, TranscribeRunResponse]],
        commit: bool = True,
        layer_name: str | None = None,
    ) -> dict:
        # Load the part locked, not with ``session.get``. Segment health decides a
        # merge or a deletion from a snapshot of which lines carry text, and
        # ``_has_text`` counts every layer, model output included: an unapproved
        # draft is the thing a reviewer is about to correct, which is why
        # ``test_an_unapproved_model_prediction_still_counts_as_text`` exists. A
        # layer committed inside that snapshot's window is therefore attached to a
        # line the apply is about to delete, and the draft goes with it. Holding
        # the part until this commits makes the two wait for each other.
        self.lock_part_sync(session, document_id=document_id, part_id=part_id)

        resolved_name = layer_name or (
            f"Model transcription {job_id.hex[:8]}"
            if job_id is not None
            else "Local model transcription"
        )
        layer = Transcription(
            document_id=document_id,
            name=resolved_name,
            kind=TranscriptionKind.model,
            created_by_job_id=job_id,
        )
        session.add(layer)
        session.flush()

        result_lines: list[dict] = []
        for line, output in lines_with_output:
            scores = character_confidences_for_storage(output)
            if scores is None:
                logger.warning(
                    "transcribe_merge_character_confidences_dropped line_id=%s text_chars=%d scores=%d",
                    line.id,
                    len(output.text),
                    len(output.character_confidences),
                )
            session.add(
                LineTranscription(
                    line_id=line.id,
                    transcription_id=layer.id,
                    text=output.text,
                    confidence=output.confidence,
                    character_confidences=scores,
                )
            )
            result_lines.append(
                {
                    "line_id": str(line.id),
                    "text": output.text,
                    "confidence": output.confidence,
                }
            )

        if commit:
            session.commit()
        return {
            "transcription_id": str(layer.id),
            "lines": result_lines,
        }
