"""Writing ground-truth text onto a line.

Three use cases converge here: the bulk line replace, page-transcription pairing, and
editing a ground-truth line directly. All three need the same two invariants held:

* a document has exactly **one** ground-truth layer, created on first write rather than
  by migration, because documents predate the layer and a document created through the
  repository already carries one;
* ``text=None`` *removes* the line's ground-truth row rather than storing an empty
  string, and any write clears ``confidence`` and ``character_confidences``. A
  confidence score is a claim the model made; once a human has supplied the text
  there is nothing left for it to describe.

Kept out of the modules that use it precisely because all three need it: this is the one
piece of shared behaviour that survived breaking up ``DocumentService``, and it is shared
by injection rather than by inheritance.
"""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Document,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)


class GroundTruthText:
    def __init__(self, documents: DocumentRepository | None = None) -> None:
        self._documents = documents or DocumentRepository()

    async def layer_for(self, session: AsyncSession, document: Document) -> Transcription:
        """The document's ground-truth layer, created on demand.

        Flushed rather than committed: the caller is mid-transaction and owns the commit.
        """
        transcription = await self._documents.get_ground_truth_transcription(session, document.id)
        if transcription is not None:
            return transcription
        transcription = Transcription(
            document_id=document.id,
            name="Ground truth",
            kind=TranscriptionKind.ground_truth,
        )
        # Two first writes for the same document both see no layer and both
        # insert one; the partial unique index ``uq_transcriptions_one_ground_truth``
        # then rejects the loser with IntegrityError. Insert inside a SAVEPOINT so
        # that rejection rolls back only this insert, not the caller's whole
        # transaction, and adopt the winner's row instead of surfacing a 500.
        try:
            async with session.begin_nested():
                session.add(transcription)
                await session.flush()
        except IntegrityError:
            existing = await self._documents.get_ground_truth_transcription(session, document.id)
            if existing is None:
                raise
            return existing
        return transcription

    async def write(
        self,
        session: AsyncSession,
        line: Line,
        ground_truth: Transcription,
        text: str | None,
    ) -> None:
        """Upsert ``text`` on ``line`` in ``ground_truth``; ``None`` removes the row."""
        existing = next(
            (
                transcription
                for transcription in line.transcriptions
                if transcription.transcription_id == ground_truth.id
            ),
            None,
        )
        if text is None:
            if existing is not None:
                line.transcriptions.remove(existing)
                await session.delete(existing)
            return
        if existing is None:
            line.transcriptions.append(
                LineTranscription(
                    transcription=ground_truth,
                    text=text,
                    confidence=None,
                    character_confidences=None,
                )
            )
            return
        existing.text = text
        existing.confidence = None
        existing.character_confidences = None
