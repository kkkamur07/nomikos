"""The text of a page: imported page transcriptions, pairing, and ground truth.

One responsibility: what a line *says*, as opposed to where it is. Three routes into the
same place, and they are one module because they all write the same rows and have to agree
about them:

* ``import_page_transcription`` loads a plain-text page and drops the pairings it
  invalidates, clearing the ground-truth text those pairings had produced;
* ``pair_page_text_line`` binds one imported text line to one drawn segment, which is the
  human decision that turns text into ground truth (see ``docs/architecture.md``);
* ``copy_to_ground_truth`` and ``patch_ground_truth_line_text`` do the same from the
  model-layer and single-line-edit directions.

Splitting these would leave the "unpair means un-write the ground truth" rule spread
across modules that would each have to remember it.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.application.ground_truth import GroundTruthText
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

MAX_PAGE_TRANSCRIPTION_LINES = 10_000


class TranscriptionService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
        ground_truth: GroundTruthText | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)
        self._ground_truth = ground_truth or GroundTruthText(documents=self._documents)

    # --- Page transcription import and pairing ---

    async def import_page_transcription(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        text: str,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # Reimport clears every pairing on the page, so it is the same race as
        # `pair_page_text_line` with the sign flipped.
        await self._documents.lock_part(session, part_id)
        part = context.part
        text_lines = self._split_page_transcription(text)
        if len(text_lines) > MAX_PAGE_TRANSCRIPTION_LINES:
            raise ValidationError(
                f"Page transcription cannot exceed {MAX_PAGE_TRANSCRIPTION_LINES} non-empty lines"
            )
        existing = await self._documents.list_page_transcription_lines(session, part.id)
        paired_line_ids = {
            text_line.paired_line_id for text_line in existing if text_line.paired_line_id
        }
        if paired_line_ids:
            ground_truth = await self._documents.get_ground_truth_transcription(
                session, context.document.id
            )
            if ground_truth is not None:
                for paired_line_id in paired_line_ids:
                    paired_line = await self._line_or_404(session, part.id, paired_line_id)
                    await self._ground_truth.write(session, paired_line, ground_truth, None)
        for text_line in existing:
            await session.delete(text_line)
        await session.flush()
        for order, line_text in enumerate(text_lines):
            session.add(PageTranscriptionLine(part_id=part.id, order=order, text=line_text))
        await session.commit()
        return await self.get_page_pairing(session, user, project_id, document_id, part_id)

    async def get_page_pairing(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        part = context.part
        text_lines = await self._documents.list_page_transcription_lines(session, part.id)
        total_lines = await self._documents.count_part_lines(session, part.id)
        paired_lines = await self._documents.count_paired_ground_truth_lines(session, part.id)
        percent = round((paired_lines / total_lines) * 100) if total_lines else 0
        return text_lines, {
            "paired_lines": paired_lines,
            "total_lines": total_lines,
            "percent": percent,
        }

    async def pair_page_text_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_id: UUID,
        text_line_order: int,
    ) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # Segment health derives a delete from a snapshot of which lines carry
        # human work, and `page_transcription_lines.paired_line_id` is ON DELETE
        # SET NULL: a pairing committed inside that snapshot's window is removed
        # by the delete without an error, and nothing records that it existed.
        # Holding the part makes the two wait for each other.
        await self._documents.lock_part(session, part_id)
        part = context.part
        line = await self._line_or_404(session, part.id, line_id)
        text_line = await self._page_transcription_line_or_404(session, part.id, text_line_order)
        ground_truth = await self._ground_truth.layer_for(session, context.document)

        previous_paired_line_id = text_line.paired_line_id
        if previous_paired_line_id is not None and previous_paired_line_id != line.id:
            previous_line = await self._line_or_404(session, part.id, previous_paired_line_id)
            await self._ground_truth.write(session, previous_line, ground_truth, None)
        for candidate in await self._documents.list_page_transcription_lines(session, part.id):
            if candidate.paired_line_id == line.id:
                candidate.paired_line_id = None
        text_line.paired_line_id = line.id
        await self._ground_truth.write(session, line, ground_truth, text_line.text)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise ConflictError("This segment is already paired to another text line") from exc
        return await self.get_page_pairing(session, user, project_id, document_id, part_id)

    # --- Ground truth layers ---

    async def copy_to_ground_truth(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        source_transcription_id: UUID,
        *,
        line_ids: list[UUID] | None = None,
    ) -> list[UUID]:
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        source = await self._transcription_or_404(session, document, source_transcription_id)
        if source.kind != TranscriptionKind.model:
            raise ConflictError("Copy to ground truth requires a model transcription layer")

        ground_truth = await self._ground_truth.layer_for(session, document)

        # This route is addressed by document, so it can touch every part in
        # one transaction. Take all of their locks up front and in one ordered
        # statement (see ``lock_parts``) rather than as each part comes up:
        # locks acquired mid-loop are the ones that deadlock. The set is read
        # before it is locked, so a line added to a part not in it during that
        # instant is not covered; the source layer's line set is fixed by this
        # point, so that needs a concurrent OCR write to happen at all.
        affected_parts = (
            select(DocumentPart.id)
            .join(Line, Line.part_id == DocumentPart.id)
            .join(LineTranscription, LineTranscription.line_id == Line.id)
            .where(
                LineTranscription.transcription_id == source.id,
                DocumentPart.document_id == document.id,
            )
            .distinct()
        )
        if line_ids is not None:
            affected_parts = affected_parts.where(LineTranscription.line_id.in_(line_ids))
        await self._documents.lock_parts(
            session, (await session.execute(affected_parts)).scalars().all()
        )

        stmt = (
            select(LineTranscription)
            .join(Line, LineTranscription.line_id == Line.id)
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(
                LineTranscription.transcription_id == source.id,
                DocumentPart.document_id == document.id,
            )
            .order_by(Line.order, Line.created_at)
        )
        if line_ids is not None:
            stmt = stmt.where(LineTranscription.line_id.in_(line_ids))
        result = await session.execute(stmt)
        source_rows = list(result.scalars().all())
        copied_line_ids = [row.line_id for row in source_rows]

        existing_by_line: dict[UUID, LineTranscription] = {}
        if copied_line_ids:
            existing_result = await session.execute(
                select(LineTranscription).where(
                    LineTranscription.transcription_id == ground_truth.id,
                    LineTranscription.line_id.in_(copied_line_ids),
                )
            )
            existing_by_line = {row.line_id: row for row in existing_result.scalars().all()}

        # ``seeded_from_transcription_id`` is written on both branches and is the
        # only record that this ground truth started as a model's guess. Later human
        # edits go through ``GroundTruthText.write`` and ``patch_ground_truth_line_text``,
        # which touch ``text`` and ``confidence`` and nothing else, so the note
        # survives the corrections it is there to describe. Removing the ground truth
        # removes the row and the note with it, which is correct: there is no longer
        # an accepted text to attribute.
        for source_row in source_rows:
            target = existing_by_line.get(source_row.line_id)
            if target is None:
                session.add(
                    LineTranscription(
                        line_id=source_row.line_id,
                        transcription_id=ground_truth.id,
                        text=source_row.text,
                        confidence=None,
                        character_confidences=None,
                        seeded_from_transcription_id=source.id,
                    )
                )
            else:
                target.text = source_row.text
                target.confidence = None
                target.character_confidences = None
                target.seeded_from_transcription_id = source.id

        await session.commit()
        return copied_line_ids

    async def patch_ground_truth_line_text(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        transcription_id: UUID,
        line_id: UUID,
        *,
        text: str,
        previous_text: str | None = None,
    ) -> LineTranscription:
        context = await self._access.require_document(session, user, project_id, document_id)
        document = context.document
        transcription = await self._transcription_or_404(session, document, transcription_id)
        if transcription.kind != TranscriptionKind.ground_truth:
            raise ConflictError("Only Ground truth transcription lines can be edited")
        line = await self._line_in_document_or_404(session, document, line_id)
        # Ground-truth text is the other thing that makes a line untouchable by
        # segment health, so writing it takes the same lock a delete holds. The
        # part comes from the line because this route is addressed by document.
        await self._documents.lock_part(session, line.part_id)
        # That first read was unlocked, taken only to learn which part to hold, so
        # a segment-health delete can commit in the gap and leave ``line`` pointing
        # at a row that is gone. Every other writer here locks before it reads and
        # so cannot see this; read it again now the part is held. Gone means a
        # plain 404, which is what the caller can act on, instead of the
        # foreign-key error the insert below would otherwise raise.
        line = await self._line_in_document_or_404(session, document, line_id)

        result = await session.execute(
            select(LineTranscription).where(
                LineTranscription.transcription_id == transcription.id,
                LineTranscription.line_id == line_id,
            )
        )
        line_transcription = result.scalar_one_or_none()
        # The client names the text its edit started from. Anything else stored
        # now was written by someone else since, and saving over it would drop
        # their work without telling either side. No stored row reads as "".
        stored_text = line_transcription.text if line_transcription is not None else ""
        if previous_text is not None and previous_text != stored_text:
            raise ConflictError("This line's text was changed elsewhere")
        if line_transcription is None:
            line_transcription = LineTranscription(
                line_id=line_id,
                transcription_id=transcription.id,
                text=text,
                confidence=None,
                character_confidences=None,
            )
            session.add(line_transcription)
        else:
            line_transcription.text = text
            line_transcription.confidence = None
            line_transcription.character_confidences = None
        await session.commit()
        await session.refresh(line_transcription)
        # The API serialises the row's layer kind from the relationship, which
        # this write never loaded. Attach the layer already in hand (in memory,
        # no IO) rather than paying a query for a value the caller supplied.
        line_transcription.transcription = transcription
        return line_transcription

    # --- Lookups ---

    def _split_page_transcription(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    async def _line_or_404(self, session: AsyncSession, part_id: UUID, line_id: UUID) -> Line:
        line = await self._documents.get_line_in_part(session, part_id, line_id)
        if line is None:
            raise NotFoundError("Line not found")
        return line

    async def _page_transcription_line_or_404(
        self, session: AsyncSession, part_id: UUID, order: int
    ) -> PageTranscriptionLine:
        text_line = await self._documents.get_page_transcription_line(session, part_id, order)
        if text_line is None:
            raise NotFoundError("Text line not found")
        return text_line

    async def _transcription_or_404(
        self, session: AsyncSession, document: Document, transcription_id: UUID
    ) -> Transcription:
        transcription = await self._documents.get_transcription_in_document(
            session, document.id, transcription_id
        )
        if transcription is None:
            raise NotFoundError("Transcription layer not found")
        return transcription

    async def _line_in_document_or_404(
        self, session: AsyncSession, document: Document, line_id: UUID
    ) -> Line:
        line = await self._documents.get_line_in_document(session, document.id, line_id)
        if line is None:
            raise NotFoundError("Line not found")
        return line
