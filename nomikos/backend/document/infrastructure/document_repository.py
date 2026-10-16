"""Document and DocumentPart persistence."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.api.pagination import PageCursor
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    DocumentWorkflow,
    Line,
    LineTranscription,
    MediaDeletionIntent,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.document.infrastructure.part_repository import PartRepository


class DocumentRepository(PartRepository):
    """Documents, layout and text rows; part ordering and media intents are inherited."""

    async def get_by_id(self, session: AsyncSession, document_id: UUID) -> Document | None:
        result = await session.execute(
            select(Document)
            .options(selectinload(Document.parts))
            .options(selectinload(Document.transcriptions))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_authz(
        self, session: AsyncSession, document_id: UUID
    ) -> Document | None:
        """Load a document row without parts/transcriptions (media authz hot path)."""
        result = await session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        session: AsyncSession,
        project_id: UUID,
        *,
        include_archived: bool = False,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Document]:
        stmt = select(Document).where(Document.project_id == project_id)
        if not include_archived:
            stmt = stmt.where(Document.workflow != DocumentWorkflow.archived)
        stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc())
        if cursor is not None:
            stmt = stmt.where(
                tuple_(Document.created_at, Document.id) < (cursor.created_at, cursor.id)
            )
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def count_parts_by_document_ids(
        self, session: AsyncSession, document_ids: list[UUID]
    ) -> dict[UUID, int]:
        if not document_ids:
            return {}
        result = await session.execute(
            select(DocumentPart.document_id, func.count())
            .where(DocumentPart.document_id.in_(document_ids))
            .group_by(DocumentPart.document_id)
        )
        return {document_id: int(count) for document_id, count in result.all()}

    async def count_documents_by_project_ids(
        self, session: AsyncSession, project_ids: list[UUID]
    ) -> dict[UUID, int]:
        if not project_ids:
            return {}
        result = await session.execute(
            select(Document.project_id, func.count())
            .where(Document.project_id.in_(project_ids))
            .group_by(Document.project_id)
        )
        return {project_id: int(count) for project_id, count in result.all()}

    async def create(
        self,
        session: AsyncSession,
        *,
        project_id: UUID,
        name: str,
        workflow: DocumentWorkflow = DocumentWorkflow.draft,
    ) -> Document:
        document = Document(project_id=project_id, name=name, workflow=workflow)
        document.transcriptions.append(
            Transcription(
                name="Ground truth",
                kind=TranscriptionKind.ground_truth,
            )
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return document

    async def update(
        self,
        session: AsyncSession,
        document: Document,
        **fields: object,
    ) -> Document:
        for key, value in fields.items():
            setattr(document, key, value)
        await session.commit()
        await session.refresh(document)
        return document

    async def delete(self, session: AsyncSession, document: Document) -> None:
        await session.delete(document)
        await session.commit()

    async def delete_with_media_intents(
        self,
        session: AsyncSession,
        document: Document,
        image_keys: list[str],
    ) -> None:
        for image_key in image_keys:
            session.add(MediaDeletionIntent(image_key=image_key))
        await session.delete(document)
        await session.commit()

    async def get_part(self, session: AsyncSession, part_id: UUID) -> DocumentPart | None:
        result = await session.execute(
            select(DocumentPart)
            .options(
                selectinload(DocumentPart.lines)
                .selectinload(Line.transcriptions)
                .selectinload(LineTranscription.transcription)
            )
            .where(DocumentPart.id == part_id)
        )
        return result.scalar_one_or_none()

    async def get_part_row(self, session: AsyncSession, part_id: UUID) -> DocumentPart | None:
        """Load a part without lines/transcriptions (media serving authz)."""
        result = await session.execute(select(DocumentPart).where(DocumentPart.id == part_id))
        return result.scalar_one_or_none()

    async def list_transcriptions(
        self, session: AsyncSession, document_id: UUID
    ) -> list[Transcription]:
        result = await session.execute(
            select(Transcription)
            .where(Transcription.document_id == document_id)
            .order_by(Transcription.created_at, Transcription.id)
        )
        return list(result.scalars().all())

    async def get_ground_truth_transcription(
        self, session: AsyncSession, document_id: UUID
    ) -> Transcription | None:
        result = await session.execute(
            select(Transcription).where(
                Transcription.document_id == document_id,
                Transcription.kind == TranscriptionKind.ground_truth,
            )
        )
        return result.scalar_one_or_none()

    async def list_part_lines(self, session: AsyncSession, part_id: UUID) -> list[Line]:
        result = await session.execute(
            select(Line)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
            .where(Line.part_id == part_id)
            .order_by(Line.order, Line.created_at)
        )
        return list(result.scalars().all())

    async def list_blocks_for_document(
        self, session: AsyncSession, document_id: UUID, *, limit: int
    ) -> list[Block]:
        # Anonymous layout paging is the only caller of this method, and a document
        # being published does not mean every one of its pages is - a held-back part's
        # geometry must not leak through here even though the parent document is public.
        result = await session.execute(
            select(Block)
            .join(DocumentPart, Block.part_id == DocumentPart.id)
            .where(DocumentPart.document_id == document_id, DocumentPart.published.is_(True))
            .order_by(DocumentPart.order, Block.order, Block.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_lines_for_document(
        self,
        session: AsyncSession,
        document_id: UUID,
        *,
        limit: int,
        cursor: PageCursor | None = None,
    ) -> list[Line]:
        """Keyset page over a document's *published* lines.

        Ordered by ``(created_at, id)`` so the shared ``PageCursor`` applies; clients
        group by part and sort by ``order`` themselves. Same reasoning as
        ``list_blocks_for_document`` for the ``published`` filter: this is the anonymous
        read, and a held-back part's lines (with their approved text) must not surface
        here just because the document around them is public.
        """
        stmt = (
            select(Line)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(DocumentPart.document_id == document_id, DocumentPart.published.is_(True))
            .order_by(Line.created_at, Line.id)
        )
        if cursor is not None:
            stmt = stmt.where(tuple_(Line.created_at, Line.id) > (cursor.created_at, cursor.id))
        result = await session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    async def list_part_blocks(self, session: AsyncSession, part_id: UUID) -> list[Block]:
        result = await session.execute(
            select(Block).where(Block.part_id == part_id).order_by(Block.order, Block.created_at)
        )
        return list(result.scalars().all())

    async def get_block_in_part(
        self, session: AsyncSession, part_id: UUID, block_id: UUID
    ) -> Block | None:
        result = await session.execute(
            select(Block).where(Block.id == block_id, Block.part_id == part_id)
        )
        return result.scalar_one_or_none()

    async def get_line_in_part(
        self, session: AsyncSession, part_id: UUID, line_id: UUID
    ) -> Line | None:
        result = await session.execute(
            select(Line)
            .where(Line.id == line_id, Line.part_id == part_id)
            .options(
                selectinload(Line.transcriptions).selectinload(LineTranscription.transcription)
            )
        )
        return result.scalar_one_or_none()

    async def get_line_in_document(
        self, session: AsyncSession, document_id: UUID, line_id: UUID
    ) -> Line | None:
        result = await session.execute(
            select(Line)
            .join(DocumentPart, Line.part_id == DocumentPart.id)
            .where(Line.id == line_id, DocumentPart.document_id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_transcription_in_document(
        self, session: AsyncSession, document_id: UUID, transcription_id: UUID
    ) -> Transcription | None:
        transcription = await session.get(Transcription, transcription_id)
        if transcription is None or transcription.document_id != document_id:
            return None
        return transcription

    async def count_part_lines(self, session: AsyncSession, part_id: UUID) -> int:
        result = await session.execute(
            select(func.count()).select_from(Line).where(Line.part_id == part_id)
        )
        return int(result.scalar_one())

    async def count_paired_ground_truth_lines(self, session: AsyncSession, part_id: UUID) -> int:
        """Lines carrying non-blank ground-truth text: the denominator of pairing progress."""
        result = await session.execute(
            select(func.count(func.distinct(Line.id)))
            .select_from(Line)
            .join(LineTranscription, LineTranscription.line_id == Line.id)
            .join(Transcription, LineTranscription.transcription_id == Transcription.id)
            .where(
                Line.part_id == part_id,
                Transcription.kind == TranscriptionKind.ground_truth,
                func.length(func.trim(LineTranscription.text)) > 0,
            )
        )
        return int(result.scalar_one())

    async def list_page_transcription_lines(
        self, session: AsyncSession, part_id: UUID
    ) -> list[PageTranscriptionLine]:
        result = await session.execute(
            select(PageTranscriptionLine)
            .where(PageTranscriptionLine.part_id == part_id)
            .order_by(PageTranscriptionLine.order, PageTranscriptionLine.created_at)
        )
        return list(result.scalars().all())

    async def paired_line_ids(self, session: AsyncSession, part_id: UUID) -> set[UUID]:
        """Lines a human has paired to a line of page transcription.

        A pairing is a decision about a line recorded outside the transcription
        layers, so a line can be paired while carrying no ground-truth text of
        its own: text imported and paired before anyone approved it has the
        pairing without the layer. Anything asking "has a person touched this
        line" has to read this as well as the text.
        """
        result = await session.execute(
            select(PageTranscriptionLine.paired_line_id).where(
                PageTranscriptionLine.part_id == part_id,
                PageTranscriptionLine.paired_line_id.is_not(None),
            )
        )
        return {line_id for line_id in result.scalars().all() if line_id is not None}

    async def lock_part(self, session: AsyncSession, part_id: UUID) -> None:
        """Hold one part's lines still until the caller commits.

        A lock only works if every writer takes it, so every path that rewrites
        a part's lines does: ``segment_merge_service`` before it replaces the
        machine geometry, ``SegmentHealthService._recompute`` before it derives
        a fix from the list, and every ``LayoutService`` method that commits.
        That last one is a rule, not a list, and it is enforced as one in
        ``test_every_layout_write_takes_the_part_lock``: enumerating the paths by
        hand is what left ``create_part_line`` unlocked through a whole round of
        review. Each reads the rows and writes back
        a decision made from them, so without a shared lock the one that commits
        second wins on rows it never read: newer geometry overwritten, a
        deleted line acted on, orders renumbered from a stale list.

        Take it after the access check, never before. Locking first would let
        any signed-in caller hold a row on a part they cannot read by naming its
        id. Selecting the id alone keeps this a lock and nothing else: no entity
        comes back, so there is no loaded instance to go stale the way
        ``reorder_parts`` describes.
        """
        await session.execute(
            select(DocumentPart.id).where(DocumentPart.id == part_id).with_for_update()
        )

    async def lock_parts(self, session: AsyncSession, part_ids: Iterable[UUID]) -> None:
        """``lock_part`` for an operation that spans several parts at once.

        Locking many rows is where a deadlock becomes possible, so they are
        taken in one statement ordered by id: two callers that overlap therefore
        queue on the same row first instead of each holding what the other wants
        next. Single-part callers cannot join such a cycle, because they take
        one row and then wait for nothing.
        """
        ids = sorted(set(part_ids))
        if not ids:
            return
        await session.execute(
            select(DocumentPart.id)
            .where(DocumentPart.id.in_(ids))
            .order_by(DocumentPart.id)
            .with_for_update()
        )

    async def get_page_transcription_line(
        self, session: AsyncSession, part_id: UUID, order: int
    ) -> PageTranscriptionLine | None:
        result = await session.execute(
            select(PageTranscriptionLine).where(
                PageTranscriptionLine.part_id == part_id,
                PageTranscriptionLine.order == order,
            )
        )
        return result.scalar_one_or_none()
