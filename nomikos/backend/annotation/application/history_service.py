from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot
from backend.core.exceptions import NotFoundError
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Line,
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User

HISTORY_RETENTION_LIMIT = 5


class AnnotationHistoryService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService(documents=self._documents)

    async def create_snapshot(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> AnnotationHistorySnapshot:
        await self._require_part(session, user, project_id, document_id, part_id)
        lines = await self._documents.list_part_lines(session, part_id)
        state = {"lines": [self._line_state(line) for line in lines]}
        snapshot = AnnotationHistorySnapshot(
            part_id=part_id,
            state=state,
            line_count=len(lines),
            paired_line_count=sum(
                1
                for line in state["lines"]
                if isinstance(line["approved_text"], str) and line["approved_text"].strip()
            ),
        )
        session.add(snapshot)
        await session.flush()
        await self._prune_snapshots(session, part_id)
        await session.commit()
        await session.refresh(snapshot)
        return snapshot

    async def list_snapshots(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[AnnotationHistorySnapshot]:
        await self._require_part(session, user, project_id, document_id, part_id)
        result = await session.execute(
            select(AnnotationHistorySnapshot)
            .options(
                load_only(
                    AnnotationHistorySnapshot.id,
                    AnnotationHistorySnapshot.part_id,
                    AnnotationHistorySnapshot.line_count,
                    AnnotationHistorySnapshot.paired_line_count,
                    AnnotationHistorySnapshot.created_at,
                )
            )
            .where(AnnotationHistorySnapshot.part_id == part_id)
            .order_by(
                AnnotationHistorySnapshot.created_at.desc(), AnnotationHistorySnapshot.id.desc()
            )
        )
        return list(result.scalars().all())

    async def restore_snapshot(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        snapshot_id: UUID,
    ) -> list[Line]:
        await self._require_part(session, user, project_id, document_id, part_id)
        snapshot = await self._snapshot_or_404(session, part_id, snapshot_id)
        lines = [
            {
                "id": UUID(line["id"]),
                "order": line["order"],
                "kind": LineGeometryKind(line["kind"]),
                "points": line["points"],
                "block_id": UUID(line["block_id"]) if line.get("block_id") is not None else None,
                "source": LineSource(line["source"]),
                "source_metadata": line["source_metadata"],
                "kraken_ceiling": line["kraken_ceiling"],
                "approved_text": line["approved_text"],
                # Absent from snapshots taken before runs were recorded; those lines
                # simply come back without an origin, as they were saved.
                "segmentation_run_id": (
                    UUID(line["segmentation_run_id"])
                    if line.get("segmentation_run_id") is not None
                    else None
                ),
                "run_external_id": line.get("run_external_id"),
            }
            for line in snapshot.state["lines"]
        ]
        return await self._document_service.replace_part_lines(
            session,
            user,
            project_id,
            document_id,
            part_id,
            lines=lines,
            allow_new_ids=True,
            restore_origin=True,
        )

    async def _require_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> None:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")

    async def _snapshot_or_404(
        self, session: AsyncSession, part_id: UUID, snapshot_id: UUID
    ) -> AnnotationHistorySnapshot:
        snapshot = await session.get(AnnotationHistorySnapshot, snapshot_id)
        if snapshot is None or snapshot.part_id != part_id:
            raise NotFoundError("History snapshot not found")
        return snapshot

    async def _prune_snapshots(self, session: AsyncSession, part_id: UUID) -> None:
        result = await session.execute(
            select(AnnotationHistorySnapshot.id)
            .where(AnnotationHistorySnapshot.part_id == part_id)
            .order_by(
                AnnotationHistorySnapshot.created_at.desc(), AnnotationHistorySnapshot.id.desc()
            )
            .offset(HISTORY_RETENTION_LIMIT)
        )
        snapshot_ids = list(result.scalars().all())
        if snapshot_ids:
            await session.execute(
                delete(AnnotationHistorySnapshot).where(
                    AnnotationHistorySnapshot.id.in_(snapshot_ids)
                )
            )

    def _line_state(self, line: Line) -> dict[str, object]:
        ground_truth = next(
            (
                transcription
                for transcription in line.transcriptions
                if transcription.transcription.kind == TranscriptionKind.ground_truth
            ),
            None,
        )
        return {
            "id": str(line.id),
            "block_id": str(line.block_id) if line.block_id is not None else None,
            "order": line.order,
            "kind": line.kind.value,
            "points": line.points,
            "source": line.source.value,
            "source_metadata": line.source_metadata,
            "kraken_ceiling": line.kraken_ceiling,
            "approved_text": ground_truth.text if ground_truth is not None else None,
            # The model origin (ADR 0013). Without it a restored proposal comes back as
            # a line nobody can trace to its run, and the diff reports it as deleted
            # and added at once instead of unchanged.
            "segmentation_run_id": (
                str(line.segmentation_run_id) if line.segmentation_run_id is not None else None
            ),
            "run_external_id": line.run_external_id,
        }
