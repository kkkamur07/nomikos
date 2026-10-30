"""Reading the segmentation provenance of a page. Nothing here writes.

Two questions, and only two. Which model runs have touched this page, and for one
of them, what did the humans change afterwards. The second is answered by
:mod:`backend.document.domain.segmentation_diff`, which is pure; this module is
the part that knows about sessions, access control and which run "latest" means.

Deliberately read-only. The run rows are immutable by construction - nothing in
the application updates one after ``SegmentMergeService`` writes it - and keeping
the reader free of writes is what makes that easy to see.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.document.application.document_access import DocumentAccess
from backend.document.domain.segmentation_diff import (
    LiveLine,
    SegmentationDiff,
    diff_segmentation_run,
    run_lines_from_payload,
)
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import SegmentationRun
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User


class SegmentationRunService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)

    async def list_runs(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[SegmentationRun]:
        """Every run recorded for this page, newest first."""
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        return await self._runs_for_part(session, context.part.id)

    async def diff(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        run_id: UUID | None = None,
    ) -> tuple[SegmentationRun, SegmentationDiff]:
        """One run against the page as it stands. ``run_id`` omitted means the latest."""
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        run = await self._require_run(session, context.part.id, run_id)
        lines = await self._documents.list_part_lines(session, context.part.id)
        live = [
            LiveLine(
                id=line.id,
                order=line.order,
                points=list(line.points or []),
                segmentation_run_id=line.segmentation_run_id,
                run_external_id=line.run_external_id,
            )
            for line in lines
        ]
        return run, diff_segmentation_run(run_lines_from_payload(run.lines), live, run_id=run.id)

    async def _runs_for_part(self, session: AsyncSession, part_id: UUID) -> list[SegmentationRun]:
        # ``created_at`` then ``id``: two runs on one page can share a timestamp when
        # a retry lands inside the same clock tick, and an unordered tie makes
        # "latest" flip between requests.
        result = await session.execute(
            select(SegmentationRun)
            .where(SegmentationRun.part_id == part_id)
            .order_by(SegmentationRun.created_at.desc(), SegmentationRun.id.desc())
        )
        return list(result.scalars().all())

    async def _require_run(
        self, session: AsyncSession, part_id: UUID, run_id: UUID | None
    ) -> SegmentationRun:
        runs = await self._runs_for_part(session, part_id)
        if not runs:
            raise NotFoundError("This page has no recorded segmentation run")
        if run_id is None:
            return runs[0]
        # Scoped to the part, not fetched by id alone: a run id from another page,
        # or another project, must not answer here.
        run = next((item for item in runs if item.id == run_id), None)
        if run is None:
            raise NotFoundError("Segmentation run not found")
        return run


__all__ = ["SegmentationRunService"]
