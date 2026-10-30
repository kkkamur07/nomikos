"""Job application service - enqueue and read job state."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import PageCursor
from backend.core.exceptions import NotFoundError
from backend.jobs.infrastructure.job_repository import JobRepository, cancel_job_async
from backend.jobs.infrastructure.orm_models import Job
from backend.jobs.infrastructure.stale_sweep import sweep_stale_jobs_on_read


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = JobRepository(session)

    async def enqueue_test_job(
        self, handler: str = "noop", *, user_id: uuid.UUID | None = None
    ) -> Job:
        return await self._repo.create_test_job(handler, user_id=user_id)

    async def get_job(self, job_id: uuid.UUID) -> Job:
        job = await self._repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job

    async def list_project_jobs(
        self,
        project_id: uuid.UUID,
        *,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Job]:
        # Hooked here rather than in the route: ``GET /projects/{id}/jobs`` lives in
        # the project package, and this is the job-owned entry point it calls.
        await sweep_stale_jobs_on_read()
        return await self._repo.list_for_project(project_id, limit=limit, cursor=cursor)

    async def clear_project_job_history(self, project_id: uuid.UUID, *, user_id: uuid.UUID) -> int:
        """Delete the caller's finished jobs of a project; active jobs keep running."""
        return await self._repo.delete_terminal_jobs_for_project(project_id, user_id=user_id)

    async def cancel_job(self, job_id: uuid.UUID) -> Job:
        job = await cancel_job_async(self._session, job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job
