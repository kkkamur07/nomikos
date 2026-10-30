"""Async request-scoped side of job persistence - enqueue, reads, and cancellation.

The other half - the sync claim/sweep/transition machinery that runs on
``sync_system_session()`` from the worker loop and the on-read stale sweep -
lives in ``job_claim_engine``, ``job_sweeps`` and ``job_terminal``. Different
session types, different lifecycles, different callers; the sync half exists
per ADR 0005.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.api.pagination import PageCursor
from backend.document.infrastructure.orm_models import Document
from backend.jobs.infrastructure.job_terminal import apply_cancellation
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import rows_affected

_TERMINAL_STATUSES = (JobStatus.done, JobStatus.failed, JobStatus.cancelled)


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_test_job(self, handler: str, *, user_id: uuid.UUID | None = None) -> Job:
        job = Job(
            type=JobType.pipeline,
            status=JobStatus.pending,
            payload={"handler": handler, "test": True},
            user_id=user_id,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> Job | None:
        result = await self._session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def list_for_project(
        self,
        project_id: uuid.UUID,
        *,
        limit: int = 50,
        cursor: PageCursor | None = None,
    ) -> list[Job]:
        query = (
            select(Job)
            .join(Document, Job.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(~Job.payload.contains({"test": True}))
            .order_by(Job.created_at.desc(), Job.id.desc())
        )
        if cursor is not None:
            query = query.where(tuple_(Job.created_at, Job.id) < (cursor.created_at, cursor.id))
        query = query.limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def seconds_since_oldest_pending_job(self) -> float | None:
        """How long the head of the pending queue has waited; ``None`` if nothing pends.

        The one observable that says "something is claiming this queue". Nothing
        in the API deployment claims a pending job: the platform worker runs only
        under ``JOB_WORKER_ENABLED`` on a separate host, and ``segment`` and
        ``transcribe`` are claimed over HTTP by an inference agent. The on-read
        stale sweep reaps timeouts and never claims. So if no consumer is up,
        pending rows accumulate and every other signal stays green - see
        ``backend.core.api.health``, the only caller.

        **Every pending row counts, agent-claimed types included.** Restricting
        this to ``claim_next_pending_job``'s population would make it permanently
        ``None`` in production: the only job types anything enqueues are
        ``transcribe`` and ``segment`` (``document_job_enqueue``), and both are
        excluded from that claim by ``AGENT_CLAIMED_JOB_TYPES``. A number that is
        structurally always ``None`` is worse than no number.

        Measured from ``created_at`` rather than ``updated_at``: it is the queue's
        own ordering key, so ``min()`` reads the head of the
        ``ix_jobs_claim_pending`` partial index, and it cannot be reset. A page
        cycling through lease expiry (``release_expired_device_leases``) keeps its
        full age here, which is intended - a page pending for an hour is stuck
        whether it was ignored for an hour or claimed and abandoned six times.
        """
        oldest_created_at = (
            await self._session.execute(
                select(func.min(Job.created_at)).where(Job.status == JobStatus.pending)
            )
        ).scalar_one_or_none()
        if oldest_created_at is None:
            return None
        # Floored at zero: ``created_at`` is a server default, so a DB clock a
        # little ahead of this host's would otherwise report a negative age.
        return max((datetime.now(UTC) - oldest_created_at).total_seconds(), 0.0)

    async def delete_terminal_jobs_for_project(
        self, project_id: uuid.UUID, *, user_id: uuid.UUID
    ) -> int:
        """Delete *user_id*'s finished jobs of a project; active rows are kept.

        Scoped to the caller: project membership lets someone read the project's
        history, not erase another member's.
        """
        terminal_ids = (
            select(Job.id)
            .join(Document, Job.document_id == Document.id)
            .where(Document.project_id == project_id)
            .where(Job.user_id == user_id)
            .where(Job.status.in_(_TERMINAL_STATUSES))
            .scalar_subquery()
        )
        result = await self._session.execute(delete(Job).where(Job.id.in_(terminal_ids)))
        await self._session.commit()
        return rows_affected(result)


async def cancel_job_async(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    """Atomically cancel a pending/running/waiting job under ``FOR UPDATE``.

    Raises ConflictError when the job is already terminal or a callback has been
    claimed (merge may already be applying document changes).
    """
    now = datetime.now(UTC)
    job = (
        await session.execute(select(Job).where(Job.id == job_id).with_for_update())
    ).scalar_one_or_none()
    if job is None:
        return None
    apply_cancellation(job, now)
    await session.commit()
    await session.refresh(job)
    # pg_notify uses a sync DB session - keep it off the event loop.
    await asyncio.to_thread(notify_platform_job_status_changed, job.id, JobStatus.cancelled)
    return job
