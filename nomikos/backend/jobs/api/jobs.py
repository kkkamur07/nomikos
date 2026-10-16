"""Job routes - test enqueue and status polling."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import GreekOCRException
from backend.core.settings.job import get_job_settings
from backend.jobs.api.schemas import (
    ClearJobHistoryResponse,
    EnqueueTestJobRequest,
    EnqueueTestJobResponse,
    JobResponse,
    job_response_from_orm,
)
from backend.jobs.application.job_service import JobService
from backend.jobs.infrastructure.notifications import job_status_broadcaster
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from backend.jobs.infrastructure.stale_sweep import sweep_stale_jobs_on_read
from backend.project.application.project_service import ProjectService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import AsyncSessionLocal, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

_project_service = ProjectService()


def _job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)


def _require_test_routes_enabled() -> None:
    if not get_job_settings().enable_test_job_routes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def _assert_job_access(job: Job, current_user: User) -> None:
    if job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _load_authorized_job(job_id: UUID, current_user: User) -> JobResponse:
    async with AsyncSessionLocal() as session:
        job = await JobService(session).get_job(job_id)
        _assert_job_access(job, current_user)
        return job_response_from_orm(job)


async def _job_events(job_id: UUID, current_user: User, request: Request) -> AsyncIterator[str]:
    settings = get_job_settings()
    queue = await job_status_broadcaster.subscribe(job_id)
    try:
        current = await _load_authorized_job(job_id, current_user)
        yield _sse_event("job", current.model_dump_json())
        if current.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
            return

        while not await request.is_disconnected():
            try:
                await asyncio.wait_for(
                    queue.get(),
                    timeout=settings.job_sse_heartbeat_seconds,
                )
            except TimeoutError:
                yield ": heartbeat\n\n"
                continue

            current = await _load_authorized_job(job_id, current_user)
            yield _sse_event("job", current.model_dump_json())
            if current.status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled):
                return
    except (HTTPException, GreekOCRException):
        # Access revoked or the job disappeared mid-stream. Headers are already
        # sent, so end with an explicit error event instead of hanging the client.
        logger.info("closing job %s event stream: no longer readable", job_id)
        yield _sse_event("error", json.dumps({"detail": "Job stream closed"}))
    finally:
        await job_status_broadcaster.unsubscribe(job_id, queue)


@router.post("/test", response_model=EnqueueTestJobResponse, status_code=201)
async def enqueue_test_job(
    body: EnqueueTestJobRequest | None = None,
    service: JobService = Depends(_job_service),
    _enabled: None = Depends(_require_test_routes_enabled),
    current_user: User = Depends(get_current_user),
) -> EnqueueTestJobResponse:
    handler = body.handler if body else "noop"
    job = await service.enqueue_test_job(handler, user_id=current_user.id)
    return EnqueueTestJobResponse(job_id=job.id)


@router.delete("/history", response_model=ClearJobHistoryResponse)
async def clear_job_history(
    project_id: UUID = Query(...),
    db: AsyncSession = Depends(get_db),
    service: JobService = Depends(_job_service),
    current_user: User = Depends(get_current_user),
) -> ClearJobHistoryResponse:
    """Delete finished jobs of a project. Active jobs are left running."""
    await _project_service.get_project(db, current_user, project_id)
    deleted = await service.clear_project_job_history(project_id)
    return ClearJobHistoryResponse(deleted=deleted)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(_job_service),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    # Recover stale jobs before reading. On a request/response host there is no
    # worker loop, so this poll is the only thing that can release an expired
    # device lease - without it an abandoned page is reported as alive forever.
    await sweep_stale_jobs_on_read()
    job = await service.get_job(job_id)
    _assert_job_access(job, current_user)
    return job_response_from_orm(job)


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    service: JobService = Depends(_job_service),
    current_user: User = Depends(get_current_user),
) -> JobResponse:
    job = await service.get_job(job_id)
    _assert_job_access(job, current_user)
    cancelled = await service.cancel_job(job_id)
    return job_response_from_orm(cancelled)


@router.get("/{job_id}/events", response_class=StreamingResponse)
async def stream_job_events(
    job_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    # Sweep before the initial load so a page whose lease already expired streams
    # its new state immediately instead of parking the client on a dead stream.
    await sweep_stale_jobs_on_read()
    # No ``get_db`` dependency here: it would pin a pooled connection for the whole
    # stream. Authorize on a short-lived session so 403/404 still surface as a
    # normal error response, then let the generator open its own sessions.
    await _load_authorized_job(job_id, current_user)
    return StreamingResponse(
        _job_events(job_id, current_user, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
