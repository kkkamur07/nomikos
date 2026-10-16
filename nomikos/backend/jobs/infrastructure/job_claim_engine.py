"""Sync claim side of job persistence, using SKIP LOCKED on ``sync_system_session()``.

Split out of ``job_repository`` because the two halves share nothing but the
``jobs`` table: this half runs on short-lived sync system sessions from the
worker loop and the on-read stale sweep, while the repository's async half runs
on the request-scoped session. This sync half exists because ADR 0005 put the
claim/lease machinery on the platform side.

The sweeps live in ``job_sweeps`` and the terminal writes in ``job_terminal``.
The names re-exported below are the ones the integration suite still imports
from here.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime

from sqlalchemy import Select, func, select

from backend.jobs.infrastructure.job_sql import AGENT_CLAIM_PREFIX, NON_TERMINAL_STATUSES
from backend.jobs.infrastructure.job_sweeps import (
    WAITING_TIMEOUT_ERROR,
    fail_stale_waiting_jobs,
    reclaim_stale_running_jobs,
    release_expired_device_leases,
)
from backend.jobs.infrastructure.job_terminal import mark_job_done
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import sync_system_session

__all__ = [
    "AGENT_CLAIMED_JOB_TYPES",
    "WAITING_TIMEOUT_ERROR",
    "claim_next_pending_job",
    "count_active_jobs",
    "fail_stale_waiting_jobs",
    "mark_job_done",
    "reclaim_stale_running_jobs",
    "release_expired_device_leases",
    "worker_identity",
]

# ``worker:`` namespaces the platform worker's id the way ``agent:`` namespaces a
# device's. Without it the identity was ``{hostname}:{pid}``, and a host called
# ``agent`` would produce ``agent:12345``, which ``job_sql.held_by_agent()`` classifies
# as an agent-held page: it would then be governed by the device lease instead
# of the running timeout, and ``fail_stale_waiting_jobs`` would skip it forever.
# A hostname cannot start with ``worker:``, so the two namespaces are disjoint
# by construction.
_WORKER_IDENTITY = f"worker:{socket.gethostname()}/{os.getpid()}"

# Raise instead of assert: `python -O` strips asserts, and this is the only
# check standing between a badly-named host and a page governed by the wrong
# timeout.
if _WORKER_IDENTITY.startswith(AGENT_CLAIM_PREFIX):
    raise RuntimeError(
        f"the platform worker's identity {_WORKER_IDENTITY!r} collides with the "
        f"{AGENT_CLAIM_PREFIX!r} claim namespace reserved for inference agents"
    )


def worker_identity() -> str:
    """Stable per-process worker id recorded on claim (``worker:host/pid``)."""
    return _WORKER_IDENTITY


# Job types an inference agent claims for itself over HTTP. The platform worker
# leaves them alone: with the second queue gone (ADR 0003) it has no way to run
# them, and claiming one would only fail a job an agent could have run.
AGENT_CLAIMED_JOB_TYPES = (JobType.segment, JobType.transcribe)


def _pending_job_query(*, test_only: bool | None = None) -> Select[tuple[Job]]:
    query = (
        select(Job)
        .where(Job.status == JobStatus.pending)
        .where(Job.type.not_in(AGENT_CLAIMED_JOB_TYPES))
        .order_by(Job.created_at, Job.id)
    )
    if test_only is True:
        query = query.where(Job.payload.contains({"test": True}))
    elif test_only is False:
        query = query.where(~Job.payload.contains({"test": True}))
    return query.with_for_update(skip_locked=True).limit(1)


def claim_next_pending_job(*, test_only: bool | None = None) -> Job | None:
    """Claim one pending job using FOR UPDATE SKIP LOCKED (sync session)."""
    with sync_system_session() as session:
        job = session.execute(_pending_job_query(test_only=test_only)).scalar_one_or_none()
        if job is None:
            return None
        now = datetime.now(UTC)
        job.status = JobStatus.running
        job.claimed_by = worker_identity()
        job.heartbeat_at = now
        job.started_at = now
        job.updated_at = now
        session.commit()
        session.refresh(job)
        notify_platform_job_status_changed(job.id, job.status)
        return job


def count_active_jobs(*, test_payload: bool | None = None) -> int:
    """Count pending, running, or waiting jobs (optionally filter by payload test flag).

    Test support only, used by the integration suite's drain helper to wait
    until the worker has finished everything a test queued. Nothing in the
    application reads it.
    """
    with sync_system_session() as session:
        query = select(func.count()).select_from(Job).where(Job.status.in_(NON_TERMINAL_STATUSES))
        if test_payload is True:
            query = query.where(Job.payload.contains({"test": True}))
        elif test_payload is False:
            query = query.where(~Job.payload.contains({"test": True}))
        return session.execute(query).scalar_one()
