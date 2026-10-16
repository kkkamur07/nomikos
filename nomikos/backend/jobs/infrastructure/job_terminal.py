"""Terminal writes on ``jobs``: the worker's done/failed and the user's cancel."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Update, update

from backend.core.exceptions import ConflictError
from backend.jobs.infrastructure.job_sql import NON_TERMINAL_STATUSES, rows_affected
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from infrastructure.db import sync_system_session


def _owned_by(statement: Update, claimed_by: str | None) -> Update:
    """Restrict a worker's terminal write to the claim it still owns.

    ``reclaim_stale_running_jobs`` clears ``claimed_by`` and re-pends the job, so
    a zombie worker that lost its lease no longer matches: either the column is
    NULL (not yet re-claimed) or it holds the new owner's id. Status alone isn't
    enough, since the reclaimed job is legitimately non-terminal again.
    """
    return statement.where(Job.claimed_by.is_not_distinct_from(claimed_by))


def mark_job_failed(job_id: uuid.UUID, error: str, *, claimed_by: str | None) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            _owned_by(
                update(Job)
                .where(Job.id == job_id)
                .where(Job.status.notin_((JobStatus.cancelled, JobStatus.done))),
                claimed_by,
            ).values(
                status=JobStatus.failed,
                error=error,
                callback_claimed_at=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
    if rows_affected(update_result):
        notify_platform_job_status_changed(job_id, JobStatus.failed)


def mark_job_done(job_id: uuid.UUID, result: dict | None = None, *, claimed_by: str | None) -> None:
    now = datetime.now(UTC)
    with sync_system_session() as session:
        update_result = session.execute(
            _owned_by(
                update(Job)
                .where(Job.id == job_id)
                .where(Job.status.notin_((JobStatus.cancelled, JobStatus.failed))),
                claimed_by,
            ).values(
                status=JobStatus.done,
                result=result or {},
                error=None,
                # Reset rather than left standing, so a job that took four tries
                # and then succeeded doesn't carry a one-lap budget into
                # whatever re-pends it next.
                claim_attempts=0,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
    if rows_affected(update_result):
        notify_platform_job_status_changed(job_id, JobStatus.done)


def apply_cancellation(job: Job, now: datetime) -> None:
    """Cancel the job and release the claim on it.

    Clearing ``claimed_by`` and ``inference_job_id`` isn't tidiness. A cancelled
    page left with both set is still, as far as the agent holding it knows, its
    page: it runs the whole thing and posts a callback that
    ``JobCallbackService`` then discards on the terminal-status check, meaning a
    laptop kept working for minutes after the researcher cancelled. Cleared, the
    agent's next callback is refused immediately: the callback route matches
    ``claimed_by`` and the merge matches ``inference_job_id``, and neither
    matches once this has run. Same two columns, same reason, as
    ``release_expired_device_leases``.
    """
    if job.status not in NON_TERMINAL_STATUSES:
        raise ConflictError(f"job {job.id} cannot be cancelled from status {job.status.value}")
    if job.callback_claimed_at is not None:
        raise ConflictError(
            f"job {job.id} cannot be cancelled while an inference callback is applied"
        )
    job.status = JobStatus.cancelled
    job.error = None
    job.result = None
    job.claimed_by = None
    job.inference_job_id = None
    job.callback_claimed_at = None
    job.completed_at = now
    job.updated_at = now
