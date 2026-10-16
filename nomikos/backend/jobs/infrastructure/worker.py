"""Background job worker - polls Postgres and runs claimed jobs.

Multiple Uvicorn workers are safe: claim uses ``FOR UPDATE SKIP LOCKED`` so only
one process runs a given job. Parallelism here means *different* jobs at once,
not the same job twice.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from backend.core.settings.device import get_device_settings
from backend.core.settings.job import JobSettings, get_job_settings
from backend.jobs.infrastructure.handlers import TestJobHandlerError, run_test_handler
from backend.jobs.infrastructure.job_claim_engine import (
    claim_next_pending_job,
    clear_stale_callback_claims,
    fail_stale_waiting_jobs,
    mark_job_done,
    mark_job_failed,
    reclaim_stale_running_jobs,
    release_expired_device_leases,
    seconds_until_next_stale_waiting_job,
)
from backend.jobs.infrastructure.orm_models import Job

if TYPE_CHECKING:
    from asyncio import Event

logger = logging.getLogger(__name__)


def _public_job_error(exc: BaseException, *, fallback: str = "Job failed") -> str:
    """Return allowlisted job output; exception text stays server-side only."""
    if isinstance(exc, TestJobHandlerError):
        return "Test job failed"
    return fallback


def execute_claimed_job(job: Job) -> None:
    """Run handler for a job already in ``running`` status.

    Inference job types never reach here: ``claim_next_pending_job`` leaves them
    for an inference agent to claim over HTTP (ADR 0003). A job of one of those
    types that somehow arrives has no handler and fails like any other.
    """
    # Terminal writes below are scoped to the claim this worker still holds, so a
    # job reclaimed mid-run cannot be finished twice. The inference callback path
    # completes ``waiting`` jobs from another process and does not go through here.
    owner = job.claimed_by

    if (job.payload or {}).get("test"):
        try:
            result = run_test_handler(job)
        except TestJobHandlerError as exc:
            logger.warning("Test job %s failed: %s", job.id, exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
            return
        except Exception as exc:
            logger.exception("Test job %s failed", job.id, exc_info=exc)
            mark_job_failed(job.id, _public_job_error(exc), claimed_by=owner)
            return
        mark_job_done(job.id, result, claimed_by=owner)
        return

    logger.error("No handler for job %s type=%s", job.id, job.type.value)
    mark_job_failed(job.id, "Job type is not supported", claimed_by=owner)


def process_one_job() -> bool:
    """Claim and execute at most one job. Returns True if work was done."""
    settings = get_job_settings()
    reclaimed = reclaim_stale_running_jobs(
        running_timeout_seconds=settings.job_worker_running_timeout_seconds
    )
    if reclaimed:
        # "released", not "reclaimed": a job whose claim has been abandoned to the
        # ceiling is failed here rather than re-pended, and the repository logs
        # each of those by id.
        logger.warning("released %s stale platform job claim(s)", reclaimed)
    # Sweep waiting before releasing claims: releasing touches the row and would
    # otherwise push the waiting deadline out by another timeout window.
    timed_out = fail_stale_waiting_jobs(
        waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds
    )
    if timed_out:
        logger.warning("failed %s platform job(s) waiting past the inference timeout", timed_out)
    # Agent-held pages are the other half of ``waiting``, and they re-pend rather
    # than fail. The worker runs the identical sweep the read paths run, so a
    # deployment with a worker and one without recover a stopped agent the same
    # way - one lease, one behaviour, not two that drift.
    re_pended = release_expired_device_leases(
        lease_seconds=get_device_settings().device_lease_seconds
    )
    if re_pended:
        logger.warning("released %s page(s) whose device lease expired", re_pended)
    released = clear_stale_callback_claims(
        claim_timeout_seconds=settings.job_worker_callback_claim_timeout_seconds
    )
    if released:
        logger.warning("released %s stale inference callback claim(s)", released)
    job = claim_next_pending_job(test_only=settings.job_worker_claim_test_only)
    if job is None:
        return False
    execute_claimed_job(job)
    return True


async def _idle_wait_seconds(settings: JobSettings, idle_interval: float) -> float:
    """Cap the idle backoff so it can never sleep past a waiting job's deadline.

    The deadline being respected is the inference-dispatch timeout, whose
    population is empty today - every ``waiting`` row carries an agent claim, so
    ``fail_stale_waiting_jobs`` has no rows. The cap still fires whenever *any*
    job is waiting, agent-held ones included, because the query does not
    discriminate. The effect is a worker that wakes sooner than it needs to,
    which is the harmless direction; it is not the lease's mechanism. Pages an
    agent holds are recovered by ``release_expired_device_leases``, from this same
    tick and from the on-read sweep.
    """
    waiting_deadline = await asyncio.to_thread(
        seconds_until_next_stale_waiting_job,
        waiting_timeout_seconds=settings.job_worker_waiting_timeout_seconds,
    )
    if waiting_deadline is None:
        return idle_interval
    # Floor at the poll interval so an already-expired deadline cannot spin the loop.
    return max(min(idle_interval, waiting_deadline), settings.job_poll_interval_seconds)


async def worker_loop(stop_event: Event) -> None:
    """Poll for pending jobs until ``stop_event`` is set; backoff when idle."""
    settings = get_job_settings()
    min_interval = settings.job_poll_interval_seconds
    idle_interval = min_interval
    max_interval = settings.job_poll_max_interval_seconds

    while not stop_event.is_set():
        try:
            worked = await asyncio.to_thread(process_one_job)
        except Exception:
            logger.exception("job worker tick failed")
            worked = False
        if worked:
            idle_interval = min_interval
        else:
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=await _idle_wait_seconds(settings, idle_interval),
                )
            except TimeoutError:
                idle_interval = min(idle_interval * 2, max_interval)
