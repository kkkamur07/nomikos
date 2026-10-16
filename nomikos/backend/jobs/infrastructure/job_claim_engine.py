"""Sync claim/sweep/transition side of job persistence, using SKIP LOCKED on ``sync_system_session()``.

Split out of ``job_repository`` because the two halves share nothing but the
``jobs`` table: this half runs on short-lived sync system sessions from the
worker loop and the on-read stale sweep, while the repository's async half runs
on the request-scoped session. This sync half exists because ADR 0005 put the
claim/lease machinery on the platform side.
"""

from __future__ import annotations

import logging
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select, update

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.exceptions import ConflictError
from backend.jobs.infrastructure.notifications import notify_platform_job_status_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

_NON_TERMINAL_STATUSES = (JobStatus.pending, JobStatus.running, JobStatus.waiting)

WAITING_TIMEOUT_ERROR = "Inference timed out with no response"

POISON_PAGE_ERROR = "This page could not be completed by any inference agent"

MAX_CLAIM_ATTEMPTS = 5
"""How many abandoned claims a page survives before it is failed.

Both stale sweeps re-pend an abandoned job by default (a closed laptop lid is
not a failed job), but a page that reliably kills whatever runs it (a corrupt
scan, an image the model crashes on) would otherwise loop forever, burning one
agent slot per lap and never telling the researcher anything.
``jobs.claim_attempts`` counts the laps and this is where they stop.

A constant rather than an environment dial, unlike the two timeouts next to it:
those are operational (how long to wait before believing an agent is gone
depends on the deployment), but "a page that has already killed five agents
will not be run by a sixth" is a property of the queue, not the deployment.
"""


def poison_page_error(max_claim_attempts: int) -> str:
    """Reason recorded when a page exhausted its claim budget.

    Allowlisted static text, same rule as ``waiting_timeout_error``: nothing the
    page itself produced reaches the client, since no agent ever got far enough
    to report anything. Naming the count tells the researcher the page is the
    problem rather than the platform.
    """
    return f"{POISON_PAGE_ERROR} (abandoned {max_claim_attempts} times)"


def waiting_timeout_error(waiting_timeout_seconds: float) -> str:
    """Reason recorded when the inference service never called back.

    Allowlisted static text, same rule as ``worker._public_job_error``: no
    exception detail reaches the client. Naming the deadline lets the UI say
    "inference went silent for 240s" instead of a generic error. The
    ``WAITING_TIMEOUT_ERROR`` prefix stays stable so callers can still recognise
    the timeout.

    Currently unreachable in production, see ``fail_stale_waiting_jobs``.
    """
    return f"{WAITING_TIMEOUT_ERROR} after {int(waiting_timeout_seconds)}s"


# Prefix ``jobs.claimed_by`` carries while an inference agent holds a page,
# written by ``job_claim_service.agent_claim_owner``. Lives here, above the
# platform worker's own identity, because neither may ever produce the other's
# shape, see ``worker_identity``.
AGENT_CLAIM_PREFIX = "agent:"

# ``worker:`` namespaces the platform worker's id the way ``agent:`` namespaces a
# device's. Without it the identity was ``{hostname}:{pid}``, and a host called
# ``agent`` would produce ``agent:12345``, which ``_held_by_agent()`` classifies
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


def _held_by_agent():
    """SQL for "an inference agent holds this page"."""
    return Job.claimed_by.startswith(AGENT_CLAIM_PREFIX)


def _not_held_by_agent():
    """SQL for the complement, NULL included.

    ``NOT LIKE`` alone evaluates to NULL for an unclaimed row and would silently
    exempt every job the platform itself dispatched, which is the whole
    population the waiting timeout exists for.
    """
    return or_(Job.claimed_by.is_(None), ~Job.claimed_by.startswith(AGENT_CLAIM_PREFIX))


def _pending_job_query(*, test_only: bool | None = None):
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
        query = (
            select(func.count())
            .select_from(Job)
            .where(Job.status.in_((JobStatus.pending, JobStatus.running, JobStatus.waiting)))
        )
        if test_payload is True:
            query = query.where(Job.payload.contains({"test": True}))
        elif test_payload is False:
            query = query.where(~Job.payload.contains({"test": True}))
        return session.execute(query).scalar_one()


def _split_by_claim_budget(
    rows: list, max_claim_attempts: int
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """Split ``(id, claim_attempts)`` rows into (retryable, exhausted).

    The comparison is against ``attempts + 1`` because the caller is about to
    record *this* abandonment: a row already at the ceiling would otherwise be
    re-pended one last time and only fail on the lap after.
    """
    retryable: list[uuid.UUID] = []
    exhausted: list[uuid.UUID] = []
    for job_id, attempts in rows:
        target = exhausted if (attempts or 0) + 1 >= max_claim_attempts else retryable
        target.append(job_id)
    return retryable, exhausted


def reclaim_stale_running_jobs(
    *, running_timeout_seconds: float, max_claim_attempts: int = MAX_CLAIM_ATTEMPTS
) -> int:
    """Take back a crashed worker's claim once its running lease expires.

    Almost always that means re-pending, so another worker runs the job. A job
    whose claim has been abandoned ``max_claim_attempts`` times is failed
    instead, see ``MAX_CLAIM_ATTEMPTS``.

    ``FOR UPDATE SKIP LOCKED`` on the select, same primitive the other sweeps
    use. It's not just concurrency control here: the split below needs each
    row's ``claim_attempts``, and an unlocked read would let a second sweeper
    decide from the same stale count.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=running_timeout_seconds)
    with sync_system_session() as session:
        rows = list(
            session.execute(
                select(Job.id, Job.claim_attempts)
                .where(Job.status == JobStatus.running)
                .where(Job.started_at <= stale_before)
                .with_for_update(skip_locked=True)
            )
        )
        if not rows:
            return 0
        retryable, exhausted = _split_by_claim_budget(rows, max_claim_attempts)
        moved = 0
        if retryable:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(retryable))
                    .where(Job.status == JobStatus.running)
                    .values(
                        status=JobStatus.pending,
                        started_at=None,
                        claimed_by=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        if exhausted:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(exhausted))
                    .where(Job.status == JobStatus.running)
                    .values(
                        status=JobStatus.failed,
                        error=poison_page_error(max_claim_attempts),
                        claimed_by=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        completed_at=now,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so an SSE subscriber would sit on
        # a job that just died. Re-pended rows are deliberately not announced:
        # pending isn't a state anything renders differently from the running
        # it replaces.
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return moved


def fail_stale_waiting_jobs(*, waiting_timeout_seconds: float) -> int:
    """Fail jobs stuck in ``waiting`` because the inference callback never arrived.

    Sweeps an empty population today. Its subject is a job the platform
    dispatched to an inference service over HTTP, and that hop is gone (ADR
    0003). Since then the only writer of ``waiting`` is ``claim_one_page``,
    which always stamps ``agent:<device_id>``, so ``_not_held_by_agent()``
    excludes every waiting row by construction. Kept because the integration
    suite still pins this sweep's behaviour on hand-built rows.

    Unlike a crashed running job there's nothing to retry: the dispatch already
    happened and the inference service went silent, so re-pending would
    duplicate work. Fail instead, with an error the user can act on.

    Agent-held pages are deliberately excluded, which (with the dispatch hop
    gone) is why the population is empty rather than merely small. ADR 0005
    makes a claimed page ``waiting`` so ``JobCallbackService`` needs no change,
    which put a researcher's laptop under this timeout by accident. The
    reasoning above doesn't hold there: nothing was dispatched anywhere, the
    page never left the queue, and a closed lid isn't a silent inference
    service. Those rows are governed by ``release_expired_device_leases``,
    which re-pends instead of failing, see ``AGENT_CLAIM_PREFIX``. Every
    waiting row is one of those today.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=waiting_timeout_seconds)
    with sync_system_session() as session:
        stale_ids = list(
            session.execute(
                select(Job.id)
                .where(Job.status == JobStatus.waiting)
                .where(_not_held_by_agent())
                .where(Job.updated_at <= stale_before)
                .with_for_update(skip_locked=True)
            ).scalars()
        )
        if not stale_ids:
            return 0
        result = session.execute(
            update(Job)
            .where(Job.id.in_(stale_ids))
            .where(Job.status == JobStatus.waiting)
            .where(_not_held_by_agent())
            .values(
                status=JobStatus.failed,
                error=waiting_timeout_error(waiting_timeout_seconds),
                callback_claimed_at=None,
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so SSE subscribers would keep
        # waiting on a job that just died. Notify each id after the commit.
        for job_id in stale_ids:
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return result.rowcount or 0


def release_expired_device_leases(
    *, lease_seconds: float, max_claim_attempts: int = MAX_CLAIM_ATTEMPTS
) -> int:
    """Return pages whose lease expired to the queue, so another agent can take them.

    A crash, a killed process, or a closed laptop lid leaves a page held by an
    agent that will never report on it. There's no heartbeat to notice (work is
    seconds to minutes, so the lease covers it with margin, ADR 0002) and no
    release endpoint, since a killed process can't call one. This sweep is the
    whole recovery mechanism.

    It re-pends, not fails, unlike ``fail_stale_waiting_jobs``: a researcher who
    closes their lid mid-page should have it picked up by the next agent, not
    see it permanently failed. Nothing was dispatched to a third party, so
    there's no duplicate work to fear, the page is still in the platform's own
    queue.

    Except after ``max_claim_attempts`` laps: re-pending is right when the agent
    is why the page came back, but a loop when the page is. A scan the model
    crashes on takes down every agent that claims it, so with no counter it
    would cycle ``pending -> waiting -> pending`` forever, never terminal, never
    reported to the researcher, holding a claim slot each lap. The counter lives
    on the row (``jobs.claim_attempts``) so this bulk statement can read and
    bump it without a per-row round trip. See ``MAX_CLAIM_ATTEMPTS``.

    The claim is cleared with it (``claimed_by``, ``inference_job_id``,
    ``started_at``, ``heartbeat_at``) so any agent may take the page next, and a
    woken zombie can't report on it: the callback route matches ``claimed_by``,
    and ``JobCallbackService`` matches ``inference_job_id``. Neither matches
    once this has run.

    Concurrency uses the same primitive as the other sweeps: ``FOR UPDATE SKIP
    LOCKED`` in the same transaction as the update means a second sweeper skips
    rows already being released rather than releasing them twice, and the
    ``status``/``claimed_by`` predicates repeat on the update so a row that
    changed hands between the two statements is left alone.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=lease_seconds)
    with sync_system_session() as session:
        rows = list(
            session.execute(
                select(Job.id, Job.claim_attempts)
                .where(Job.status == JobStatus.waiting)
                .where(_held_by_agent())
                .where(Job.updated_at <= stale_before)
                .with_for_update(skip_locked=True)
            )
        )
        if not rows:
            return 0
        retryable, exhausted = _split_by_claim_budget(rows, max_claim_attempts)
        moved = 0
        if retryable:
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(retryable))
                    .where(Job.status == JobStatus.waiting)
                    .where(_held_by_agent())
                    .values(
                        status=JobStatus.pending,
                        claimed_by=None,
                        inference_job_id=None,
                        started_at=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        if exhausted:
            # Claim cleared here too, same reason as the re-pend: a woken zombie
            # must not be able to report on the page. Terminal status alone
            # wouldn't stop it, since the callback path matches
            # ``inference_job_id``, not ``status``.
            moved += (
                session.execute(
                    update(Job)
                    .where(Job.id.in_(exhausted))
                    .where(Job.status == JobStatus.waiting)
                    .where(_held_by_agent())
                    .values(
                        status=JobStatus.failed,
                        error=poison_page_error(max_claim_attempts),
                        claimed_by=None,
                        inference_job_id=None,
                        started_at=None,
                        heartbeat_at=None,
                        claim_attempts=Job.claim_attempts + 1,
                        callback_claimed_at=None,
                        completed_at=now,
                        updated_at=now,
                    )
                ).rowcount
                or 0
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so a browser watching the job
        # would sit on "waiting" until something else touched the row.
        for job_id in retryable:
            notify_platform_job_status_changed(job_id, JobStatus.pending)
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
            notify_platform_job_status_changed(job_id, JobStatus.failed)
        return moved


def clear_stale_callback_claims(*, claim_timeout_seconds: float) -> int:
    """Release abandoned inference callback claims on non-terminal jobs.

    A callback that crashes after claiming never clears ``callback_claimed_at``,
    which makes the job permanently uncancellable. Dropping the stale claim
    hands control back to the user.
    """
    now = datetime.now(UTC)
    stale_before = now - timedelta(seconds=claim_timeout_seconds)
    with sync_system_session() as session:
        result = session.execute(
            update(Job)
            .where(Job.status.in_(_NON_TERMINAL_STATUSES))
            .where(Job.callback_claimed_at.is_not(None))
            .where(Job.callback_claimed_at <= stale_before)
            # Keep ``updated_at`` as-is: the column's ``onupdate`` would otherwise
            # bump it and push the waiting-timeout deadline back another window.
            .values(callback_claimed_at=None, updated_at=Job.updated_at)
        )
        session.commit()
        return result.rowcount or 0


def seconds_until_next_stale_waiting_job(*, waiting_timeout_seconds: float) -> float | None:
    """Return seconds until the oldest waiting job is eligible for the timeout sweep.

    Note it counts *every* waiting row, agent-held ones included, while the sweep
    it paces only fails the non-agent ones. That is deliberate and harmless - it
    can only wake the worker early - but it means a non-None answer here is not
    evidence that ``fail_stale_waiting_jobs`` has anything to do. In practice it
    never does; see ``fail_stale_waiting_jobs``.
    """
    with sync_system_session() as session:
        oldest_updated_at = session.execute(
            select(func.min(Job.updated_at)).where(Job.status == JobStatus.waiting)
        ).scalar_one_or_none()
    if oldest_updated_at is None:
        return None

    now = datetime.now(UTC)
    fail_at = oldest_updated_at + timedelta(seconds=waiting_timeout_seconds)
    return max((fail_at - now).total_seconds(), 0.0)


def _owned_by(statement, claimed_by: str | None):
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
    if update_result.rowcount:
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
    if update_result.rowcount:
        notify_platform_job_status_changed(job_id, JobStatus.done)


_CANCELABLE = _NON_TERMINAL_STATUSES


def _apply_cancellation(job: Job, now: datetime) -> None:
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
    if job.status not in _CANCELABLE:
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
