"""Stale-job sweeps: re-pend or fail work whose claim, lease or callback went quiet.

Split out of ``job_claim_engine``. Each sweep runs on its own short-lived
``sync_system_session()`` from the worker loop and from the on-read stale sweep,
and each uses ``FOR UPDATE SKIP LOCKED`` so concurrent sweepers never decide
from the same row.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from sqlalchemy import Row, select, update

from backend.jobs.infrastructure.job_sql import NON_TERMINAL_STATUSES, held_by_agent
from backend.jobs.infrastructure.notifications import notify_platform_job_statuses_changed
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from infrastructure.db import rows_affected, sync_system_session

logger = logging.getLogger(__name__)

POISON_PAGE_ERROR = "This page could not be completed by any inference agent"

MAX_CLAIM_ATTEMPTS = 5
"""How many abandoned claims a page survives before it is failed.

Both stale sweeps re-pend an abandoned job by default (a closed laptop lid is
not a failed job), but a page that reliably kills whatever runs it (a corrupt
scan, an image the model crashes on) would otherwise loop forever, burning one
agent slot per lap and never telling the researcher anything.
``jobs.claim_attempts`` counts the laps and this is where they stop.

A constant rather than an environment dial, unlike the running timeout and the
device lease:
those are operational (how long to wait before believing an agent is gone
depends on the deployment), but "a page that has already killed five agents
will not be run by a sixth" is a property of the queue, not the deployment.
"""


def poison_page_error(max_claim_attempts: int) -> str:
    """Reason recorded when a page exhausted its claim budget.

    Allowlisted static text, same rule as ``worker._public_job_error``: nothing the
    page itself produced reaches the client, since no agent ever got far enough
    to report anything. Naming the count tells the researcher the page is the
    problem rather than the platform.
    """
    return f"{POISON_PAGE_ERROR} (abandoned {max_claim_attempts} times)"


def _split_by_claim_budget(
    rows: Iterable[Row[tuple[uuid.UUID, int]]], max_claim_attempts: int
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
            moved += rows_affected(
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
                )
            )
        if exhausted:
            moved += rows_affected(
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
                )
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so an SSE subscriber would sit on
        # a job that just died. Re-pended rows are deliberately not announced:
        # pending isn't a state anything renders differently from the running
        # it replaces.
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
        notify_platform_job_statuses_changed((job_id, JobStatus.failed) for job_id in exhausted)
        return moved


def release_expired_device_leases(
    *, lease_seconds: float, max_claim_attempts: int = MAX_CLAIM_ATTEMPTS
) -> int:
    """Return pages whose lease expired to the queue, so another agent can take them.

    A crash, a killed process, or a closed laptop lid leaves a page held by an
    agent that will never report on it. There's no heartbeat to notice (work is
    seconds to minutes, so the lease covers it with margin, ADR 0002) and no
    release endpoint, since a killed process can't call one. This sweep is the
    whole recovery mechanism.

    It re-pends rather than fails: a researcher who
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
                .where(held_by_agent())
                .where(Job.updated_at <= stale_before)
                .with_for_update(skip_locked=True)
            )
        )
        if not rows:
            return 0
        retryable, exhausted = _split_by_claim_budget(rows, max_claim_attempts)
        moved = 0
        if retryable:
            moved += rows_affected(
                session.execute(
                    update(Job)
                    .where(Job.id.in_(retryable))
                    .where(Job.status == JobStatus.waiting)
                    .where(held_by_agent())
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
                )
            )
        if exhausted:
            # Claim cleared here too, same reason as the re-pend: a woken zombie
            # must not be able to report on the page. Terminal status alone
            # wouldn't stop it, since the callback path matches
            # ``inference_job_id``, not ``status``.
            moved += rows_affected(
                session.execute(
                    update(Job)
                    .where(Job.id.in_(exhausted))
                    .where(Job.status == JobStatus.waiting)
                    .where(held_by_agent())
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
                )
            )
        session.commit()
        # A bulk update emits no per-job NOTIFY, so a browser watching the job
        # would sit on "waiting" until something else touched the row.
        for job_id in exhausted:
            logger.warning("failing job %s: claim abandoned %s times", job_id, max_claim_attempts)
        notify_platform_job_statuses_changed(
            [(job_id, JobStatus.pending) for job_id in retryable]
            + [(job_id, JobStatus.failed) for job_id in exhausted]
        )
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
            .where(Job.status.in_(NON_TERMINAL_STATUSES))
            .where(Job.callback_claimed_at.is_not(None))
            .where(Job.callback_claimed_at <= stale_before)
            # Keep ``updated_at`` as-is: the column's ``onupdate`` would otherwise
            # bump it and push the device-lease deadline back another window.
            .values(callback_claimed_at=None, updated_at=Job.updated_at)
        )
        session.commit()
        return rows_affected(result)
