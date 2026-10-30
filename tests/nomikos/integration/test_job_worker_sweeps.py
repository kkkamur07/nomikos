"""``process_one_job``'s stale sweeps, against live Postgres.

The unit test that used to stand in for this file patched four of the five
repository calls and asserted the *order* they were made in. That is a test of
the call list, not of recovery: every sweep's actual population is decided by a
``WHERE`` clause, and a mocked sweep has no ``WHERE`` clause. It also left one
call unpatched, so it opened a real connection from the unit lane - which has no
database - and was red. The ordering assertion it was reaching for is cheap and
now lives in ``tests/nomikos/unit/test_job_lifecycle.py`` with **all five**
calls patched; the recovery behaviour lives here, where there are rows.

**Expiry is created by writing the timestamp, not by waiting for it.** Same rule
as ``test_device_lease.py``: the shortest deadline in play is 240 seconds, so a
test that waited for one is a test nobody runs, and a patched clock would prove
something about the patch rather than about the SQL Postgres executes.

**Every row here is a ``segment`` job.** No sweep filters on type, and
``claim_next_pending_job`` skips the agent-claimed types (ADR 0003), so the
lifespan worker - which is always running and polls every 250 ms - cannot claim
a row this test just re-pended and race the assertion about it. The one thing
type controls is the claim, and the claim is not what is under test.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings.device import get_device_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.application.job_claim_service import agent_claim_owner
from backend.jobs.infrastructure import worker
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import sync_system_session

pytestmark = pytest.mark.integration


def _ago(seconds: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


def _seed_job(
    *,
    status: JobStatus,
    started_at: datetime | None = None,
    updated_at: datetime | None = None,
    claimed_by: str | None = None,
    inference_job_id: uuid.UUID | None = None,
    callback_claimed_at: datetime | None = None,
) -> uuid.UUID:
    """Insert one job with its clock already where the test needs it.

    The timestamps are written on the INSERT rather than moved afterwards by an
    UPDATE, so ``updated_at``'s ``onupdate`` never gets a chance to overwrite the
    age this row is supposed to have.
    """
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=status,
                payload={},
                started_at=started_at,
                updated_at=updated_at or datetime.now(UTC),
                claimed_by=claimed_by,
                inference_job_id=inference_job_id,
                callback_claimed_at=callback_claimed_at,
            )
        )
        session.commit()
    return job_id


def _stored(job_id: uuid.UUID) -> Job:
    with sync_system_session() as session:
        job = session.execute(select(Job).where(Job.id == job_id)).scalar_one()
        session.expunge(job)
        return job


# The worker is asked to do a full tick, exactly as ``worker_loop`` asks it to.
# Calling the individual repository functions would test the same SQL while
# leaving the thing this file is about - that a tick runs *all* of them -
# unproven, which is how the lease sweep came to be missing from the old
# ordering assertion in the first place.
def _tick() -> None:
    worker.process_one_job()


# ---------------------------------------------------------------------------
# A crashed worker's job goes back to the queue
# ---------------------------------------------------------------------------


def test_a_job_running_past_the_deadline_is_returned_to_pending(client: TestClient):
    timeout = get_job_settings().job_worker_running_timeout_seconds
    job_id = _seed_job(
        status=JobStatus.running,
        started_at=_ago(timeout * 2),
        updated_at=_ago(timeout * 2),
        claimed_by="host:4242",
    )

    _tick()

    job = _stored(job_id)
    assert job.status is JobStatus.pending
    # The claim has to go with it, or the next worker to pick the job up writes
    # terminal state scoped to an owner that no longer holds it.
    assert job.claimed_by is None
    assert job.started_at is None
    assert job.heartbeat_at is None


def test_a_job_running_inside_the_deadline_is_left_alone(client: TestClient):
    timeout = get_job_settings().job_worker_running_timeout_seconds
    job_id = _seed_job(
        status=JobStatus.running,
        started_at=_ago(timeout / 2),
        claimed_by="host:4242",
    )

    _tick()

    # Without the deadline predicate every in-flight job is torn off its worker
    # on the next tick and the queue never finishes anything.
    job = _stored(job_id)
    assert job.status is JobStatus.running
    assert job.claimed_by == "host:4242"


# ---------------------------------------------------------------------------
# The two halves of ``waiting`` recover differently - and that is the point
# ---------------------------------------------------------------------------


def test_a_slept_laptop_re_pends_and_an_unclaimed_row_is_left_alone(client: TestClient):
    """Both rows are ``waiting`` and both are past the lease.

    Only the agent-held one is under the lease, so a sweep that dropped the agent
    requirement would take both rows and this fails.
    """
    lease = get_device_settings().device_lease_seconds
    unclaimed_id = _seed_job(
        status=JobStatus.waiting,
        claimed_by=None,
        updated_at=_ago(lease * 2),
    )
    leased_inference_id = uuid.uuid4()
    leased_id = _seed_job(
        status=JobStatus.waiting,
        claimed_by=agent_claim_owner(uuid.uuid4()),
        inference_job_id=leased_inference_id,
        updated_at=_ago(lease * 2),
    )

    _tick()

    unclaimed = _stored(unclaimed_id)
    assert unclaimed.status is JobStatus.waiting
    assert unclaimed.error is None

    leased = _stored(leased_id)
    assert leased.status is JobStatus.pending
    assert leased.error is None
    # Cleared so *any* agent may take the page next, and so a woken zombie cannot
    # report on it: the callback route matches ``claimed_by`` and the callback
    # service matches ``inference_job_id``. Neither matches now.
    assert leased.claimed_by is None
    assert leased.inference_job_id is None
    assert leased.started_at is None
    assert leased.heartbeat_at is None


# ---------------------------------------------------------------------------
# An abandoned callback claim is released without moving the deadline
# ---------------------------------------------------------------------------


def test_an_abandoned_callback_claim_is_released_without_touching_the_row_clock(
    client: TestClient,
):
    claim_timeout = get_job_settings().job_worker_callback_claim_timeout_seconds
    seeded_updated_at = _ago(5)
    job_id = _seed_job(
        status=JobStatus.running,
        started_at=_ago(5),
        updated_at=seeded_updated_at,
        callback_claimed_at=_ago(claim_timeout * 2),
    )

    _tick()

    job = _stored(job_id)
    # A callback that crashed after claiming leaves the job permanently
    # uncancellable; dropping the claim hands control back to the user.
    assert job.callback_claimed_at is None
    assert job.status is JobStatus.running
    # ...and releasing it must not count as activity. Letting ``onupdate`` bump
    # this pushes the lease deadline out by another full window every time the
    # sweep runs, so a page with a stale claim is never released at all.
    assert abs((job.updated_at - seeded_updated_at).total_seconds()) < 0.001
