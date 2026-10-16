"""The device **lease**, end to end, against live Postgres and the real app.

Everything here goes through ``create_app()``. ADR 0001 records why that is not
negotiable: an earlier device layer was never mounted and the integration suite
hid it behind its own FastAPI app. So there is no local app in this file, no
substitute for Postgres, and the device credentials are minted by running the
real pairing protocol.

**Expiry is created by writing the timestamp, not by waiting for it.** The lease
is 600 seconds; a test that slept through one is a test nobody runs, and a
patched clock would prove something about the patch rather than about the SQL
that Postgres will actually execute in production. So the row's ``updated_at`` is
aged in the database and the real sweep is then asked what it makes of it.

The one thing this file exists to prove: an expired lease sends the page **back
to the queue**, not to ``failed``. ADR 0005 made a claimed page ``waiting`` so
``JobCallbackService`` needed no change, which quietly put a researcher's laptop
under the 240-second waiting timeout - a timeout that *fails* the job. A closed
lid is not a failed job.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings.device import get_device_settings
from backend.core.settings.job import get_job_settings
from backend.jobs.application.job_claim_service import agent_claim_owner
from backend.jobs.infrastructure import stale_sweep
from backend.jobs.infrastructure.job_sweeps import release_expired_device_leases
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.jobs.infrastructure.stale_sweep import run_stale_job_sweep
from backend.ml.api.agent_version import AGENT_VERSION_HEADER
from backend.ml.application.agent_credentials import SERVICE_TOKEN_HEADER, WORKER_NAME_HEADER
from infrastructure.db import sync_system_session

# Module-scoped autouse; see its docstring in `helpers.py` for issue #63.
from tests.nomikos.integration.helpers import (
    CALLBACK_URL,
    CLAIM_URL,
    CURRENT_AGENT_VERSION,
    DEVICE_SERVICE_TOKEN,
    claim_page,
    return_pooled_connections_before_leaving,  # noqa: F401
)
from tests.nomikos.integration.helpers import device_headers as _device_headers
from tests.nomikos.integration.helpers import make_part as _make_part
from tests.nomikos.integration.helpers import prefer_local as _prefer_local
from tests.nomikos.integration.helpers import running_agent as _running_agent
from tests.nomikos.integration.helpers import stored_job as _stored_job
from tests.nomikos.integration.helpers import submit_segment as _submit_segment

pytestmark = pytest.mark.integration

# Every claim states which agent is calling (issue 055); one that does not is
# refused before it is authenticated. `CURRENT_AGENT_VERSION` is comfortably
# above the configured floor - the floor itself is tested in
# ``test_agent_version_floor.py``.
SERVICE_HEADERS = {
    SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN,
    AGENT_VERSION_HEADER: CURRENT_AGENT_VERSION,
    # Required: without it two hosted workers resolve to one helper_devices row
    # and neither can be told from the other on a claim.
    WORKER_NAME_HEADER: "cloud-worker",
}


@pytest.fixture(autouse=True)
def _clean_sweep_throttle():
    """The throttle is process-global; never let it leak between tests."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stale_sweep, "_last_sweep_at", None)
        yield


# ---------------------------------------------------------------------------
# Live fixtures: real pairing, real capacity, real jobs
# ---------------------------------------------------------------------------


def _claim(client: TestClient, token: str):
    """Claims in this module are written token-first: every test here is about
    *which agent* holds a page, so the token is the interesting argument."""
    return claim_page(client, _device_headers(token))


def _claim_allowed_to_sweep(client: TestClient, token: str):
    """A claim that is not sitting inside another claim's throttle window.

    ``sweep_stale_jobs_on_read`` runs at most once per process per
    ``JOB_STALE_SWEEP_MIN_INTERVAL_SECONDS`` - thirty seconds - so a hot endpoint
    cannot become a sweep loop. These tests make several claims inside one
    second, so the window is opened explicitly at the point where the recovery is
    the thing under test. That bounds detection lag in production by the
    throttle, not by the lease, and the throttle has its own tests; asserting on
    it here would be measuring the wrong mechanism.
    """
    stale_sweep._last_sweep_at = None
    return _claim(client, token)


def _age_by(job_id: str | uuid.UUID, seconds: float) -> None:
    """Move the row's clock back, in the database, by writing ``updated_at``.

    This is what a laptop that went to sleep ``seconds`` ago looks like to
    Postgres. ``updated_at`` is the column both waiting sweeps compare against,
    and an explicit value overrides the ORM's ``onupdate``.
    """
    key = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)
    aged = datetime.now(UTC) - timedelta(seconds=seconds)
    with sync_system_session() as session:
        session.execute(update(Job).where(Job.id == key).values(updated_at=aged))
        session.commit()


def _leased_page(
    client: TestClient, headers: dict[str, str], project: dict, *, agent_name: str = "Laptop"
) -> tuple[dict, dict]:
    """One agent holding one real page, claimed over HTTP."""
    agent = _running_agent(client, headers, agent_name)
    _prefer_local(client, headers)
    ids = _make_part(client, headers, project)
    _submit_segment(client, headers, project, ids)
    page = _claim(client, agent["device_token"]).json()["page"]
    assert page is not None
    return agent, page


# ---------------------------------------------------------------------------
# An expired lease returns the page to the queue
# ---------------------------------------------------------------------------


def test_an_expired_lease_returns_the_page_to_a_different_agent(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The whole point of the lease.

    A researcher closes their lid mid-page. The page must be picked up by the
    next agent to ask - any agent, not the one that stopped - without the
    researcher resubmitting anything.
    """
    stopped_agent, page = _leased_page(client, owner_headers, owner_project, agent_name="Lid shut")
    job_id = page["product_job_id"]
    second_agent = _running_agent(client, owner_headers, "Second laptop")
    assert _stored_job(job_id).claimed_by == agent_claim_owner(
        uuid.UUID(stopped_agent["device_id"])
    )

    # The lid closes. Nothing else happens for longer than the lease.
    _age_by(job_id, get_device_settings().device_lease_seconds + 60)

    # The next agent to ask sweeps on the way in and then claims.
    response = _claim_allowed_to_sweep(client, second_agent["device_token"])

    assert response.status_code == 200, response.text
    handed_over = response.json()["page"]
    assert handed_over is not None, "the abandoned page was not returned to the queue"
    assert handed_over["product_job_id"] == job_id
    stored = _stored_job(job_id)
    assert stored.status is JobStatus.waiting
    assert stored.claimed_by == agent_claim_owner(uuid.UUID(second_agent["device_id"]))
    # A fresh lease, and a fresh inference_job_id: the stopped agent's callback
    # contract no longer matches this row.
    assert stored.inference_job_id != uuid.UUID(page["inference_job_id"])


def test_the_stopped_agent_cannot_report_on_a_page_it_no_longer_holds(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """A laptop that wakes up hours later must not overwrite the job.

    The same guarantee ``reclaim_stale_running_jobs`` gets from ``_owned_by``:
    the claim is cleared with the lease, so the zombie's credential no longer
    names the holder of that page.
    """
    agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    _age_by(job_id, get_device_settings().device_lease_seconds + 60)
    assert run_stale_job_sweep() == 1

    late = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json={
            "inference_job_id": page["inference_job_id"],
            "product_job_id": job_id,
            "task": "segment",
            "status": "done",
            "output": {"kind": "segment", "data": {"lines": []}},
        },
    )

    assert late.status_code == 403, late.text
    assert _stored_job(job_id).status is JobStatus.pending


# ---------------------------------------------------------------------------
# A page inside its lease is untouched
# ---------------------------------------------------------------------------


def test_a_page_past_the_waiting_timeout_but_inside_its_lease_is_untouched(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The two timeouts are distinct, and this is the window that proves it.

    At 300 seconds an agent-held page is well past the 240-second waiting
    timeout and well inside the 600-second lease. Before this issue the waiting
    sweep would have failed it here. The lease is the only deadline that governs
    an agent-held page.
    """
    settings = get_job_settings()
    device_settings = get_device_settings()
    age = (settings.job_worker_waiting_timeout_seconds + device_settings.device_lease_seconds) / 2
    assert settings.job_worker_waiting_timeout_seconds < age < device_settings.device_lease_seconds

    agent, page = _leased_page(client, owner_headers, owner_project)
    job_id = page["product_job_id"]
    _age_by(job_id, age)

    assert run_stale_job_sweep() == 0

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.waiting
    assert stored.claimed_by == agent_claim_owner(uuid.UUID(agent["device_id"]))
    # No other agent may take it either: the lease has not expired.
    other = _running_agent(client, owner_headers, "Impatient laptop")
    assert _claim(client, other["device_token"]).json()["page"] is None


# ---------------------------------------------------------------------------
# The device lease is distinct from, and shorter than, the global job timeout
# ---------------------------------------------------------------------------


def test_the_device_lease_is_shorter_than_the_global_job_timeout() -> None:
    """1800 seconds is right for a server that does not sleep and wrong for a
    laptop that does. A closed lid must not hold a page for half an hour."""
    device_settings = get_device_settings()
    settings = get_job_settings()

    assert device_settings.device_lease_seconds == 600
    assert device_settings.device_lease_seconds < settings.job_worker_running_timeout_seconds
    assert settings.job_worker_running_timeout_seconds == 1800
    # And distinct from the waiting timeout, which governs the other population.
    assert device_settings.device_lease_seconds != settings.job_worker_waiting_timeout_seconds


def test_a_hosted_worker_inherits_the_same_lease(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """One lease, both credentials.

    A server that does not sleep will never trip it, but a hosted worker that is
    redeployed mid-page must not strand that page any more than a laptop does -
    and the recovery is the same sweep, not a cloud-specific one.
    """
    first = client.post(
        CLAIM_URL,
        headers={**SERVICE_HEADERS, WORKER_NAME_HEADER: "cloud-worker"},
        json={"wait_seconds": 0},
    )
    assert first.status_code == 200, first.text
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    page = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0}).json()["page"]
    assert page is not None and page["execution_target"] == "cloud"

    _age_by(job_id, get_device_settings().device_lease_seconds + 60)

    assert run_stale_job_sweep() == 1

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.pending
    assert stored.status is not JobStatus.failed
    assert stored.claimed_by is None
    # And the next hosted worker to ask gets it.
    reclaimed = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})
    assert reclaimed.json()["page"]["product_job_id"] == job_id


# ---------------------------------------------------------------------------
# Concurrent sweeps do not double-release or corrupt job state
# ---------------------------------------------------------------------------


def _seed_expired_agent_pages(count: int) -> list[uuid.UUID]:
    """``count`` pages held by a stopped agent, each already past its lease."""
    device_id = uuid.uuid4()
    aged = datetime.now(UTC) - timedelta(seconds=get_device_settings().device_lease_seconds + 60)
    job_ids = [uuid.uuid4() for _ in range(count)]
    with sync_system_session() as session:
        for job_id in job_ids:
            session.add(
                Job(
                    id=job_id,
                    type=JobType.segment,
                    status=JobStatus.waiting,
                    payload={"handler": "noop"},
                    inference_job_id=uuid.uuid4(),
                    claimed_by=agent_claim_owner(device_id),
                    started_at=aged,
                    heartbeat_at=aged,
                    updated_at=aged,
                )
            )
        session.commit()
    return job_ids


def test_concurrent_lease_releases_never_double_release_a_page() -> None:
    """Eight sweepers, one queue, real Postgres.

    The advisory lock in ``run_stale_job_sweep`` normally serializes replicas, so
    this deliberately goes underneath it and calls the release directly from
    eight threads at once. If ``FOR UPDATE SKIP LOCKED`` and the repeated
    ``status``/``claimed_by`` predicates were not doing their job, the counts
    would sum to more than the number of pages.
    """
    job_ids = _seed_expired_agent_pages(12)

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(
            pool.map(
                lambda _: release_expired_device_leases(
                    lease_seconds=get_device_settings().device_lease_seconds
                ),
                range(8),
            )
        )

    # Every page released exactly once, across all eight sweepers.
    assert sum(counts) == len(job_ids), f"pages were released more than once: {counts}"
    with sync_system_session() as session:
        rows = session.execute(select(Job).where(Job.id.in_(job_ids))).scalars().all()
    assert len(rows) == len(job_ids)
    for row in rows:
        assert row.status is JobStatus.pending
        assert row.claimed_by is None
        assert row.inference_job_id is None
        assert row.error is None
        assert row.completed_at is None
