"""Platform health — public HTTP interface via FastAPI TestClient."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from sqlalchemy import text

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.api import health as health_module
from backend.core.schema_revision import DEFAULT_SCRIPT_LOCATION, code_revisions
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import sync_system_session

pytestmark = pytest.mark.integration


# Every seeded row is a ``segment`` job, the same rule as
# ``test_job_worker_sweeps.py``: the lifespan worker polls every 250 ms and
# ``claim_next_pending_job`` skips the agent-claimed types (ADR 0003), so a row
# this file seeds as pending is still pending when the assertion reads it. No
# inference agent is connected in these tests, so nothing else claims it either.
def _seed_pending_job(*, age_seconds: float) -> uuid.UUID:
    """Insert one pending job whose ``created_at`` is already the age under test.

    Written on the INSERT rather than moved by a later UPDATE, so nothing can
    overwrite the age this row is supposed to have. The deadline in play is 900
    seconds by default; a test that waited for one is a test nobody runs.
    """
    job_id = uuid.uuid4()
    created_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=JobStatus.pending,
                payload={},
                created_at=created_at,
                updated_at=created_at,
            )
        )
        session.commit()
    return job_id


# --- Health with database ---
# Tests /health when Postgres is reachable. Does not test inference service health.


def test_health_returns_ok_when_database_is_reachable(client: TestClient):
    """GET /health reports ok when Postgres (kalamos) accepts a connection."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    # An empty queue is null, not 0: "nothing is pending" and "something has been
    # pending for no time at all" are different answers and only one of them is
    # ever true of a queue with no rows.
    head = code_revisions().head
    assert body == {
        "status": "ok",
        "database": "ok",
        "oldest_pending_job_seconds": None,
        "schema": "ok",
        "schema_revision": head,
        "expected_schema_revision": head,
    }


def _set_database_revision(revision: str) -> None:
    with sync_system_session() as session:
        session.execute(text("UPDATE alembic_version SET version_num = :rev"), {"rev": revision})
        session.commit()


def test_health_returns_503_when_the_schema_is_behind_the_code(client: TestClient):
    """A database one migration behind the shipped head takes the API out of rotation.

    The 2026-11-04 incident: code needing ``011`` deployed against a database at
    ``010`` and ``/health`` kept answering ok. ``alembic_version`` is rewound to the
    head's parent for this one request and restored in ``finally``, because the
    whole suite shares the ``kalamos`` database.
    """
    head = code_revisions().head
    parent = ScriptDirectory(str(DEFAULT_SCRIPT_LOCATION)).get_revision(head).down_revision
    assert isinstance(parent, str)
    with sync_system_session() as session:
        original = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

    _set_database_revision(parent)
    try:
        response = client.get("/health")
    finally:
        _set_database_revision(original)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "ok"
    assert body["schema"] == "behind"
    assert body["schema_revision"] == parent
    assert body["expected_schema_revision"] == head


# --- The queue's only alarm ---
# Nothing in the API deployment claims a pending job: the platform worker runs on
# a separate host under JOB_WORKER_ENABLED and segment/transcribe are claimed by
# an inference agent over HTTP. If neither is up, jobs sit in pending and every
# other signal stays green. These tests pin the number that says so.


def test_health_reports_the_age_of_the_oldest_pending_job(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    """The oldest row wins, and an ordinary queue is not an alarm."""
    _seed_pending_job(age_seconds=120)
    _seed_pending_job(age_seconds=30)

    with caplog.at_level(logging.WARNING, logger=health_module.__name__):
        response = client.get("/health")

    assert response.status_code == 200
    age = response.json()["oldest_pending_job_seconds"]
    assert age is not None
    # The 120s row, not the 30s one - the head of the queue is what has been
    # waiting, and a max() here would go quiet exactly as a backlog built up.
    assert 120 <= age < 180
    # Well inside the 900s threshold, so a queue that is merely busy must not
    # page anyone; a warning here would train operators to ignore the real one.
    assert caplog.records == []


def test_health_warns_but_still_returns_200_when_the_queue_has_stalled(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    """A stalled queue is loud in the logs and invisible to the load balancer.

    Both halves matter. The WARNING is the whole point - nobody reads a probe's
    body, so without it the queue stalls in silence. And the 200 is the other
    half: pending jobs pile up because a *different* host stopped claiming, so a
    503 would pull a healthy API out of rotation without bringing that host back.
    """
    threshold = get_job_settings().job_queue_stall_warning_seconds
    _seed_pending_job(age_seconds=threshold * 2)

    with caplog.at_level(logging.WARNING, logger=health_module.__name__):
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["oldest_pending_job_seconds"] >= threshold

    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert "oldest pending job has waited" in warnings[0]


def test_the_stall_threshold_is_configurable(
    client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """A deployment can move the threshold; the default is not the mechanism.

    Read per request rather than captured at import, so this takes effect without
    a restart - and so the accessor's cache, which the autouse fixture clears
    between tests, is the only state involved.
    """
    monkeypatch.setenv("JOB_QUEUE_STALL_WARNING_SECONDS", "60")
    get_job_settings.cache_clear()
    _seed_pending_job(age_seconds=120)

    with caplog.at_level(logging.WARNING, logger=health_module.__name__):
        response = client.get("/health")

    assert response.status_code == 200
    # 120s is silent under the 900s default and an alarm under this one.
    assert len(caplog.records) == 1
    assert "threshold 60s" in caplog.records[0].getMessage()
