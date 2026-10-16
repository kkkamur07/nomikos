"""Product job runner — enqueue, status, worker loop, failure persistence."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import infrastructure.models  # noqa: F401 — register all ORM mappers
from backend.jobs.application.inference_dispatcher import build_page_run
from backend.jobs.infrastructure.job_claim_engine import claim_next_pending_job
from backend.jobs.infrastructure.job_sql import NON_TERMINAL_STATUSES
from backend.jobs.infrastructure.job_sweeps import reclaim_stale_running_jobs
from backend.jobs.infrastructure.job_terminal import mark_job_done
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.jobs.infrastructure.worker import execute_claimed_job
from infrastructure.db import sync_system_session
from tests.nomikos.integration.helpers import (
    MINIMAL_PNG,
    documents_url,
    pair_inference_device,
    poll_job,
    user_id_for_email,
)

pytestmark = pytest.mark.integration


def _count_active_test_jobs() -> int:
    """Pending, running or waiting jobs this suite queued (payload test=true)."""
    with sync_system_session() as session:
        return session.execute(
            select(func.count())
            .select_from(Job)
            .where(Job.status.in_(NON_TERMINAL_STATUSES))
            .where(Job.payload.contains({"test": True}))
        ).scalar_one()


def _wait_until_no_active_test_jobs(*, timeout: float = 5.0) -> None:
    """Wait until the lifespan worker finished API test jobs (payload test=true)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _count_active_test_jobs() == 0:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"test jobs still active after {timeout}s (count={_count_active_test_jobs()})"
    )


@pytest.fixture(autouse=True)
def _drain_active_test_jobs() -> None:
    """Let the lifespan worker finish API test jobs before parent conftest truncates."""
    yield
    _wait_until_no_active_test_jobs(timeout=5.0)


def _create_document_part_with_lines(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
) -> tuple[str, str, list[dict]]:
    project_id = owner_project["id"]
    base = documents_url(project_id)
    create = client.post(base, headers=owner_headers, json={"name": "Transcribe batch job"})
    assert create.status_code == 201
    document_id = create.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201
    part_id = upload.json()["id"]
    replace = client.put(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={
            "lines": [
                {
                    "order": 0,
                    "kind": "polygon",
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "source": "manual",
                },
                {
                    "order": 1,
                    "kind": "polygon",
                    "points": [[0, 1], [1, 1], [1, 2], [0, 2]],
                    "source": "manual",
                },
            ]
        },
    )
    assert replace.status_code == 200
    return document_id, part_id, replace.json()


# --- Jobs API auth ---
# Tests /jobs endpoints require a logged-in user. Does not test job execution.


def test_enqueue_test_job_requires_auth(client: TestClient):
    response = client.post("/jobs/test")
    assert response.status_code == 401


def test_get_job_requires_auth(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 401


def test_job_events_requires_auth(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}/events")
    assert response.status_code == 401


# --- Test job lifecycle ---
# Tests enqueue, poll, and status fields for noop handler. Does not test real ML jobs.


def test_enqueue_test_job_returns_job_id_immediately(
    client: TestClient, registered_user: dict[str, str]
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", headers=auth_headers)

    assert response.status_code == 201
    job_id = response.json()["job_id"]
    uuid.UUID(job_id)

    # Polling to `done` is itself the proof that the lifespan worker claimed and
    # ran the job: nothing in this test advances it.
    body = poll_job(client, job_id, expect_status="done", headers=auth_headers, timeout=8.0)
    assert body["type"] == "pipeline"
    assert body["payload"]["handler"] == "noop"
    assert body["result"] == {"ok": True}
    assert body["error"] is None
    assert body["started_at"] is not None
    assert body["completed_at"] is not None


def test_cancel_pending_job_marks_cancelled_and_discards_partials(
    client: TestClient, registered_user: dict[str, str]
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    me = client.get("/me", headers=auth_headers)
    assert me.status_code == 200
    user_id = uuid.UUID(me.json()["id"])

    # Keep the job from completing before we cancel: insert directly as pending
    # without relying on the worker race.
    with sync_system_session() as session:
        job = Job(
            type=JobType.pipeline,
            status=JobStatus.pending,
            payload={"handler": "noop", "test": True},
            user_id=user_id,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = str(job.id)

    cancelled = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["completed_at"] is not None

    again = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert again.status_code == 409


def test_cancel_rejects_when_callback_already_claimed(
    client: TestClient, registered_user: dict[str, str]
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    me = client.get("/me", headers=auth_headers)
    assert me.status_code == 200
    user_id = uuid.UUID(me.json()["id"])

    with sync_system_session() as session:
        job = Job(
            type=JobType.segment,
            status=JobStatus.waiting,
            # Avoid payload test=True so the autouse drain fixture is not blocked
            # by a permanently waiting row.
            payload={"handler": "noop"},
            user_id=user_id,
            callback_claimed_at=datetime.now(UTC),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = str(job.id)

    rejected = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert rejected.status_code == 409, rejected.text
    # *Which* conflict, not merely that there was one. A 409 is also what an
    # already-terminal job returns, so the bare status code cannot tell the
    # callback guard from the status guard - and this message is what a client
    # shows the researcher who pressed Cancel.
    error = rejected.json()["error"]
    assert error["code"] == "CONFLICT"
    assert "inference callback" in error["message"], error

    with sync_system_session() as session:
        row = session.get(Job, uuid.UUID(job_id))
        assert row is not None
        # The refusal left the row exactly as it found it. `callback_claimed_at`
        # is not re-asserted here: this test set it four lines above and nothing
        # in between could clear it, so reading it back proves only that
        # Postgres stores what it is given.
        assert row.status == JobStatus.waiting
        assert row.completed_at is None


def test_job_events_streams_current_snapshot(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]
    poll_job(client, job_id, expect_status="done", headers=auth_headers)

    with client.stream("GET", f"/jobs/{job_id}/events", headers=auth_headers) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: job" in body
    assert f'"id":"{job_id}"' in body
    assert '"status":"done"' in body


# --- Job access control ---
# Tests users can only read their own jobs. Does not test admin or shared-project access.


def test_get_job_denies_access_to_other_users_job(client: TestClient):
    owner_suffix = uuid.uuid4().hex[:8]
    other_suffix = uuid.uuid4().hex[:8]

    def _register(suffix: str) -> str:
        r = client.post(
            "/auth/register",
            json={
                "email": f"{suffix}@test.kalamos",
                "username": suffix,
                "password": "test-pass-123",
            },
        )
        return r.json()["access_token"]

    owner_token = _register(f"owner-{owner_suffix}")
    other_token = _register(f"other-{other_suffix}")

    created = client.post("/jobs/test", headers={"Authorization": f"Bearer {owner_token}"})
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}", headers={"Authorization": f"Bearer {other_token}"})
    assert response.status_code == 403


def test_get_job_denies_access_to_ownerless_job(
    client: TestClient, registered_user: dict[str, str]
):
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.pipeline,
                status=JobStatus.done,
                payload={"handler": "noop"},
                result={"ok": True},
                user_id=None,
            )
        )
        session.commit()

    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.get(f"/jobs/{job_id}", headers=auth_headers)

    assert response.status_code == 403


# --- Job failure ---
# Tests error persistence through the lifespan worker. Does not test inference callbacks.


def test_failed_handler_stores_error_message(client: TestClient, registered_user: dict[str, str]):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    response = client.post("/jobs/test", json={"handler": "fail"}, headers=auth_headers)
    job_id = response.json()["job_id"]

    body = poll_job(client, job_id, expect_status="failed", headers=auth_headers)
    assert body["error"] == "Test job failed"
    assert body["completed_at"] is not None


# --- Transcribe enqueue ---
# Tests the enqueued job waits for an inference agent and carries one batched
# line payload. Does not run Kraken or Calamari.


@pytest.mark.integration
def test_transcribe_job_is_left_pending_for_an_inference_agent(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    owner_project: dict,
):
    """ADR 0003: the platform worker has no inference queue to hand this to.

    It therefore must not claim the job. Segment and transcribe rows stay
    ``pending`` until an agent claims them over HTTP.
    """
    # Submission is gated on **capacity** (051): a job may only be created for a
    # host that has a machine able to take it, so a cloud worker must be up.
    pair_inference_device(user_id=user_id_for_email(owner_user["email"]), host="cloud")

    document_id, part_id, _lines = _create_document_part_with_lines(
        client,
        owner_headers,
        owner_project,
    )

    base = documents_url(owner_project["id"])
    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert response.status_code == 202
    job_id = uuid.UUID(response.json()["job_id"])

    with sync_system_session() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == JobStatus.pending
        assert job.claimed_by is None
        assert job.inference_job_id is None

    # The structural half, and the whole guarantee: asked directly, the platform's
    # claim query declines this row. A wall-clock poll of the running worker only
    # ever confirmed what this single call states, so it is not spent here.
    assert claim_next_pending_job() is None


@pytest.mark.integration
def test_transcribe_job_payload_batches_every_selected_line(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    owner_project: dict,
):
    """The claim payload an agent will be handed, built from the real job row."""
    # Submission is gated on **capacity** (051): a job may only be created for a
    # host that has a machine able to take it, so a cloud worker must be up.
    pair_inference_device(user_id=user_id_for_email(owner_user["email"]), host="cloud")
    document_id, part_id, lines = _create_document_part_with_lines(
        client,
        owner_headers,
        owner_project,
    )

    base = documents_url(owner_project["id"])
    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
    )
    assert response.status_code == 202

    with sync_system_session() as session:
        job = session.get(Job, uuid.UUID(response.json()["job_id"]))
        assert job is not None
        request = build_page_run(job).request

    assert request.task.value == "transcribe"
    assert [line["line_id"] for line in request.params["lines"]] == [line["id"] for line in lines]
    assert [line["line_index"] for line in request.params["lines"]] == [0, 1]


# --- Opportunistic on-read sweep ---
# Tests the timeout is enforced by a plain read, with no worker loop involved.
# That is the production API deployment, which runs JOB_WORKER_ENABLED=false.
# The sweep populations themselves live in test_job_worker_sweeps.py.


@pytest.mark.integration
def test_reading_a_job_releases_a_stale_callback_claim(
    client: TestClient, registered_user: dict[str, str]
):
    from backend.jobs.infrastructure import stale_sweep

    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    me = client.get("/me", headers=auth_headers)
    assert me.status_code == 200
    user_id = uuid.UUID(me.json()["id"])

    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=JobStatus.waiting,
                payload={"handler": "noop"},
                user_id=user_id,
                callback_claimed_at=datetime.now(UTC) - timedelta(seconds=900),
                updated_at=datetime.now(UTC),
            )
        )
        session.commit()

    blocked = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert blocked.status_code == 409, blocked.text

    stale_sweep._last_sweep_at = None
    read = client.get(f"/jobs/{job_id}", headers=auth_headers)
    assert read.status_code == 200, read.text
    # Fresh updated_at, so only the claim is released - the job stays waiting.
    assert read.json()["status"] == "waiting"

    cancelled = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers)
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"


# --- Worker ownership ---
# Tests a reclaimed job cannot be finished by the worker that lost it.


@pytest.mark.integration
def test_reclaimed_job_rejects_the_zombie_workers_terminal_write(monkeypatch):
    # The session-scoped client's lifespan worker claims any pending job (this
    # suite leaves JOB_WORKER_CLAIM_TEST_ONLY unset), so it races this test for
    # the row twice: at the claim below, and again once the reclaim re-pends it.
    # The worker re-reads the shared settings singleton every tick, so excluding
    # test payloads for the duration makes both windows deterministic.
    from backend.core.settings.job import get_job_settings

    monkeypatch.setattr(get_job_settings(), "job_worker_claim_test_only", False)
    # A tick that read the unpatched settings can still be in flight; one poll
    # interval lets it finish before the pending row exists.
    time.sleep(get_job_settings().job_poll_interval_seconds)

    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.pipeline,
                status=JobStatus.pending,
                payload={"handler": "noop", "test": True},
            )
        )
        session.commit()

    claimed = claim_next_pending_job(test_only=True)
    assert claimed is not None
    assert claimed.id == job_id
    zombie_owner = claimed.claimed_by
    assert zombie_owner is not None

    with sync_system_session() as session:
        running = session.get(Job, job_id)
        assert running is not None
        running.started_at = datetime.now(UTC) - timedelta(minutes=31)
        session.commit()

    assert reclaim_stale_running_jobs(running_timeout_seconds=1800.0) == 1

    # The lease is gone; the zombie must not overwrite the re-pended job.
    mark_job_done(job_id, {"stale": True}, claimed_by=zombie_owner)

    with sync_system_session() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == JobStatus.pending
        assert row.claimed_by is None
        assert row.result is None
        # Unlike the other terminal writes in this file, this one is not dead
        # code: the payload carries ``test: True``, so `_drain_active_test_jobs`
        # would otherwise wait five seconds at teardown for the lifespan worker
        # to finish a row this test deliberately left claimable. The parent
        # conftest's truncate happens at the start of the *next* test, which is
        # after that wait.
        row.status = JobStatus.failed
        row.completed_at = datetime.now(UTC)
        session.commit()


# --- Clear job history ---
# Tests DELETE /jobs/history removes finished jobs only, for authorized projects.


def _project_job(
    *,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    status: JobStatus,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Job(
                id=job_id,
                type=JobType.segment,
                status=status,
                payload={"handler": "noop"},
                user_id=user_id,
                document_id=document_id,
                completed_at=datetime.now(UTC)
                if status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled)
                else None,
            )
        )
        session.commit()
    return job_id


@pytest.mark.integration
def test_clear_job_history_deletes_terminal_jobs_only(
    client: TestClient,
    owner_headers: dict[str, str],
    owner_project: dict,
):
    project_id = owner_project["id"]
    created = client.post(
        documents_url(project_id), headers=owner_headers, json={"name": "History doc"}
    )
    assert created.status_code == 201
    document_id = uuid.UUID(created.json()["id"])
    me = client.get("/me", headers=owner_headers)
    user_id = uuid.UUID(me.json()["id"])

    terminal_ids = [
        _project_job(document_id=document_id, user_id=user_id, status=status)
        for status in (JobStatus.done, JobStatus.failed, JobStatus.cancelled)
    ]
    active_id = _project_job(document_id=document_id, user_id=user_id, status=JobStatus.waiting)

    response = client.delete(f"/jobs/history?project_id={project_id}", headers=owner_headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"deleted": 3}

    with sync_system_session() as session:
        for job_id in terminal_ids:
            assert session.get(Job, job_id) is None
        surviving = session.get(Job, active_id)
        assert surviving is not None
        assert surviving.status == JobStatus.waiting


@pytest.mark.integration
def test_clear_job_history_requires_auth(client: TestClient, owner_project: dict):
    response = client.delete(f"/jobs/history?project_id={owner_project['id']}")
    assert response.status_code == 401


@pytest.mark.integration
def test_clear_job_history_denies_non_members(
    client: TestClient,
    owner_project: dict,
    outsider_headers: dict[str, str],
):
    response = client.delete(
        f"/jobs/history?project_id={owner_project['id']}", headers=outsider_headers
    )
    assert response.status_code in (403, 404)


# --- SSE authorization ---
# Tests dropping the request-scoped session kept the rejection path clean.


@pytest.mark.integration
def test_job_events_denies_access_to_other_users_job(
    client: TestClient,
    registered_user: dict[str, str],
    outsider_headers: dict[str, str],
):
    auth_headers = {"Authorization": f"Bearer {registered_user['access_token']}"}
    created = client.post("/jobs/test", headers=auth_headers)
    job_id = created.json()["job_id"]

    response = client.get(f"/jobs/{job_id}/events", headers=outsider_headers)

    # Rejected before the stream opens, not a hung event-stream response.
    assert response.status_code == 403
    assert not response.headers["content-type"].startswith("text/event-stream")


# --- Handler dispatch ---
# Tests unknown job types fail cleanly. Does not test registered handlers.


def test_execute_claimed_job_marks_unknown_handler_failed():
    job_id = uuid.uuid4()
    with sync_system_session() as session:
        job = Job(
            id=job_id,
            type=JobType.binarize,
            status=JobStatus.running,
            payload={"test": False},
        )
        session.add(job)
        session.commit()
        session.refresh(job)

    execute_claimed_job(job)

    with sync_system_session() as session:
        row = session.get(Job, job_id)
        assert row is not None
        assert row.status == JobStatus.failed
        assert (
            "not supported" in (row.error or "").lower()
            or "no handler" in (row.error or "").lower()
        )
