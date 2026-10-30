"""Unit tests for one agent claim: the row transition and the payload it hands out.

A fake session returns one pending job; the build, the media store and the
terminal write are patched, so the claim's own decisions are checked without
Postgres. The SKIP LOCKED query itself is integration-tested in
test_device_claim.py.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest

from backend.jobs.application import job_claim_service
from backend.jobs.application.inference_dispatcher import PageRun, PageRunRequest
from backend.jobs.application.job_claim_service import (
    UNBUILDABLE_PAYLOAD_ERROR,
    agent_claim_owner,
    claim_one_page,
    job_is_held_by,
)
from backend.jobs.infrastructure.orm_models import JobStatus, JobType
from backend.ml.application.agent_credentials import InferenceAgent
from backend.ml.domain.execution import ExecutionTarget
from nomikos_inference.contracts.common import InferenceTask as WireInferenceTask


def _agent(target: ExecutionTarget = ExecutionTarget.local) -> InferenceAgent:
    return InferenceAgent(
        device_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        execution_target=target,
        is_service_worker=target is ExecutionTarget.cloud,
    )


class _Result:
    def __init__(self, job: object | None) -> None:
        self._job = job

    def scalar_one_or_none(self) -> object | None:
        return self._job


class _ClaimSession:
    def __init__(self, job: object | None) -> None:
        self._job = job
        self.statements: list[str] = []
        self.commits = 0

    def execute(self, statement) -> _Result:
        self.statements.append(str(statement))
        return _Result(self._job)

    def commit(self) -> None:
        self.commits += 1


class _MediaStore:
    def signed_object_url(self, key: str, *, expires_at) -> str:
        return f"/media/{key}?expires={expires_at.isoformat()}"


@pytest.fixture
def claim(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    job = SimpleNamespace(
        id=uuid.uuid4(),
        type=JobType.transcribe,
        execution_target=ExecutionTarget.local,
        status=JobStatus.pending,
    )
    state = SimpleNamespace(job=job, sessions=[], notified=[], failed=[], build_error=None)

    @contextmanager
    def _factory():
        session = _ClaimSession(state.job)
        state.sessions.append(session)
        yield session

    def _build(built_job):
        if state.build_error is not None:
            raise state.build_error
        request = PageRunRequest(
            task=WireInferenceTask.segment,
            registry_model_id="blla-segment",
            registry_tag="stable",
            product_job_id=built_job.id,
        )
        return PageRun(request=request, image_key="pages/p1.jpg")

    def _mark_failed(job_id, error, *, claimed_by):
        state.failed.append((job_id, error, claimed_by))

    monkeypatch.setattr(job_claim_service, "sync_system_session", _factory)
    monkeypatch.setattr(job_claim_service, "build_page_run", _build)
    monkeypatch.setattr(job_claim_service, "get_media_store", _MediaStore)
    monkeypatch.setattr(
        job_claim_service,
        "notify_platform_job_status_changed",
        lambda job_id, status: state.notified.append((job_id, status)),
    )
    monkeypatch.setattr("backend.jobs.infrastructure.job_terminal.mark_job_failed", _mark_failed)
    return state


def _claim(agent: InferenceAgent):
    return claim_one_page(
        agent, lease_seconds=300, page_image_ttl_seconds=60, base_url="http://api.test/"
    )


# --- Claiming ---
# Tests the row a claim writes and the payload it returns.


def test_an_empty_queue_returns_no_page(claim: SimpleNamespace):
    claim.job = None

    assert _claim(_agent()) is None
    assert claim.notified == []


def test_a_claim_moves_the_page_to_waiting_for_this_agent(claim: SimpleNamespace):
    agent = _agent()

    page = _claim(agent)

    assert page is not None
    job = claim.job
    assert job.status is JobStatus.waiting
    assert job.claimed_by == agent_claim_owner(agent.device_id)
    assert job.inference_job_id == page.inference_job_id
    assert job.started_at == job.heartbeat_at == job.updated_at
    assert job.callback_claimed_at is None
    assert claim.sessions[0].commits == 1
    assert claim.notified == [(job.id, JobStatus.waiting)]
    assert job_is_held_by(job, agent)
    assert not job_is_held_by(job, _agent())


def test_a_claim_hands_out_a_signed_link_and_two_lifetimes(claim: SimpleNamespace):
    page = _claim(_agent())

    assert page is not None
    assert page.page_image_url.startswith("http://api.test/media/pages/p1.jpg?expires=")
    assert page.lease_expires_at - page.page_image_expires_at == timedelta(seconds=240)
    assert page.request.product_job_id == claim.job.id
    assert page.job_type is JobType.transcribe


def test_a_device_sees_only_its_own_pages_and_a_cloud_worker_everyones(
    claim: SimpleNamespace,
):
    _claim(_agent(ExecutionTarget.local))
    _claim(_agent(ExecutionTarget.cloud))

    local_sql, cloud_sql = (session.statements[0] for session in claim.sessions)
    assert "jobs.user_id = " in local_sql
    assert "jobs.user_id = " not in cloud_sql


def test_an_unbuildable_page_is_failed_and_returns_an_empty_claim(claim: SimpleNamespace):
    claim.build_error = ValueError("Document part not found")
    agent = _agent()

    assert _claim(agent) is None
    assert claim.failed == [
        (claim.job.id, UNBUILDABLE_PAYLOAD_ERROR, agent_claim_owner(agent.device_id))
    ]
