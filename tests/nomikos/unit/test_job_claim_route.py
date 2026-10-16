"""Unit tests for POST /device/v1/jobs/claim: the long poll around one claim.

The route function is called directly. Authentication, the on-read sweep and
``claim_one_page`` are stubbed, so the wait, the disconnect checks and the
response shape are checked without a database. The real app's wiring is in
test_device_claim_wiring.py; the claim itself in test_job_claim_service.py.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.core.exceptions import InvalidCredentialsError
from backend.jobs.api import device_claim
from backend.jobs.api.claim_schemas import JobClaimRequest
from backend.jobs.application.inference_dispatcher import PageRunRequest
from backend.jobs.application.job_claim_service import ClaimedPage
from backend.jobs.infrastructure.orm_models import JobType
from backend.ml.api.agent_version import AgentVersionNotice
from backend.ml.application.agent_credentials import InferenceAgent
from backend.ml.domain.execution import ExecutionTarget
from nomikos_inference.contracts.common import InferenceTask as WireInferenceTask

_VERSION = AgentVersionNotice(
    agent_version="1.0.0",
    minimum_version="1.0.0",
    latest_version="1.0.0",
    outdated=False,
    package="nomikos-agent",
    upgrade_command="pip install -U nomikos-agent",
)


class _Request:
    """Connected for the first *connected_checks* disconnect checks, then gone."""

    def __init__(self, connected_checks: int = 1_000) -> None:
        self._connected_checks = connected_checks
        self.base_url = "http://api.test/"

    async def is_disconnected(self) -> bool:
        self._connected_checks -= 1
        return self._connected_checks < 0


def _page() -> ClaimedPage:
    job_id = uuid.uuid4()
    now = datetime.now(UTC)
    return ClaimedPage(
        product_job_id=job_id,
        inference_job_id=uuid.uuid4(),
        job_type=JobType.segment,
        execution_target=ExecutionTarget.local,
        lease_expires_at=now + timedelta(minutes=5),
        request=PageRunRequest(
            task=WireInferenceTask.segment,
            registry_model_id="blla-segment",
            registry_tag="stable",
            product_job_id=job_id,
            params={"text_direction": "rl"},
        ),
        page_image_url="http://api.test/media/p1.jpg",
        page_image_expires_at=now + timedelta(minutes=1),
    )


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    agent = InferenceAgent(
        device_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        execution_target=ExecutionTarget.local,
        is_service_worker=False,
    )
    state = SimpleNamespace(agent=agent, pages=[], claims=[], sweeps=0, auth=[])
    settings = _settings_with_max_wait(1)

    async def _authenticate(**kwargs):
        state.auth.append(kwargs)
        return agent

    async def _sweep() -> None:
        state.sweeps += 1

    def _claim(claiming_agent, **kwargs):
        state.claims.append((claiming_agent, kwargs))
        return state.pages.pop(0) if state.pages else None

    monkeypatch.setattr(device_claim, "get_device_settings", lambda: settings)
    monkeypatch.setattr(device_claim, "_authenticate", _authenticate)
    monkeypatch.setattr(device_claim, "sweep_stale_jobs_on_read", _sweep)
    monkeypatch.setattr(device_claim, "claim_one_page", _claim)
    monkeypatch.setattr(device_claim, "client_ip_for_request", lambda _request: "203.0.113.9")
    return state


async def _call(request: _Request, body: JobClaimRequest | None = None):
    return await device_claim.claim_job(
        request, _VERSION, body, x_nomikos_device_token="device-token"
    )


# --- One claim ---


async def test_a_waiting_page_is_handed_over_at_once(route: SimpleNamespace):
    page = _page()
    route.pages = [page]

    response = await _call(_Request())

    assert response.page is not None
    assert response.page.product_job_id == page.product_job_id
    assert response.page.request.params == {"text_direction": "rl"}
    assert response.page.page_image_url == page.page_image_url
    assert response.poll_after_seconds == 0.0
    assert response.lease_seconds == 300
    assert response.agent == _VERSION
    assert route.sweeps == 1
    (agent, kwargs) = route.claims[0]
    assert agent is route.agent
    assert kwargs == {
        "lease_seconds": 300,
        "page_image_ttl_seconds": 60,
        "base_url": "http://api.test/",
    }
    assert route.auth == [
        {
            "device_token": "device-token",
            "service_token": None,
            "worker_name": None,
            "request_ip": "203.0.113.9",
        }
    ]


async def test_an_empty_queue_without_a_wait_is_one_attempt(route: SimpleNamespace):
    response = await _call(_Request())

    assert response.page is None
    assert response.poll_after_seconds == 7.5
    assert len(route.claims) == 1


# --- Long poll ---


async def test_a_long_poll_keeps_asking_until_a_page_arrives(route: SimpleNamespace):
    page = _page()
    route.pages = [None, None, page]

    response = await _call(_Request(), JobClaimRequest(wait_seconds=1))

    assert response.page is not None
    assert response.page.product_job_id == page.product_job_id
    assert len(route.claims) == 3


async def test_a_long_poll_ends_empty_at_the_deadline(
    route: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    clock = iter(range(100))
    monkeypatch.setattr(device_claim.time, "monotonic", lambda: float(next(clock)))

    response = await _call(_Request(), JobClaimRequest(wait_seconds=1))

    # Deadline is one tick away: one attempt, then the clock has run out.
    assert response.page is None
    assert response.poll_after_seconds == 7.5
    assert len(route.claims) == 1


async def test_the_wait_is_capped_by_the_platform(
    route: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(device_claim, "get_device_settings", lambda: _settings_with_max_wait(0))

    response = await _call(_Request(), JobClaimRequest(wait_seconds=25))

    assert response.page is None
    assert len(route.claims) == 1


def _settings_with_max_wait(seconds: int) -> SimpleNamespace:
    return SimpleNamespace(
        device_claim_max_wait_seconds=seconds,
        device_claim_poll_interval_seconds=0.001,
        device_claim_idle_poll_seconds=7.5,
        device_lease_seconds=300,
        device_page_image_url_ttl_seconds=60,
    )


# --- Disconnects ---
# Tests a caller that hung up is never handed a page it cannot receive.


async def test_a_caller_gone_before_the_first_attempt_claims_nothing(route: SimpleNamespace):
    route.pages = [_page()]

    response = await _call(_Request(connected_checks=0), JobClaimRequest(wait_seconds=1))

    assert response.page is None
    assert route.claims == []


async def test_a_caller_gone_mid_wait_stops_the_poll(route: SimpleNamespace):
    route.pages = [None, _page()]

    # Connected for the pre-claim check, gone by the post-claim check.
    response = await _call(_Request(connected_checks=1), JobClaimRequest(wait_seconds=1))

    assert response.page is None
    assert len(route.claims) == 1


# --- Authentication ---


async def test_a_bad_credential_is_a_401_naming_the_device_header(
    monkeypatch: pytest.MonkeyPatch,
):
    sessions: list[str] = []

    @asynccontextmanager
    async def _session():
        sessions.append("opened")
        yield object()
        sessions.append("closed")

    async def _refuse(_session, **_kwargs):
        raise InvalidCredentialsError("unknown device token")

    monkeypatch.setattr(device_claim, "AsyncSessionLocal", _session)
    monkeypatch.setattr(device_claim, "resolve_inference_agent", _refuse)

    with pytest.raises(HTTPException) as refused:
        await device_claim._authenticate(
            device_token="stale", service_token=None, worker_name=None, request_ip=None
        )

    assert refused.value.status_code == 401
    assert refused.value.headers == {"WWW-Authenticate": device_claim.DEVICE_TOKEN_HEADER}
    assert sessions == ["opened"]
