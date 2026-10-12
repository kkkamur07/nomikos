"""Shared helpers for nomikos integration tests.

Nothing here imports the backend at module scope. ``conftest.py`` sets the
environment the settings objects read *before* it imports ``backend.core.app``,
and it imports this module for the service-token constant, so a top-level
backend import here would be resolved too early. The backend imports are inside
the functions that need them, deliberately.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fixtures.paths import MINIMAL_PNG

__all__ = [
    "CALLBACK_URL",
    "CLAIM_URL",
    "CURRENT_AGENT_VERSION",
    "DEVICE_SERVICE_TOKEN",
    "MINIMAL_PNG",
    "assert_api_error",
    "claim_page",
    "device_headers",
    "documents_url",
    "make_part",
    "pair_device_over_http",
    "pair_inference_device",
    "poll_job",
    "prefer_local",
    "return_pooled_connections_before_leaving",
    "running_agent",
    "seed_transcription_model",
    "service_headers",
    "stored_job",
    "stored_minimal_page_bytes",
    "submit_segment",
    "user_id_for_email",
]

# ---------------------------------------------------------------------------
# The device claim surface, in one place
# ---------------------------------------------------------------------------
#
# `test_device_claim.py`, `test_device_lease.py`, `test_agent_version_floor.py`
# and `test_signed_page_image_link.py` each used to carry their own copy of the
# constants and helpers below. Four copies of "how an agent is paired and made
# to report capacity" is four places to update when that changes, and three of
# them re-declared the service token as a literal that only conftest is entitled
# to define - so a token changed in one place would have failed in the other
# three for a reason that looked nothing like the change.

CLAIM_URL = "/device/v1/jobs/claim"
CALLBACK_URL = "/internal/inference/job-complete"

DEVICE_SERVICE_TOKEN = "test-inference-worker-service-token-not-for-production"
"""The hosted worker's claim credential. ``conftest.py`` puts this in the
environment the settings object reads, so this constant is the definition and
every test that sends the header must take it from here rather than repeat it."""

CURRENT_AGENT_VERSION = "1.0.0"
"""What a claim states about itself when the version floor is not the subject.

Comfortably above any floor these suites configure, so nothing outside
``test_agent_version_floor.py`` is accidentally testing the floor."""


def device_headers(token: str, *, version: str | None = CURRENT_AGENT_VERSION) -> dict[str, str]:
    """Credentials for one paired laptop. ``version=None`` sends no version at
    all, which is the state ``test_agent_version_floor.py`` refuses."""
    from backend.ml.api.agent_version import AGENT_VERSION_HEADER
    from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER

    headers = {DEVICE_TOKEN_HEADER: token}
    if version is not None:
        headers[AGENT_VERSION_HEADER] = version
    return headers


def service_headers(
    *, version: str | None = CURRENT_AGENT_VERSION, worker_name: str | None = None
) -> dict[str, str]:
    """Credentials for the hosted worker - a service token, not a device one."""
    from backend.ml.api.agent_version import AGENT_VERSION_HEADER
    from backend.ml.application.agent_credentials import SERVICE_TOKEN_HEADER, WORKER_NAME_HEADER

    headers = {SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN}
    if version is not None:
        headers[AGENT_VERSION_HEADER] = version
    if worker_name is not None:
        headers[WORKER_NAME_HEADER] = worker_name
    return headers


def claim_page(client: TestClient, headers: dict[str, str], *, wait_seconds: int = 0):
    """One claim, returned unasserted: the status code is often the subject."""
    return client.post(CLAIM_URL, headers=headers, json={"wait_seconds": wait_seconds})


def running_agent(client: TestClient, headers: dict[str, str], name: str = "Laptop") -> dict:
    """Pair a laptop and let it announce itself the way the agent does: by asking
    for work. That first empty claim is what gives ``local`` **capacity**."""
    paired = pair_device_over_http(client, headers, name=name)
    empty = claim_page(client, device_headers(paired["device_token"]))
    assert empty.status_code == 200, empty.text
    assert empty.json()["page"] is None
    return paired


def prefer_local(client: TestClient, headers: dict[str, str]) -> None:
    """Send this account's work to its own laptop rather than to the cloud."""
    response = client.put(
        "/account/execution-target", headers=headers, json={"prefer_local_inference": True}
    )
    assert response.status_code == 200, response.text


def make_part(
    client: TestClient, headers: dict[str, str], project: dict, *, name: str = "Claimable page"
) -> tuple[str, str]:
    """A document with one uploaded page. Returns ``(document_id, part_id)``."""
    base = documents_url(project["id"])
    created = client.post(base, headers=headers, json={"name": name})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201, upload.text
    return document_id, upload.json()["id"]


def submit_segment(
    client: TestClient, headers: dict[str, str], project: dict, ids: tuple[str, str]
) -> str:
    """Queue one page for segmentation. Returns the product job id."""
    document_id, part_id = ids
    response = client.post(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segment",
        headers=headers,
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def stored_job(job_id: str | uuid.UUID):
    """The job row as Postgres holds it, not as the API renders it."""
    from backend.jobs.infrastructure.orm_models import Job
    from infrastructure.db import sync_system_session

    key = job_id if isinstance(job_id, uuid.UUID) else uuid.UUID(job_id)
    with sync_system_session() as session:
        return session.execute(select(Job).where(Job.id == key)).scalar_one()


@pytest.fixture(scope="module", autouse=True)
def return_pooled_connections_before_leaving(client: TestClient):
    """Empty the async pool on the way out, on the loop that filled it.

    Import this into any module that opens more than a connection or two at
    once. Those modules leave an async pool full of live ``asyncpg`` connections
    bound to the session client's event loop; ``test_device_pairing.py`` then
    starts a *second* app on its own loop and inherits them, which is the
    collision tracked as issue #63.

    Disposing through ``client.portal`` runs the teardown on the loop that owns
    those connections, so they are closed rather than orphaned. It does not fix
    #63; it stops these modules feeding it.

    It is a fixture rather than a plain function so that importing the name is
    the whole of the wiring - an autouse fixture imported into a test module is
    scoped to that module, so this does not fire for the rest of the suite.
    """
    from infrastructure.db import engine

    yield
    client.portal.call(engine.dispose)


def pair_device_over_http(
    client: TestClient,
    headers: dict[str, str],
    *,
    name: str = "Researcher laptop",
) -> dict[str, str]:
    """Run the whole pairing protocol against the real app and return the token.

    A real device credential, minted the way production mints one: helper starts
    a pairing, the browser consents, the helper collects. Nothing is written to
    ``helper_devices`` by hand, so a change that breaks pairing breaks the tests
    that build on it too.

    The helper deliberately does **not** poll before consent. A compliant poll
    cadence is enforced on the pairing row itself, and a second poll inside the
    default 5s interval would legitimately answer ``slow_down``; there is nothing
    to learn from that here, and it is covered where it belongs.
    """
    started = client.post(
        "/device/v1/pairings",
        json={
            "device_name": name,
            "platform": "darwin-arm64",
            "helper_version": "0.2.0",
            "capabilities": {"runtime": "torch"},
        },
    )
    assert started.status_code == 201, started.text
    pairing = started.json()
    verification_token = pairing["verification_url"].split("#", 1)[1]

    approved = client.post(
        f"/devices/pairings/{pairing['pairing_id']}/approve",
        headers=headers,
        json={"verification_token": verification_token},
    )
    assert approved.status_code == 200, approved.text

    collected = client.post(
        "/device/v1/pairings/token",
        json={"pairing_id": pairing["pairing_id"], "device_code": pairing["device_code"]},
    )
    assert collected.status_code == 200, collected.text
    body = collected.json()
    assert body["status"] == "approved", body
    return {"device_id": body["device_id"], "device_token": body["device_token"]}


def user_id_for_email(email: str) -> uuid.UUID:
    from backend.users.infrastructure.orm_models import User
    from infrastructure.db import sync_system_session

    with sync_system_session() as session:
        return session.execute(select(User.id).where(User.email == email)).scalar_one()


def pair_inference_device(
    *,
    user_id: uuid.UUID,
    host: str = "cloud",
    seen_seconds_ago: float | None = 5,
) -> uuid.UUID:
    """Give an **inference host** **capacity** by writing a recently-seen device.

    Submission is gated on capacity, so any test that expects a 202 has to say
    which host is running. This writes the real row the real query reads -
    ``last_seen_at`` is the production signal, and controlling it is how capacity
    is made deterministic without patching a clock.

    ``seen_seconds_ago=None`` writes a device that has never checked in: paired
    but not running, which must not count as capacity.
    """
    from backend.ml.domain.execution import ExecutionTarget
    from backend.ml.infrastructure.device_orm_models import HelperDevice
    from infrastructure.db import sync_system_session

    device_id = uuid.uuid4()
    now = datetime.now(UTC)
    with sync_system_session() as session:
        session.add(
            HelperDevice(
                id=device_id,
                user_id=user_id,
                inference_host=ExecutionTarget(host),
                name=f"{host} worker",
                platform="linux-x86_64",
                helper_version="0.2.0",
                capabilities={},
                token_hash="a" * 64,
                token_prefix="nmd1.test",
                last_seen_at=(
                    None if seen_seconds_ago is None else now - timedelta(seconds=seen_seconds_ago)
                ),
            )
        )
        session.commit()
    return device_id


def assert_api_error(
    response,
    *,
    code: str,
    message: str | None = None,
) -> dict:
    """Assert allowlisted API error shape (code/message); allow correlation `ref`."""
    body = response.json()
    assert "error" in body
    error = body["error"]
    assert error["code"] == code
    if message is not None:
        assert error["message"] == message
    if "ref" in error:
        assert isinstance(error["ref"], str) and error["ref"]
    return error


@lru_cache
def stored_minimal_page_bytes() -> bytes:
    """Bytes expected after upload normalization to stored WebP."""
    from backend.document.infrastructure.media_store.encoding import encode_part_image

    return encode_part_image(MINIMAL_PNG)


def seed_transcription_model() -> str:
    """One transcription row in the catalog, for a transcribe that names its model.

    A transcribe with no ``model_id`` and no binding is refused (422), so a test
    that is about something else sends this id in the body.
    """
    from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
    from infrastructure.db import sync_system_session

    with sync_system_session() as session:
        model = InferenceModel(
            name=f"calamari-test-{uuid.uuid4().hex[:8]}",
            provider="calamari",
            task=InferenceTask.transcribe,
            artifact_ref="registry://calamari-test?tag=stable",
            default_params={},
        )
        session.add(model)
        session.commit()
        return str(model.id)


def documents_url(project_id: str) -> str:
    return f"/projects/{project_id}/documents"


def poll_job(
    client: TestClient,
    job_id: str,
    *,
    expect_status: str = "done",
    headers: dict[str, str],
    timeout: float = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expect_status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach status {expect_status!r} in {timeout}s")
