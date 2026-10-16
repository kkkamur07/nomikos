"""The version floor over HTTP, against the real app and live Postgres.

Everything here goes through ``create_app()``. ADR 0001 records why that is not
negotiable: a device layer that was never mounted hid behind a green suite for a
whole phase, because the integration tests built their own FastAPI app. So there
is no local app in this file, no substitute for Postgres, and the device
credential is minted by running the real pairing protocol.

The floor is moved the way an operator moves it - by changing configuration and
letting the platform re-read it - never by editing a constant or patching a
function. A test that reached into the code to change the floor would prove the
opposite of what this issue is for: the whole point is that a known-bad agent can
be stopped without shipping anything.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings import get_device_settings, reset_settings_caches
from backend.jobs.infrastructure.orm_models import JobStatus
from backend.ml.api.agent_version import (
    AGENT_VERSION_HEADER,
    AGENT_VERSION_REFUSED_STATUS,
    AGENT_VERSION_UNSUPPORTED,
)
from backend.ml.application.agent_credentials import SERVICE_TOKEN_HEADER, WORKER_NAME_HEADER
from infrastructure.db import sync_system_session

# Module-scoped autouse; see its docstring in `helpers.py` for issue #63.
from tests.nomikos.integration.helpers import (
    CLAIM_URL,
    DEVICE_SERVICE_TOKEN,
    MINIMAL_PNG,
    claim_page,
    documents_url,
    pair_device_over_http,
    return_pooled_connections_before_leaving,  # noqa: F401
)
from tests.nomikos.integration.helpers import device_headers as _headers
from tests.nomikos.integration.helpers import stored_job as _stored_job

pytestmark = pytest.mark.integration

SERVICE_HEADERS = {
    SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN,
    # Required: without it two hosted workers resolve to one helper_devices row
    # and neither can be told from the other on a claim. The test-helper
    # extraction on `fix/remediation-tests` predates that rule, so this header
    # has to be restored here rather than inherited.
    WORKER_NAME_HEADER: "cloud-worker",
}


@pytest.fixture
def version_policy(monkeypatch):
    """Move the floor the way an operator would: environment, then re-read.

    ``INFERENCE_AGENT_MIN_VERSION`` and ``INFERENCE_AGENT_LATEST_VERSION`` are
    ordinary device settings, memoized per process like every other. Setting the
    variable and dropping the cache is exactly what a redeploy does, and nothing
    in the platform is patched to make it happen.

    No reset on the way out: ``monkeypatch`` restores the environment at
    teardown, and the autouse ``isolated_platform_state`` fixture drops every
    settings cache at the *start* of the next test, so the restored values are
    the ones the next test reads.
    """

    def apply(*, minimum: str | None = None, latest: str | None = None) -> None:
        if minimum is not None:
            monkeypatch.setenv("INFERENCE_AGENT_MIN_VERSION", minimum)
        if latest is not None:
            monkeypatch.setenv("INFERENCE_AGENT_LATEST_VERSION", latest)
        reset_settings_caches()
        settings = get_device_settings()
        if minimum is not None:
            assert settings.inference_agent_min_version == minimum
        if latest is not None:
            assert settings.inference_agent_latest_version == latest

    return apply


# ---------------------------------------------------------------------------
# Live fixtures
# ---------------------------------------------------------------------------


def _claim(client: TestClient, token: str, version: str | None):
    """Unlike every other module, this one states the version explicitly on every
    call - including ``None``, which is a header that was never sent."""
    return claim_page(client, _headers(token, version=version))


def _paired_agent(client: TestClient, owner_headers, name: str = "Laptop") -> str:
    """Pair a laptop and let it announce itself by asking for work, which is what
    gives ``local`` **capacity**. The announcing claim states a current version:
    an agent below the floor never gets that far, which is the point."""
    paired = pair_device_over_http(client, owner_headers, name=name)
    empty = _claim(client, paired["device_token"], "9.9.9")
    assert empty.status_code == 200, empty.text
    return paired["device_token"]


def _submitted_page(client: TestClient, owner_headers, owner_project) -> str:
    """One pending ``local`` page, ready to be claimed."""
    prefer = client.put(
        "/account/execution-target", headers=owner_headers, json={"prefer_local_inference": True}
    )
    assert prefer.status_code == 200, prefer.text
    base = documents_url(owner_project["id"])
    created = client.post(base, headers=owner_headers, json={"name": "Claimable page"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=owner_headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201, upload.text
    submitted = client.post(
        f"{base}/{document_id}/parts/{upload.json()['id']}/segment", headers=owner_headers
    )
    assert submitted.status_code == 202, submitted.text
    return submitted.json()["job_id"]


def _refusal(response) -> dict:
    assert response.status_code == AGENT_VERSION_REFUSED_STATUS, response.text
    return response.json()["error"]


# ---------------------------------------------------------------------------
# Below the floor: refused, and told what would fix it
# ---------------------------------------------------------------------------


def test_an_agent_below_the_floor_is_refused_with_an_actionable_response(
    client: TestClient, owner_user, owner_headers, owner_project, version_policy
) -> None:
    """The refusal has to be something a CLI can *act on*: not "it failed", but
    "you are 0.3.0, you need 0.4.0, install this, and do not bother retrying"."""
    token = _paired_agent(client, owner_headers)
    job_id = _submitted_page(client, owner_headers, owner_project)
    version_policy(minimum="0.4.0", latest="0.6.2")

    response = _claim(client, token, "0.3.0")

    error = _refusal(response)
    assert error["code"] == AGENT_VERSION_UNSUPPORTED
    assert error["reason"] == "below_floor"
    assert error["agent_version"] == "0.3.0"
    assert error["minimum_version"] == "0.4.0"
    assert error["latest_version"] == "0.6.2"
    assert error["package"] == "nomikos-inference"
    assert error["upgrade_command"] == "uv tool upgrade nomikos-inference"
    assert error["retryable"] is False
    assert "0.4.0" in error["message"]
    # And it took no work with it.
    assert _stored_job(job_id).status is JobStatus.pending


def test_a_refused_agent_is_told_apart_from_every_other_failure(
    client: TestClient, owner_user, owner_headers, version_policy
) -> None:
    """Four outcomes on one endpoint, and a claim loop has to branch differently
    on each. If the refusal were a 401 it would look like an expired credential
    and the agent would re-pair; if it were a 200 with no page it would look like
    an empty queue and the agent would poll forever.
    """
    token = _paired_agent(client, owner_headers)
    version_policy(minimum="0.4.0", latest="0.4.0")

    stale = _claim(client, token, "0.1.0")
    bad_credential = _claim(client, "nmd1.not-a-token", "0.4.0")
    empty_queue = _claim(client, token, "0.4.0")

    assert stale.status_code == 426
    assert bad_credential.status_code == 401
    assert empty_queue.status_code == 200

    assert stale.json()["error"]["code"] == AGENT_VERSION_UNSUPPORTED
    # Pinned, not merely "not the version code". `!=` is satisfied by an empty
    # string, a typo, or any other code in the platform - none of which a claim
    # loop could branch on, which is the whole subject of this test. A bad
    # credential is UNAUTHORIZED, the same code every other 401 uses, so an
    # agent can reuse the re-pair branch it already has.
    assert bad_credential.json()["error"]["code"] == "UNAUTHORIZED"
    assert empty_queue.json()["page"] is None
    assert "error" not in empty_queue.json()


def test_the_floor_outranks_the_credential(
    client: TestClient, owner_headers, version_policy
) -> None:
    """A stale agent is refused before it is authenticated, so it never touches a
    session and never records **capacity**. That is deliberate: an agent that may
    not claim must stop reporting that it could, or submission would keep
    creating pages for a host that cannot take them.
    """
    version_policy(minimum="0.4.0", latest="0.4.0")

    unpaired_and_stale = _claim(client, "nmd1.not-a-token", "0.1.0")

    assert unpaired_and_stale.status_code == 426
    assert unpaired_and_stale.json()["error"]["reason"] == "below_floor"


# ---------------------------------------------------------------------------
# At or above the floor: served
# ---------------------------------------------------------------------------


def test_an_agent_above_the_floor_but_behind_the_latest_is_served_and_told(
    client: TestClient, owner_user, owner_headers, owner_project, version_policy
) -> None:
    """The state that must never collapse into the refusal: outdated is a notice
    delivered *with* the work, not instead of it. Most upgrades are not urgent,
    and refusing them would make every release an outage for anyone who had not
    restarted."""
    token = _paired_agent(client, owner_headers)
    job_id = _submitted_page(client, owner_headers, owner_project)
    version_policy(minimum="0.4.0", latest="0.6.2")

    response = _claim(client, token, "0.5.0")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page"] is not None, "an outdated agent is still handed work"
    assert body["page"]["product_job_id"] == job_id
    assert body["agent"]["outdated"] is True
    assert body["agent"]["agent_version"] == "0.5.0"
    assert body["agent"]["minimum_version"] == "0.4.0"
    assert body["agent"]["latest_version"] == "0.6.2"
    assert body["agent"]["upgrade_command"] == "uv tool upgrade nomikos-inference"
    assert _stored_job(job_id).status is JobStatus.waiting


# ---------------------------------------------------------------------------
# The floor is configuration
# ---------------------------------------------------------------------------


def test_raising_the_floor_stops_an_agent_that_was_claiming_a_moment_ago(
    client: TestClient, owner_user, owner_headers, owner_project, version_policy
) -> None:
    """The kill switch, in the order it happens in production: the agent is
    working, someone raises the floor, the very next poll is refused."""
    token = _paired_agent(client, owner_headers)
    version_policy(minimum="0.4.0", latest="0.4.0")
    first_job = _submitted_page(client, owner_headers, owner_project)
    assert _claim(client, token, "0.4.0").json()["page"]["product_job_id"] == first_job

    version_policy(minimum="0.4.1", latest="0.4.1")
    second_job = _submitted_page(client, owner_headers, owner_project)
    refused = _claim(client, token, "0.4.0")

    assert refused.status_code == 426
    assert _stored_job(second_job).status is JobStatus.pending


# ---------------------------------------------------------------------------
# The hosted worker is an agent like any other
# ---------------------------------------------------------------------------


def test_a_refused_hosted_worker_never_provisions_its_device_row(
    client: TestClient, version_policy
) -> None:
    """Refusal lands before authentication, so a stale worker does not register
    itself as cloud **capacity** on its way to being turned away."""
    from backend.ml.domain.execution import ExecutionTarget
    from backend.ml.infrastructure.device_orm_models import HelperDevice

    version_policy(minimum="0.4.0", latest="0.4.0")

    refused = client.post(
        CLAIM_URL,
        headers={**SERVICE_HEADERS, AGENT_VERSION_HEADER: "0.3.0"},
        json={"wait_seconds": 0},
    )

    assert refused.status_code == 426
    with sync_system_session() as session:
        cloud_devices = (
            session.execute(
                select(HelperDevice).where(HelperDevice.inference_host == ExecutionTarget.cloud)
            )
            .scalars()
            .all()
        )
    assert cloud_devices == []
