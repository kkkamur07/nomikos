"""The claim endpoint, over HTTP, against the real app and live Postgres.

Everything here goes through ``create_app()``. ADR 0001 records why that is not
negotiable: an earlier device layer was never mounted, and the integration suite
hid it by building its own FastAPI app - a whole phase unreachable behind a green
suite. So there is no local app in this file, no substitute for Postgres, and the
device credential is minted by running the real pairing protocol rather than by
writing a row.

There are no assertions about how long a long poll takes. The contract is what a
claim returns and to whom; the clock is not part of it, and an assertion on it
would fail for reasons that are not defects.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.core.settings import get_inference_settings
from backend.jobs.application.job_claim_service import agent_claim_owner
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from backend.ml.api.agent_version import AGENT_VERSION_HEADER
from backend.ml.application.agent_credentials import (
    SERVICE_ACCOUNT_ID,
    SERVICE_TOKEN_HEADER,
    WORKER_NAME_HEADER,
)
from backend.ml.application.device_auth import DEVICE_TOKEN_HEADER
from backend.ml.domain.execution import ExecutionTarget
from backend.ml.infrastructure.device_orm_models import HelperDevice
from infrastructure.db import sync_system_session
from nomikos_inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER

# An autouse fixture imported into a module is scoped to that module: this closes
# the asyncpg connections the concurrent claims below open, on the loop that owns
# them. See its docstring in `helpers.py` for issue #63.
from tests.nomikos.integration.helpers import (
    CALLBACK_URL,
    CLAIM_URL,
    CURRENT_AGENT_VERSION,
    DEVICE_SERVICE_TOKEN,
    pair_device_over_http,
    return_pooled_connections_before_leaving,  # noqa: F401
)
from tests.nomikos.integration.helpers import claim_page as _claim
from tests.nomikos.integration.helpers import device_headers as _device_headers
from tests.nomikos.integration.helpers import make_part as _make_part
from tests.nomikos.integration.helpers import prefer_local as _prefer_local
from tests.nomikos.integration.helpers import running_agent as _running_agent
from tests.nomikos.integration.helpers import stored_job as _stored_job
from tests.nomikos.integration.helpers import submit_segment as _submit_segment

pytestmark = pytest.mark.integration

# Every claim states which agent is calling (issue 055): one that does not is
# refused rather than assumed current. `CURRENT_AGENT_VERSION` is comfortably
# above the configured floor, so nothing in this file is testing the floor -
# that is ``test_agent_version_floor.py``.
SERVICE_HEADERS = {
    SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN,
    AGENT_VERSION_HEADER: CURRENT_AGENT_VERSION,
    # Required: without it two hosted workers resolve to one helper_devices row
    # and neither can be told from the other on a claim.
    WORKER_NAME_HEADER: "cloud-worker",
}


# ---------------------------------------------------------------------------
# Live fixtures: real pairing, real capacity, real jobs
# ---------------------------------------------------------------------------


def _running_cloud_worker(client: TestClient, *, worker_name: str = "cloud-worker") -> dict:
    """A hosted worker registers itself by working. Its first claim provisions the
    ``cloud`` device row that reports cloud **capacity**."""
    response = client.post(
        CLAIM_URL,
        headers={**SERVICE_HEADERS, WORKER_NAME_HEADER: worker_name},
        json={"wait_seconds": 0},
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# One claim returns at most one page
# ---------------------------------------------------------------------------


def test_one_claim_hands_over_exactly_one_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """A batch is N claims, not one claim of N pages."""
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    first = _submit_segment(client, owner_headers, owner_project, ids)
    second = _submit_segment(client, owner_headers, owner_project, ids)

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    body = response.json()
    page = body["page"]
    assert page is not None
    assert page["product_job_id"] == first, "the oldest pending page goes first"
    assert page["job_type"] == "segment"
    assert page["execution_target"] == "local"
    # The claim carries the instruction, and the scan comes from the signed link
    # beside it. Shipping both meant every claim streamed a base64 manuscript page
    # through a serverless API for a field no agent reads.
    assert page["request"]["product_job_id"] == first
    assert page["request"]["task"] == "segment"
    assert "image_bytes" not in page["request"]
    assert "signature=" in page["page_image_url"], "the scan comes from the signed link only"
    assert page["page_image_expires_at"]
    assert page["inference_job_id"] == str(_stored_job(first).inference_job_id)
    assert body["poll_after_seconds"] == 0
    assert body["lease_seconds"] > 0

    # The second page is untouched and still claimable.
    assert _stored_job(second).status is JobStatus.pending
    assert _stored_job(first).status is JobStatus.waiting
    assert _stored_job(first).claimed_by == agent_claim_owner(uuid.UUID(agent["device_id"]))


# ---------------------------------------------------------------------------
# Authorization: account scope, and execution target
# ---------------------------------------------------------------------------


def test_a_device_cannot_claim_another_accounts_work(
    client: TestClient,
    owner_user,
    owner_headers,
    outsider_user,
    outsider_headers,
    owner_project,
) -> None:
    """The device credential's scope is one ``helper_devices.user_id`` foreign key,
    and the queue predicate is that same column. Not a code-review promise."""
    owner_agent = _running_agent(client, owner_headers, name="Owner laptop")
    outsider_agent = _running_agent(client, outsider_headers, name="Outsider laptop")
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)

    stolen = _claim(client, _device_headers(outsider_agent["device_token"]))

    assert stolen.status_code == 200, stolen.text
    assert stolen.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending
    # And the rightful owner still gets it.
    assert (
        _claim(client, _device_headers(owner_agent["device_token"])).json()["page"][
            "product_job_id"
        ]
        == job_id
    )


def test_a_device_cannot_claim_cloud_work(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """``cloud`` work is platform work: it is not scoped by the one account a
    device token names, so honouring a device token for it would hand one
    laptop every account's pages."""
    agent = _running_agent(client, owner_headers)
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    assert response.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending


def test_a_service_credential_claims_cloud_work_from_the_same_endpoint(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """One agent implementation, two credentials - not two code paths kept in
    parity by discipline (ADR 0003)."""
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)

    response = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})

    assert response.status_code == 200, response.text
    page = response.json()["page"]
    assert page is not None
    assert page["product_job_id"] == job_id
    assert page["execution_target"] == "cloud"
    assert _stored_job(job_id).status is JobStatus.waiting


def test_a_service_credential_cannot_claim_local_work(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The asymmetry runs both ways: a hosted worker never takes a page a
    researcher's own computer was chosen for."""
    _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    assert _stored_job(job_id).execution_target is ExecutionTarget.local

    response = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0})

    assert response.status_code == 200, response.text
    assert response.json()["page"] is None
    assert _stored_job(job_id).status is JobStatus.pending


def test_an_unauthenticated_or_wrong_credential_never_reaches_the_queue(
    client: TestClient, owner_headers
) -> None:
    """Every call here states a current agent version, so the only thing being
    judged is the credential. A stale agent is a different refusal with a
    different status; ``test_agent_version_floor.py`` proves they stay apart."""
    version = {AGENT_VERSION_HEADER: CURRENT_AGENT_VERSION}
    assert client.post(CLAIM_URL, headers=version, json={"wait_seconds": 0}).status_code == 401
    assert (
        client.post(
            CLAIM_URL,
            headers={**version, DEVICE_TOKEN_HEADER: "nmd1.not-a-token"},
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )
    assert (
        client.post(
            CLAIM_URL,
            headers={
                **version,
                SERVICE_TOKEN_HEADER: "wrong-service-token-but-long-enough-to-pass",
            },
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )
    # A valid service token with no worker name is refused as well. It used to be
    # accepted and silently resolved to a shared "cloud-worker" row, which made
    # every anonymous hosted worker the same device: whichever one claimed a page,
    # any of the others could complete it.
    assert (
        client.post(
            CLAIM_URL,
            headers={**version, SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN},
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )
    assert (
        client.post(
            CLAIM_URL,
            headers={
                **version,
                SERVICE_TOKEN_HEADER: DEVICE_SERVICE_TOKEN,
                WORKER_NAME_HEADER: "   ",
            },
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )
    # A browser access token is not an agent credential either.
    bearer = owner_headers["Authorization"].split(" ", 1)[1]
    assert (
        client.post(
            CLAIM_URL,
            headers={**version, DEVICE_TOKEN_HEADER: bearer},
            json={"wait_seconds": 0},
        ).status_code
        == 401
    )


def test_a_revoked_device_stops_claiming_on_its_very_next_call(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent = _running_agent(client, owner_headers)
    revoked = client.delete(f"/devices/{agent['device_id']}", headers=owner_headers)
    assert revoked.status_code == 204, revoked.text

    assert _claim(client, _device_headers(agent["device_token"])).status_code == 401


# ---------------------------------------------------------------------------
# The service account, decided in this issue
# ---------------------------------------------------------------------------


def test_the_hosted_worker_is_owned_by_a_service_account_no_person_can_hold(
    client: TestClient, owner_user, owner_headers
) -> None:
    """``helper_devices.user_id`` is NOT NULL by design, so a hosted worker's row
    needs an owner - and that owner must not be a researcher, or an account
    deletion would take cloud inference with it."""
    _running_cloud_worker(client)

    with sync_system_session() as session:
        devices = (
            session.execute(
                select(HelperDevice).where(HelperDevice.inference_host == ExecutionTarget.cloud)
            )
            .scalars()
            .all()
        )
        assert len(devices) == 1
        worker = devices[0]
        assert worker.user_id == SERVICE_ACCOUNT_ID
        # It has no device credential at all: the service token is the only way
        # in, so revoking it is one environment variable rather than a row edit.
        assert worker.token_hash == ""
        assert worker.last_seen_at is not None

    # It never appears in a researcher's device list.
    assert client.get("/devices", headers=owner_headers).json() == []


def test_repeated_hosted_worker_claims_reuse_one_device_row(client: TestClient) -> None:
    for _ in range(3):
        _running_cloud_worker(client)
    _running_cloud_worker(client, worker_name="second-worker")

    with sync_system_session() as session:
        names = sorted(
            session.execute(
                select(HelperDevice.name).where(
                    HelperDevice.inference_host == ExecutionTarget.cloud
                )
            )
            .scalars()
            .all()
        )
    assert names == ["cloud-worker", "second-worker"]


# ---------------------------------------------------------------------------
# The empty queue
# ---------------------------------------------------------------------------


def test_an_empty_queue_is_a_well_formed_response_and_not_an_error(
    client: TestClient, owner_headers
) -> None:
    """An empty queue is the normal state of a healthy platform. A 404 here would
    teach every agent to treat "nothing to do" as a failure."""
    agent = _running_agent(client, owner_headers)

    response = _claim(client, _device_headers(agent["device_token"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page"] is None
    assert body["poll_after_seconds"] > 0
    assert body["lease_seconds"] > 0
    assert body["server_time"]
    assert "error" not in body


# ---------------------------------------------------------------------------
# Two agents never receive the same page
# ---------------------------------------------------------------------------


def test_two_agents_polling_concurrently_never_receive_the_same_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """``FOR UPDATE SKIP LOCKED`` against live Postgres, from real concurrent
    HTTP requests. A second claimer skips the locked row and takes the next one;
    it never waits on the first and never sees the same id."""
    first_agent = _running_agent(client, owner_headers, name="Laptop one")
    second_agent = _running_agent(client, owner_headers, name="Laptop two")
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    submitted = [_submit_segment(client, owner_headers, owner_project, ids) for _ in range(6)]

    tokens = [first_agent["device_token"], second_agent["device_token"]] * 4
    with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
        responses = list(pool.map(lambda token: _claim(client, _device_headers(token)), tokens))

    assert all(response.status_code == 200 for response in responses)
    claimed = [
        response.json()["page"]["product_job_id"]
        for response in responses
        if response.json()["page"] is not None
    ]
    assert len(claimed) == len(set(claimed)), "the same page was handed to two agents"
    assert set(claimed) == set(submitted)
    with sync_system_session() as session:
        statuses = (
            session.execute(select(Job.status).where(Job.id.in_([uuid.UUID(j) for j in submitted])))
            .scalars()
            .all()
        )
    assert set(statuses) == {JobStatus.waiting}


def test_many_agents_can_long_poll_while_ordinary_traffic_is_served(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """More concurrent long polls than the connection pool has connections.

    ``DB_POOL_SIZE + DB_MAX_OVERFLOW`` is 15 on the defaults, which is the
    fifteen-device ceiling ADR 0001 names. A claim that took ``Depends(get_db)``
    would hold one of those for the whole poll; twenty of them would leave
    nothing for the researcher's own browser. The structural half of this
    guarantee - that the route declares no session dependency at all - is
    asserted in the unit suite, which is where a precise answer belongs.
    """
    agent = _running_agent(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    _prefer_local(client, owner_headers)
    job_id = _submit_segment(client, owner_headers, owner_project, ids)
    # Claim it so the pollers below find an empty queue and actually wait.
    assert _claim(client, _device_headers(agent["device_token"])).json()["page"] is not None

    headers = _device_headers(agent["device_token"])
    with ThreadPoolExecutor(max_workers=21) as pool:
        pollers = [pool.submit(_claim, client, headers, wait_seconds=2) for _ in range(20)]
        browsing = pool.submit(client.get, f"/jobs/{job_id}", headers=owner_headers)
        read = browsing.result()
        claims = [poller.result() for poller in pollers]

    assert read.status_code == 200, read.text
    assert read.json()["status"] == "waiting"
    assert all(response.status_code == 200 for response in claims)
    assert all(response.json()["page"] is None for response in claims)


# ---------------------------------------------------------------------------
# Completion and failure go through the existing callback contract
# ---------------------------------------------------------------------------


def _segment_done_body(page: dict) -> dict:
    return {
        "inference_job_id": page["inference_job_id"],
        "product_job_id": page["product_job_id"],
        "task": "segment",
        "status": "done",
        "output": {
            "kind": "segment",
            "data": {
                "lines": [
                    {
                        "external_id": "l1",
                        "order": 0,
                        "baseline": {"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                        "points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                    }
                ]
            },
        },
    }


def _segment_failed_body(page: dict) -> dict:
    return {
        "inference_job_id": page["inference_job_id"],
        "product_job_id": page["product_job_id"],
        "task": "segment",
        "status": "failed",
        "error": "the model would not load on this machine",
    }


def _claimed_page(client: TestClient, owner_headers, owner_project) -> tuple[dict, dict]:
    agent = _running_agent(client, owner_headers)
    _prefer_local(client, owner_headers)
    ids = _make_part(client, owner_headers, owner_project)
    _submit_segment(client, owner_headers, owner_project, ids)
    page = _claim(client, _device_headers(agent["device_token"])).json()["page"]
    assert page is not None
    return agent, page


def test_completion_flows_through_the_existing_callback_contract(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """Not a new endpoint and not a new payload: the agent posts the same
    ``JobCallbackRequest`` the inference service always posted, and the same
    ``JobCallbackService`` merges it."""
    agent, page = _claimed_page(client, owner_headers, owner_project)

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json=_segment_done_body(page),
    )

    assert response.status_code == 204, response.text
    stored = _stored_job(page["product_job_id"])
    assert stored.status is JobStatus.done
    assert stored.result["lines_count"] == 1
    body = client.get(f"/jobs/{page['product_job_id']}", headers=owner_headers).json()
    assert body["status"] == "done"


def test_failure_flows_through_the_existing_callback_contract(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent, page = _claimed_page(client, owner_headers, owner_project)

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(agent["device_token"]),
        json=_segment_failed_body(page),
    )

    assert response.status_code == 204, response.text
    stored = _stored_job(page["product_job_id"])
    assert stored.status is JobStatus.failed
    assert "would not load" in stored.error


def test_an_agent_cannot_report_on_a_page_it_is_not_holding(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    agent, page = _claimed_page(client, owner_headers, owner_project)
    other = pair_device_over_http(client, owner_headers, name="Another laptop")

    response = client.post(
        CALLBACK_URL,
        headers=_device_headers(other["device_token"]),
        json=_segment_done_body(page),
    )

    assert response.status_code == 403, response.text
    assert _stored_job(page["product_job_id"]).status is JobStatus.waiting


def test_a_hosted_worker_reports_its_own_cloud_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    _running_cloud_worker(client)
    ids = _make_part(client, owner_headers, owner_project)
    _submit_segment(client, owner_headers, owner_project, ids)
    page = client.post(CLAIM_URL, headers=SERVICE_HEADERS, json={"wait_seconds": 0}).json()["page"]
    assert page is not None

    response = client.post(CALLBACK_URL, headers=SERVICE_HEADERS, json=_segment_done_body(page))

    assert response.status_code == 204, response.text
    assert _stored_job(page["product_job_id"]).status is JobStatus.done


def test_the_platform_webhook_credential_cannot_take_a_leased_page(
    client: TestClient, owner_user, owner_headers, owner_project
) -> None:
    """The platform secret authenticates the platform, not a claim on this page.

    This used to return 204 and mark the job done. The webhook branch checked the
    secret and nothing else, so a holder of INFERENCE_WEBHOOK_SECRET could
    complete or fail a page a researcher's laptop was in the middle of running -
    discarding that run, and leaving the agent's own callback to be rejected as a
    duplicate. The agent credentials are narrowed to the page they hold; this one
    is now narrowed to the pages no agent holds.

    The credential has to be a genuinely valid one, and this test has to be able
    to say so. It used to send the string literal from the conftest default,
    which is only the configured secret while nobody has
    ``INFERENCE_WEBHOOK_SECRET`` exported already - ``setdefault`` yields to an
    existing value. With one exported, the header became a *wrong* secret, which
    the endpoint also refuses with 403, and whose detail the error handler
    redacts to the same "Access denied" body. The test stayed green while proving
    only that bad credentials are rejected, which is a different test's job.

    So the secret is read from settings rather than retyped, and the probe below
    shows the credential really is accepted before the leased page refuses it.
    """
    _agent, page = _claimed_page(client, owner_headers, owner_project)
    # `isolated_platform_state` reset the settings caches for this test, so this
    # is the same object the endpoint's dependency will compare against.
    configured_secret = get_inference_settings().inference_webhook_secret
    assert configured_secret, "INFERENCE_WEBHOOK_SECRET must be configured for this test"
    webhook_headers = {INFERENCE_WEBHOOK_SECRET_HEADER: configured_secret}

    # Positive control, and the thing that makes the 403 below mean what it says:
    # a request whose only fault is an unknown job gets *past* the credential
    # check and lands on 404. A wrong secret never reaches that far - it is 403,
    # with a body indistinguishable from the refusal being asserted below.
    accepted = client.post(
        CALLBACK_URL,
        headers=webhook_headers,
        json=_segment_done_body(
            {"inference_job_id": str(uuid.uuid4()), "product_job_id": str(uuid.uuid4())}
        ),
    )
    assert accepted.status_code == 404, accepted.text

    response = client.post(CALLBACK_URL, headers=webhook_headers, json=_segment_done_body(page))

    assert response.status_code == 403, response.text
    # Still leased, and still the agent's to report on.
    assert _stored_job(page["product_job_id"]).status is JobStatus.waiting
