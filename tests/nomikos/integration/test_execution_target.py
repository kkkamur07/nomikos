"""**Execution target** and **capacity** gating, over HTTP against live Postgres.

Everything here goes through the real ``create_app()`` and the real database.
ADR 0001 records why that is not negotiable: an earlier device layer was never
mounted, and the integration suite hid it by building its own FastAPI app in the
test file - an entire phase unreachable behind a green suite. So there is no
local app here, and no substitute for Postgres.

**Capacity is time-dependent, and it is controlled here by writing
``last_seen_at``**, not by patching a clock. That is the actual production
signal: a host has capacity when one of its devices was seen inside
``DEVICE_IDLE_WINDOW_SECONDS``. A test that froze time would prove the freezing
worked; a test that writes a timestamp proves the query does.

The four submission outcomes, which are the whole of this issue:

1. the preferred host has capacity - the job goes there and says so;
2. the preferred host does not and the other does - the job goes to the other
   *and says so*, never silently, and says so **in the enqueue response itself**:
   the target is fixed at submission, so it is announced at submission, not on
   the first status update (issue 64);
3. neither has capacity - submission is refused with a reason, rather than
   creating a job nobody will claim;
4. a submitted job's target cannot be changed - not through the mapper, and not
   through raw SQL either.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

import infrastructure.models  # noqa: F401 — register all ORM mappers
from backend.core.settings.device import get_device_settings
from backend.jobs.infrastructure.job_claim_engine import mark_job_failed
from backend.jobs.infrastructure.orm_models import Job, JobStatus
from backend.ml.domain.execution import ExecutionTarget
from infrastructure.db import sync_system_session
from tests.nomikos.integration.helpers import (
    MINIMAL_PNG,
    assert_api_error,
    documents_url,
    pair_inference_device,
    seed_transcription_model,
    user_id_for_email,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Live fixtures: real rows, real timestamps
# ---------------------------------------------------------------------------


def _stale_seconds() -> float:
    """One second past the window that still counts a device as capacity."""
    return get_device_settings().device_idle_window_seconds + 1


def _make_part(client: TestClient, headers: dict[str, str], project: dict) -> tuple[str, str]:
    base = documents_url(project["id"])
    created = client.post(base, headers=headers, json={"name": "Execution target"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    upload = client.post(
        f"{base}/{document_id}/parts",
        headers=headers,
        files={"file": ("page.png", MINIMAL_PNG, "image/png")},
    )
    assert upload.status_code == 201, upload.text
    return document_id, upload.json()["id"]


def _segment(client: TestClient, headers: dict[str, str], project: dict, ids: tuple[str, str]):
    document_id, part_id = ids
    return client.post(
        f"{documents_url(project['id'])}/{document_id}/parts/{part_id}/segment",
        headers=headers,
    )


def _stored_job(job_id: str) -> Job:
    with sync_system_session() as session:
        return session.execute(select(Job).where(Job.id == uuid.UUID(job_id))).scalar_one()


ANNOUNCEMENT_FIELDS = (
    "execution_target",
    "preferred_execution_target",
    "execution_target_substituted",
)


def _announced(body: dict) -> dict:
    """The three fields that make up the announcement, off any job-shaped body.

    Both the 202 from an enqueue route and the job read back from ``/jobs/{id}``
    carry them, and the test for "never disagree" is that these two dicts are
    equal.
    """
    return {field: body[field] for field in ANNOUNCEMENT_FIELDS}


# ---------------------------------------------------------------------------
# Outcome 1: the preferred host has capacity
# ---------------------------------------------------------------------------


def test_a_job_runs_on_the_preferred_host_when_it_has_capacity(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    enqueued = response.json()
    job_id = enqueued["job_id"]
    # The researcher is always told which host will run the job - including when
    # nothing unusual happened - and told at submission, in the 202 itself.
    assert _announced(enqueued) == {
        "execution_target": "cloud",
        "preferred_execution_target": "cloud",
        "execution_target_substituted": False,
    }
    read = client.get(f"/jobs/{job_id}", headers=owner_headers)
    assert read.status_code == 200
    assert _announced(read.json()) == _announced(enqueued)
    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud


def test_the_account_setting_sends_a_job_to_the_researchers_own_computer(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """ "Use my computer when it is available", and it is."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=5)
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    setting = client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    assert setting.status_code == 200, setting.text
    assert setting.json() == {
        "prefer_local_inference": True,
        "preferred_execution_target": "local",
        "available_targets": ["cloud", "local"],
    }
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    body = client.get(f"/jobs/{response.json()['job_id']}", headers=owner_headers).json()
    assert body["execution_target"] == "local"
    assert body["execution_target_substituted"] is False


def test_another_researchers_laptop_is_not_capacity_for_this_account(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    outsider_user: dict[str, str],
    owner_project,
) -> None:
    """``local`` capacity is scoped to the owner; ``cloud`` is not. A stranger's
    running laptop must never make one of my jobs submittable."""
    pair_inference_device(
        user_id=user_id_for_email(outsider_user["email"]), host="local", seen_seconds_ago=5
    )
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 409, response.text
    assert (
        client.get("/account/execution-target", headers=owner_headers).json()["available_targets"]
        == []
    )


# ---------------------------------------------------------------------------
# Outcome 2: substituted, and the job says so
# ---------------------------------------------------------------------------


def test_an_unavailable_preferred_host_substitutes_and_the_job_reports_it(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The laptop is paired but has not checked in inside the window, so it has no
    **capacity**. The job goes to the cloud - and the substitution is recorded on
    the job itself, not announced once into a toast the researcher may not see."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=_stale_seconds())
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    body = client.get(f"/jobs/{response.json()['job_id']}", headers=owner_headers).json()
    assert body["preferred_execution_target"] == "local"
    assert body["execution_target"] == "cloud"
    assert body["execution_target_substituted"] is True


def test_the_substitution_is_announced_in_the_enqueue_response_itself(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The researcher asked for their laptop; it is paired but has not checked in,
    so the job goes to the cloud. That has to be said in the 202, not first on
    the next poll: between the click and the first status update the interface
    would otherwise have nothing to say about where the job went, and a
    downgrade the researcher does not notice is the silent fallback ADR 0002
    rejected, arrived at by another route.

    The response and the job read back afterwards must agree: both are the
    same three columns, mapped once."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=_stale_seconds())
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 202, response.text
    enqueued = response.json()
    assert _announced(enqueued) == {
        "execution_target": "cloud",
        "preferred_execution_target": "local",
        "execution_target_substituted": True,
    }
    later = client.get(f"/jobs/{enqueued['job_id']}", headers=owner_headers).json()
    assert _announced(later) == _announced(enqueued)


def test_a_transcription_announces_its_host_at_submission_too(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """Same contract on the other enqueue route: the two share one mapping from
    the job row, so neither can drift from the other or from ``/jobs/{id}``."""
    user_id = user_id_for_email(owner_user["email"])
    pair_inference_device(user_id=user_id, host="local", seen_seconds_ago=_stale_seconds())
    pair_inference_device(user_id=user_id, host="cloud", seen_seconds_ago=5)
    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )
    document_id, part_id = _make_part(client, owner_headers, owner_project)
    base = documents_url(owner_project["id"])
    # Transcription needs a line to transcribe; one hand-drawn rectangle will do.
    drawn = client.post(
        f"{base}/{document_id}/parts/{part_id}/lines",
        headers=owner_headers,
        json={"order": 0, "kind": "rectangle", "points": [[0, 0], [10, 0], [10, 5], [0, 5]]},
    )
    assert drawn.status_code == 201, drawn.text

    response = client.post(
        f"{base}/{document_id}/parts/{part_id}/transcribe",
        headers=owner_headers,
        json={"model_id": seed_transcription_model()},
    )

    assert response.status_code == 202, response.text
    enqueued = response.json()
    assert _announced(enqueued) == {
        "execution_target": "cloud",
        "preferred_execution_target": "local",
        "execution_target_substituted": True,
    }
    later = client.get(f"/jobs/{enqueued['job_id']}", headers=owner_headers).json()
    assert _announced(later) == _announced(enqueued)


# ---------------------------------------------------------------------------
# Outcome 3: neither host has capacity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seen_seconds_ago", [None, 901.0])
def test_no_capacity_anywhere_refuses_the_submission_and_writes_no_job(
    client: TestClient,
    owner_user: dict[str, str],
    owner_headers: dict[str, str],
    owner_project,
    seen_seconds_ago: float | None,
) -> None:
    """Refusing is the point. A job created for a host nobody is claiming from
    has no terminal outcome - it is the failure mode ``local_only`` had, and it
    is why there is no hold window and no unclaimed-job sweeper."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]),
        host="local",
        seen_seconds_ago=seen_seconds_ago,
    )
    ids = _make_part(client, owner_headers, owner_project)

    response = _segment(client, owner_headers, owner_project, ids)

    assert response.status_code == 409, response.text
    error = assert_api_error(response, code="CONFLICT")
    assert "No inference host is available" in error["message"]
    with sync_system_session() as session:
        assert session.execute(select(Job)).scalars().all() == []


# ---------------------------------------------------------------------------
# Outcome 4: a submitted job's target cannot be changed
# ---------------------------------------------------------------------------


def test_a_submitted_jobs_execution_target_cannot_be_changed_by_raw_sql(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The application guard binds statements that go through the mapper. The
    platform also issues bulk ``UPDATE``s against ``jobs`` - the stale sweep, the
    callback path - so the rule is in Postgres as well."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    with sync_system_session() as session:
        with pytest.raises(Exception, match="fixed at submission"):
            session.execute(
                text("UPDATE jobs SET execution_target = 'local' WHERE id = :id"),
                {"id": uuid.UUID(job_id)},
            )
        session.rollback()

    assert _stored_job(job_id).execution_target is ExecutionTarget.cloud


def test_an_ordinary_update_that_leaves_the_target_alone_still_works(
    client: TestClient, owner_user: dict[str, str], owner_headers: dict[str, str], owner_project
) -> None:
    """The guard must be a guard, not a lock: every other column stays writable."""
    pair_inference_device(
        user_id=user_id_for_email(owner_user["email"]), host="cloud", seen_seconds_ago=5
    )
    ids = _make_part(client, owner_headers, owner_project)
    job_id = _segment(client, owner_headers, owner_project, ids).json()["job_id"]

    mark_job_failed(uuid.UUID(job_id), "inference failed", claimed_by=None)

    stored = _stored_job(job_id)
    assert stored.status is JobStatus.failed
    assert stored.execution_target is ExecutionTarget.cloud


# ---------------------------------------------------------------------------
# The account setting
# ---------------------------------------------------------------------------


def test_the_execution_target_preference_persists_and_is_readable(
    client: TestClient, owner_headers: dict[str, str]
) -> None:
    assert client.get("/account/execution-target", headers=owner_headers).json() == {
        "prefer_local_inference": False,
        "preferred_execution_target": "cloud",
        "available_targets": [],
    }

    client.put(
        "/account/execution-target",
        headers=owner_headers,
        json={"prefer_local_inference": True},
    )

    assert client.get("/account/execution-target", headers=owner_headers).json() == {
        "prefer_local_inference": True,
        "preferred_execution_target": "local",
        "available_targets": [],
    }
    # Also on the account itself, so a client that already has the user does not
    # need a second round trip to render the setting.
    assert client.get("/me", headers=owner_headers).json()["prefer_local_inference"] is True


def test_the_preference_route_requires_a_logged_in_researcher(client: TestClient) -> None:
    assert client.get("/account/execution-target").status_code == 401
    assert (
        client.put("/account/execution-target", json={"prefer_local_inference": True}).status_code
        == 401
    )
