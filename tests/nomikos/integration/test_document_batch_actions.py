"""Document-level batch actions, over HTTP against live Postgres.

The unit of work in this platform is one page, and that does not change here: a batch
action is a fan-out that decides *which* pages, then hands each one to the same per-part
enqueue service. So what these tests hold to account is the deciding, not the enqueuing:

* the additive scopes really do skip the pages that already have the work done, because
  that is the only thing standing between "segment the pages that need it" and losing a
  chapter's transcriptions;
* the counts the menu renders match, exactly, what the corresponding scope would queue -
  a label that says 12 and an action that queues 9 is worse than no label;
* a batch with nothing to do succeeds rather than erroring, on an empty document and on
  a finished one alike;
* no capacity refuses the whole batch before the first job, so a chapter is never left
  half queued.

Segment and transcribe jobs stay ``pending`` throughout: the platform worker leaves
inference job types for an agent to claim over HTTP, and no agent runs in this suite.
That is what lets these tests read job rows straight out of Postgres.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.document.infrastructure.orm_models import (
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from infrastructure.db import sync_system_session
from tests.nomikos.integration.helpers import (
    MINIMAL_PNG,
    assert_api_error,
    documents_url,
    pair_inference_device,
    return_pooled_connections_before_leaving,  # noqa: F401 - autouse fixture
    seed_transcription_model,
    user_id_for_email,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# A chapter, and the two things that can have been done to a page of it
# ---------------------------------------------------------------------------


def _document_with_pages(
    client: TestClient, headers: dict[str, str], project_id: str, *, pages: int
) -> tuple[str, list[str]]:
    """A document with ``pages`` uploaded page images. Returns ``(id, part_ids)``."""
    base = documents_url(project_id)
    created = client.post(base, headers=headers, json={"name": "Chapter"})
    assert created.status_code == 201, created.text
    document_id = created.json()["id"]
    part_ids = []
    for _ in range(pages):
        upload = client.post(
            f"{base}/{document_id}/parts",
            headers=headers,
            files={"file": ("page.png", MINIMAL_PNG, "image/png")},
        )
        assert upload.status_code == 201, upload.text
        part_ids.append(upload.json()["id"])
    return document_id, part_ids


def _segment_page(
    client: TestClient, headers: dict[str, str], project_id: str, document_id: str, part_id: str
) -> str:
    """Give a page one line, which is what makes it segmented as far as SQL can tell."""
    response = client.post(
        f"{documents_url(project_id)}/{document_id}/parts/{part_id}/lines",
        headers=headers,
        json={"order": 0, "kind": "rectangle", "points": [[0, 0], [10, 0], [10, 5], [0, 5]]},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _write_transcription_text(document_id: str, part_id: str, text: str = "μῆνιν ἄειδε") -> None:
    """Attach text to every line of a page, the way a finished transcribe job would.

    Written directly rather than through a job, because running one would need a paired
    agent and real weights; the batch routes only ever read the *result* of that, which
    is a line transcription carrying text.
    """
    with sync_system_session() as session:
        transcription = Transcription(
            document_id=uuid.UUID(document_id),
            name="Model transcription",
            kind=TranscriptionKind.model,
        )
        session.add(transcription)
        session.flush()
        line_ids = session.execute(
            select(Line.id).where(Line.part_id == uuid.UUID(part_id))
        ).scalars()
        for line_id in line_ids:
            session.add(
                LineTranscription(line_id=line_id, transcription_id=transcription.id, text=text)
            )
        session.commit()


def _stored_jobs(document_id: str, job_type: JobType) -> list[Job]:
    with sync_system_session() as session:
        return list(
            session.execute(
                select(Job).where(Job.document_id == uuid.UUID(document_id), Job.type == job_type)
            )
            .scalars()
            .all()
        )


def _give_cloud_capacity(email: str) -> None:
    """Submission is gated on capacity, so any test expecting a 202 has to say so."""
    pair_inference_device(user_id=user_id_for_email(email), host="cloud")


def _batch_url(project_id: str, document_id: str, action: str) -> str:
    return f"{documents_url(project_id)}/{document_id}/{action}"


# ---------------------------------------------------------------------------
# The additive scope skips work already done
# ---------------------------------------------------------------------------
# Tests that "segment unsegmented pages" leaves segmented pages alone, and that the
# default body is that scope. Does not test what the segment job itself produces.


@pytest.mark.integration
def test_segmenting_unsegmented_pages_skips_the_pages_that_already_have_lines(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=4)
    _segment_page(client, owner_headers, project_id, document_id, part_ids[0])
    _segment_page(client, owner_headers, project_id, document_id, part_ids[2])
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "unsegmented", "model_id": None},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["queued"] == 2
    assert body["skipped"] == 2
    assert len(body["jobs"]) == 2
    # A batch announces the host per job, at submission, the same way a single
    # enqueue does: the researcher is told where the chapter went before the
    # first poll, not after it.
    for job in body["jobs"]:
        assert job["execution_target"] == "cloud"
        assert job["preferred_execution_target"] == "cloud"
        assert job["execution_target_substituted"] is False
    queued_parts = {str(job.document_part_id) for job in _stored_jobs(document_id, JobType.segment)}
    assert queued_parts == {part_ids[1], part_ids[3]}


@pytest.mark.integration
def test_a_segment_request_with_no_body_uses_the_scope_that_cannot_lose_work(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=3)
    _segment_page(client, owner_headers, project_id, document_id, part_ids[0])
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"), headers=owner_headers
    )

    assert response.status_code == 202, response.text
    assert response.json()["queued"] == 2
    queued_parts = {str(job.document_part_id) for job in _stored_jobs(document_id, JobType.segment)}
    assert part_ids[0] not in queued_parts


# ---------------------------------------------------------------------------
# The destructive scope has to be asked for by name
# ---------------------------------------------------------------------------
# Tests scope="all" reaches every page, and that an unrecognised scope is refused rather
# than quietly widened. Does not test the deletion itself, which the segment job owns.


@pytest.mark.integration
def test_re_segmenting_every_page_is_available_but_only_when_named(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=3)
    _segment_page(client, owner_headers, project_id, document_id, part_ids[0])
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["queued"] == 3
    assert body["skipped"] == 0
    queued_parts = {str(job.document_part_id) for job in _stored_jobs(document_id, JobType.segment)}
    assert queued_parts == set(part_ids)


@pytest.mark.integration
def test_an_unrecognised_scope_is_refused_rather_than_widened(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=2)
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "everything"},
    )

    assert response.status_code == 422
    assert_api_error(response, code="VALIDATION_ERROR")
    assert _stored_jobs(document_id, JobType.segment) == []


# ---------------------------------------------------------------------------
# Transcription follows segmentation
# ---------------------------------------------------------------------------
# Tests "unpaired" skips pages that already carry text, and that neither scope reaches a
# page with no segments. Does not test line selection, which is a per-part concern.


@pytest.mark.integration
def test_transcribing_unpaired_pages_skips_pages_that_already_carry_text(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=3)
    for part_id in part_ids:
        _segment_page(client, owner_headers, project_id, document_id, part_id)
    _write_transcription_text(document_id, part_ids[1])
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/transcribe"),
        headers=owner_headers,
        json={"scope": "unpaired", "model_id": seed_transcription_model()},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["queued"] == 2
    assert body["skipped"] == 1
    queued_parts = {
        str(job.document_part_id) for job in _stored_jobs(document_id, JobType.transcribe)
    }
    assert queued_parts == {part_ids[0], part_ids[2]}


@pytest.mark.integration
def test_transcribing_every_page_still_leaves_out_the_pages_with_no_segments(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=3)
    _segment_page(client, owner_headers, project_id, document_id, part_ids[0])
    _write_transcription_text(document_id, part_ids[0])
    _segment_page(client, owner_headers, project_id, document_id, part_ids[1])
    _give_cloud_capacity(owner_user["email"])

    response = client.post(
        _batch_url(project_id, document_id, "jobs/transcribe"),
        headers=owner_headers,
        json={"scope": "all", "model_id": seed_transcription_model()},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    # Both segmented pages, including the one that already has text. The third page has
    # no lines, so there is nothing for a transcribe job to run over.
    assert body["queued"] == 2
    assert body["skipped"] == 1
    queued_parts = {
        str(job.document_part_id) for job in _stored_jobs(document_id, JobType.transcribe)
    }
    assert queued_parts == {part_ids[0], part_ids[1]}


# ---------------------------------------------------------------------------
# A batch with nothing to do is a completed request
# ---------------------------------------------------------------------------
# Tests the two ways a scope can match no page. Does not test capacity, which is checked
# only once there is work to do.


@pytest.mark.integration
def test_a_document_with_no_pages_returns_a_queued_count_of_zero(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=0)
    _give_cloud_capacity(owner_user["email"])

    for action in ("jobs/segment", "jobs/transcribe"):
        response = client.post(_batch_url(project_id, document_id, action), headers=owner_headers)
        assert response.status_code == 202, response.text
        assert response.json() == {"jobs": [], "queued": 0, "skipped": 0}


@pytest.mark.integration
def test_a_fully_segmented_chapter_queues_nothing_and_does_not_error(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=2)
    for part_id in part_ids:
        _segment_page(client, owner_headers, project_id, document_id, part_id)

    # Deliberately no capacity: a batch that was never going to write a job must not be
    # refused for want of a host it would not have used.
    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "unsegmented"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["queued"] == 0
    assert response.json()["skipped"] == 2


# ---------------------------------------------------------------------------
# No capacity refuses the whole batch, not half of it
# ---------------------------------------------------------------------------
# Tests that a chapter is never left partly queued when no inference host is running.
# Does not test target substitution, which test_execution_target.py owns.


@pytest.mark.integration
def test_no_inference_host_refuses_before_a_single_job_is_written(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=3)

    response = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )

    assert response.status_code == 409
    assert_api_error(response, code="CONFLICT")
    assert _stored_jobs(document_id, JobType.segment) == []


# ---------------------------------------------------------------------------
# A page already queued is not queued twice
# ---------------------------------------------------------------------------
# Tests the in-flight guard, which is what makes the menu safe to press twice. Does not
# test finished jobs, which are deliberately no bar to a re-run.


@pytest.mark.integration
def test_a_page_with_a_job_still_in_flight_is_skipped_by_the_next_batch(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=3)
    _give_cloud_capacity(owner_user["email"])

    first = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )
    assert first.status_code == 202, first.text
    assert first.json()["queued"] == 3

    second = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )

    assert second.status_code == 202, second.text
    assert second.json() == {"jobs": [], "queued": 0, "skipped": 3}
    assert len(_stored_jobs(document_id, JobType.segment)) == 3
    assert len(part_ids) == 3


@pytest.mark.integration
def test_a_finished_job_does_not_block_a_re_run(client, owner_headers, owner_project, owner_user):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=2)
    _give_cloud_capacity(owner_user["email"])

    first = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )
    assert first.status_code == 202, first.text
    with sync_system_session() as session:
        for job in session.execute(select(Job)).scalars().all():
            job.status = JobStatus.done
        session.commit()

    second = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "all"},
    )

    assert second.status_code == 202, second.text
    assert second.json()["queued"] == 2


# ---------------------------------------------------------------------------
# The counts the menu renders
# ---------------------------------------------------------------------------
# Tests the four numbers against a document in every state at once, and against what the
# scopes actually queue. Does not test pagination or ordering; there is neither.


@pytest.mark.integration
def test_workflow_counts_report_every_state_of_a_mixed_chapter(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=5)
    # page 0: nothing at all. page 1: segmented, no text. page 2: segmented and
    # transcribed. page 3: segmented, transcribed, reviewed. page 4: nothing at all.
    for part_id in part_ids[1:4]:
        _segment_page(client, owner_headers, project_id, document_id, part_id)
    _write_transcription_text(document_id, part_ids[2])
    _write_transcription_text(document_id, part_ids[3])
    reviewed = client.patch(
        f"{documents_url(project_id)}/{document_id}/parts/{part_ids[3]}",
        headers=owner_headers,
        json={"reviewed": True},
    )
    assert reviewed.status_code == 200, reviewed.text

    response = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=owner_headers
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"total": 5, "reviewed": 1, "unsegmented": 2, "unpaired": 1}


@pytest.mark.integration
def test_blank_transcription_text_still_counts_the_page_as_unpaired(
    client, owner_headers, owner_project
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=1)
    _segment_page(client, owner_headers, project_id, document_id, part_ids[0])
    _write_transcription_text(document_id, part_ids[0], text="   ")

    response = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=owner_headers
    )

    assert response.status_code == 200, response.text
    assert response.json()["unpaired"] == 1


@pytest.mark.integration
def test_the_counts_match_what_the_matching_scope_queues(
    client, owner_headers, owner_project, owner_user
):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=6)
    for part_id in part_ids[:4]:
        _segment_page(client, owner_headers, project_id, document_id, part_id)
    _write_transcription_text(document_id, part_ids[0])
    _give_cloud_capacity(owner_user["email"])

    counts = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=owner_headers
    ).json()
    segmented = client.post(
        _batch_url(project_id, document_id, "jobs/segment"),
        headers=owner_headers,
        json={"scope": "unsegmented"},
    )
    transcribed = client.post(
        _batch_url(project_id, document_id, "jobs/transcribe"),
        headers=owner_headers,
        json={"scope": "unpaired", "model_id": seed_transcription_model()},
    )

    assert segmented.status_code == 202, segmented.text
    assert transcribed.status_code == 202, transcribed.text
    assert counts["unsegmented"] == segmented.json()["queued"] == 2
    assert counts["unpaired"] == transcribed.json()["queued"] == 3
    assert counts["total"] == 6


@pytest.mark.integration
def test_counts_on_a_document_with_no_pages_are_all_zero(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=0)

    response = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=owner_headers
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"total": 0, "reviewed": 0, "unsegmented": 0, "unpaired": 0}


@pytest.mark.integration
def test_counts_are_scoped_to_the_document_asked_about(client, owner_headers, owner_project):
    project_id = owner_project["id"]
    document_id, part_ids = _document_with_pages(client, owner_headers, project_id, pages=2)
    other_id, other_parts = _document_with_pages(client, owner_headers, project_id, pages=3)
    _segment_page(client, owner_headers, project_id, other_id, other_parts[0])

    response = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=owner_headers
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"total": 2, "reviewed": 0, "unsegmented": 2, "unpaired": 0}
    assert len(part_ids) == 2


# ---------------------------------------------------------------------------
# Who may run a batch
# ---------------------------------------------------------------------------
# Tests membership is the gate: a collaborator may segment a chapter, an outsider may not
# read its counts. Does not test the anonymous surface, which carries none of these routes.


@pytest.mark.integration
def test_a_collaborator_may_run_a_batch_without_owning_the_project(
    client, owner_headers, owner_project, owner_user, collaborator_user, collaborator_headers
):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=2)
    shared = client.post(
        f"/projects/{project_id}/share",
        headers=owner_headers,
        json={"username": collaborator_user["username"]},
    )
    assert shared.status_code == 204, shared.text
    _give_cloud_capacity(collaborator_user["email"])

    counts = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=collaborator_headers
    )
    queued = client.post(
        _batch_url(project_id, document_id, "jobs/segment"), headers=collaborator_headers
    )

    assert counts.status_code == 200, counts.text
    assert queued.status_code == 202, queued.text
    assert queued.json()["queued"] == 2


@pytest.mark.integration
def test_an_outsider_reaches_neither_the_counts_nor_the_batch(
    client, owner_headers, owner_project, outsider_headers
):
    project_id = owner_project["id"]
    document_id, _ = _document_with_pages(client, owner_headers, project_id, pages=2)

    counts = client.get(
        _batch_url(project_id, document_id, "workflow-counts"), headers=outsider_headers
    )
    queued = client.post(
        _batch_url(project_id, document_id, "jobs/segment"), headers=outsider_headers
    )

    assert counts.status_code == 403
    assert queued.status_code == 403
    assert _stored_jobs(document_id, JobType.segment) == []


@pytest.mark.integration
def test_a_document_in_another_project_is_reported_as_missing(
    client, owner_headers, owner_project, collaborator_user, collaborator_headers
):
    other_project = client.post(
        "/projects",
        headers=collaborator_headers,
        json={"slug": f"proj-{uuid.uuid4().hex[:8]}", "name": "Elsewhere"},
    )
    assert other_project.status_code == 201, other_project.text
    document_id, _ = _document_with_pages(
        client, collaborator_headers, other_project.json()["id"], pages=1
    )

    response = client.get(
        _batch_url(owner_project["id"], document_id, "workflow-counts"), headers=owner_headers
    )

    assert response.status_code == 404
    assert_api_error(response, code="NOT_FOUND")
