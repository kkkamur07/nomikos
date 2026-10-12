"""Inference job completion callback: webhook auth, status transition, idempotency."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.core.settings import get_inference_settings
from backend.document.api.line_responses import line_transcription_response
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineGeometryKind,
    LineSource,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)
from backend.jobs.application import job_callback_service
from backend.jobs.infrastructure.orm_models import Job, JobStatus, JobType
from backend.project.infrastructure.orm_models import Project
from infrastructure.db import sync_system_session
from nomikos_inference.contracts.webhooks import INFERENCE_WEBHOOK_SECRET_HEADER

CALLBACK_URL = "/internal/inference/job-complete"
WEBHOOK_HEADERS = {INFERENCE_WEBHOOK_SECRET_HEADER: "test-inference-webhook-secret"}


def _segment_done_payload(
    *,
    product_job_id: uuid.UUID,
    inference_job_id: uuid.UUID,
) -> dict:
    return {
        "inference_job_id": str(inference_job_id),
        "product_job_id": str(product_job_id),
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


def _transcribe_done_payload(
    *,
    product_job_id: uuid.UUID,
    inference_job_id: uuid.UUID,
    line_id: uuid.UUID,
) -> dict:
    return {
        "inference_job_id": str(inference_job_id),
        "product_job_id": str(product_job_id),
        "task": "transcribe",
        "status": "done",
        "output": {
            "kind": "transcribe",
            "data": {
                "lines": [
                    {
                        "line_id": str(line_id),
                        "line_index": 0,
                        "output": {
                            "text": "Αβ",
                            "confidence": 0.91,
                            "character_confidences": [
                                {"char": "Α", "confidence": 0.93},
                                {"char": "β", "confidence": 0.89},
                            ],
                        },
                    }
                ]
            },
        },
    }


def _seed_waiting_job(
    *,
    product_job_id: uuid.UUID | None = None,
    inference_job_id: uuid.UUID | None = None,
    job_type: JobType = JobType.segment,
) -> tuple[uuid.UUID, uuid.UUID]:
    product_job_id = product_job_id or uuid.uuid4()
    inference_job_id = inference_job_id or uuid.uuid4()
    document_id = None
    part_id = None
    with sync_system_session() as session:
        if job_type == JobType.segment:
            project_id = uuid.uuid4()
            document_id = uuid.uuid4()
            part_id = uuid.uuid4()
            session.add(
                Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}")
            )
            session.flush()
            session.add(
                Document(
                    id=document_id,
                    project_id=project_id,
                    name="Test document",
                )
            )
            session.flush()
            session.add(
                DocumentPart(
                    id=part_id,
                    document_id=document_id,
                    image_key="test/page.png",
                )
            )
            session.flush()
        session.add(
            Job(
                id=product_job_id,
                type=job_type,
                status=JobStatus.waiting,
                payload={},
                inference_job_id=inference_job_id,
                document_id=document_id,
                document_part_id=part_id,
            )
        )
        session.commit()
    return product_job_id, inference_job_id


def _seed_transcribe_waiting_job(
    *,
    product_job_id: uuid.UUID | None = None,
    inference_job_id: uuid.UUID | None = None,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    product_job_id = product_job_id or uuid.uuid4()
    inference_job_id = inference_job_id or uuid.uuid4()
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    part_id = uuid.uuid4()
    line_id = uuid.uuid4()
    with sync_system_session() as session:
        session.add(
            Project(id=project_id, name="Callback test", slug=f"callback-{uuid.uuid4().hex}")
        )
        session.flush()
        session.add(
            Document(
                id=document_id,
                project_id=project_id,
                name="Test document",
            )
        )
        session.flush()
        session.add(
            DocumentPart(
                id=part_id,
                document_id=document_id,
                image_key="test/page.png",
            )
        )
        session.flush()
        session.add(
            Line(
                id=line_id,
                part_id=part_id,
                baseline={"type": "LineString", "coordinates": [[1, 1], [2, 1]]},
                points=[[1, 1], [2, 1], [2, 2], [1, 2]],
                kind=LineGeometryKind.polygon,
            )
        )
        session.flush()
        session.add(
            Job(
                id=product_job_id,
                type=JobType.transcribe,
                status=JobStatus.waiting,
                payload={},
                inference_job_id=inference_job_id,
                document_id=document_id,
                document_part_id=part_id,
            )
        )
        session.commit()
    return product_job_id, inference_job_id, line_id


def _get_job(job_id: uuid.UUID) -> Job:
    with sync_system_session() as session:
        job = session.get(Job, job_id)
        assert job is not None
        session.expunge(job)
        return job


# --- Webhook authentication ---
# Tests secret header validation on the callback endpoint. Does not merge inference output into documents.


def test_callback_missing_secret_returns_401(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 401


def test_callback_wrong_secret_returns_403(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers={INFERENCE_WEBHOOK_SECRET_HEADER: "wrong-secret"},
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 403


def test_callback_unconfigured_secret_returns_503(client: TestClient, monkeypatch):
    # Patch the cached settings instance rather than the environment: settings
    # also read `backend/core/.env`, falling back to `.env.supabase`, and
    # `env_ignore_empty=True` makes an empty env var mean "not set" - so an
    # emptied variable would leave whatever the dotenv supplies and this would
    # assert nothing in a configured checkout.
    monkeypatch.setattr(get_inference_settings(), "inference_webhook_secret", None)
    product_job_id, inference_job_id = _seed_waiting_job()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 503
    get_inference_settings.cache_clear()


# --- Successful callbacks ---
# Tests waiting jobs transition to done with merged results. Does not run real inference.


def test_callback_success_marks_job_done(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.inference_job_id == inference_job_id
    assert job.error is None
    assert job.result is not None
    assert job.result["blocks_count"] == 0
    assert job.result["lines_count"] == 1
    assert job.result["added_lines"] == 1
    assert job.result["pruned_lines"] == 0
    assert job.result["preserved_manual_lines"] == 0
    assert job.result["preserved_transcribed_lines"] == 0
    assert job.completed_at is not None


def _line(
    part_id: uuid.UUID,
    order: int,
    *,
    manual_geometry: bool = False,
    points: list[list[int]] | None = None,
) -> Line:
    # Rows of ink well away from the one line the callback payload draws at y 1..2, so
    # a kept line and the fresh one never claim the same pixels unless a test says so.
    top = 100 + 10 * order
    return Line(
        id=uuid.uuid4(),
        part_id=part_id,
        baseline={"points": [[1, top + 1], [2, top + 1]]},
        kind=LineGeometryKind.polygon,
        points=points or [[1, top], [2, top], [2, top + 2], [1, top + 2]],
        source=LineSource.manual if manual_geometry else LineSource.kraken,
        manual_geometry=manual_geometry,
        order=order,
    )


def test_resegment_keeps_every_line_a_human_touched(client: TestClient):
    """Approving text or pairing a line is human work even when its polygon never moved.

    Before this guard the merge kept only ``manual_geometry`` lines, so a page whose text
    was corrected and approved without a single polygon being nudged lost all of it, and
    the pairing rows pointing at those lines were silently unlinked, to one re-segment.
    """
    product_job_id, inference_job_id = _seed_waiting_job()
    job = _get_job(product_job_id)
    part_id = job.document_part_id
    document_id = job.document_id
    assert part_id is not None and document_id is not None

    approved = _line(part_id, 0)
    paired = _line(part_id, 1)
    drawn = _line(part_id, 2, manual_geometry=True)
    untouched = _line(part_id, 3)
    with sync_system_session() as session:
        ground_truth = Transcription(
            id=uuid.uuid4(),
            document_id=document_id,
            name="Ground truth",
            kind=TranscriptionKind.ground_truth,
        )
        model_layer = Transcription(
            id=uuid.uuid4(), document_id=document_id, name="Model", kind=TranscriptionKind.model
        )
        session.add_all([ground_truth, model_layer, approved, paired, drawn, untouched])
        session.flush()
        session.add_all(
            [
                LineTranscription(
                    line_id=approved.id, transcription_id=ground_truth.id, text="approved"
                ),
                # The model's own output is exactly what a re-segment may replace.
                LineTranscription(
                    line_id=untouched.id, transcription_id=model_layer.id, text="guess"
                ),
                PageTranscriptionLine(
                    part_id=part_id, order=0, text="paired", paired_line_id=paired.id
                ),
            ]
        )
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 204

    with sync_system_session() as session:
        surviving = {
            line.id: line
            for line in session.scalars(select(Line).where(Line.part_id == part_id)).all()
        }
        assert approved.id in surviving
        assert paired.id in surviving
        assert drawn.id in surviving
        assert untouched.id not in surviving
        # Three kept plus the one line the job drew.
        assert len(surviving) == 4
        assert (
            session.scalars(
                select(LineTranscription.text).where(LineTranscription.line_id == approved.id)
            ).one()
            == "approved"
        )
        assert (
            session.scalars(
                select(PageTranscriptionLine.paired_line_id).where(
                    PageTranscriptionLine.part_id == part_id
                )
            ).one()
            == paired.id
        )

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.result["added_lines"] == 1
    assert job.result["pruned_lines"] == 1
    assert job.result["preserved_manual_lines"] == 1
    assert job.result["preserved_transcribed_lines"] == 2
    assert job.result["skipped_covered_lines"] == 0


def test_resegment_does_not_draw_a_kept_line_twice(client: TestClient):
    """The segmenter redraws the ink a kept line already covers; that twin is not added.

    Without this a fully approved page came back with every approved line under a fresh
    near-identical copy, which is better than losing the text but not what anyone asked
    for. The callback payload draws a line at exactly the approved line's polygon.
    """
    product_job_id, inference_job_id = _seed_waiting_job()
    job = _get_job(product_job_id)
    part_id = job.document_part_id
    document_id = job.document_id
    assert part_id is not None and document_id is not None

    approved = _line(part_id, 0, points=[[1, 1], [2, 1], [2, 2], [1, 2]])
    with sync_system_session() as session:
        ground_truth = Transcription(
            id=uuid.uuid4(),
            document_id=document_id,
            name="Ground truth",
            kind=TranscriptionKind.ground_truth,
        )
        session.add_all([ground_truth, approved])
        session.flush()
        session.add(
            LineTranscription(line_id=approved.id, transcription_id=ground_truth.id, text="word")
        )
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 204

    with sync_system_session() as session:
        surviving = session.scalars(select(Line.id).where(Line.part_id == part_id)).all()
        assert surviving == [approved.id]

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.result["lines_count"] == 1
    assert job.result["added_lines"] == 0
    assert job.result["skipped_covered_lines"] == 1
    assert job.result["preserved_transcribed_lines"] == 1


def test_callback_transcribe_success_marks_job_done(client: TestClient):
    product_job_id, inference_job_id, line_id = _seed_transcribe_waiting_job()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_transcribe_done_payload(
            product_job_id=product_job_id,
            inference_job_id=inference_job_id,
            line_id=line_id,
        ),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.inference_job_id == inference_job_id
    assert job.result is not None
    assert job.result["transcription_id"]
    assert job.result["lines"][0]["line_id"] == str(line_id)
    assert job.result["lines"][0]["text"] == "Αβ"
    assert job.result["lines"][0]["confidence"] == 0.91

    # Postgres round trip: the merge stored the per-character floats, and the
    # API serves them paired with the code points of the stored text.
    with sync_system_session() as session:
        row = (
            session.execute(select(LineTranscription).where(LineTranscription.line_id == line_id))
            .scalars()
            .one()
        )
        assert row.text == "Αβ"
        assert row.character_confidences == [0.93, 0.89]
        served = line_transcription_response(row)
    assert served.character_confidences is not None
    assert [(entry.char, entry.confidence) for entry in served.character_confidences] == [
        ("Α", 0.93),
        ("β", 0.89),
    ]


# --- Idempotency and validation ---
# Tests duplicate callbacks and mismatched job metadata. Does not enqueue new inference work.
# Failure persistence and error redaction are unit-tested against the same pure
# functions in unit/test_job_callback_service.py.


def test_callback_after_cancel_skips_merge(client: TestClient):
    """Cancelled waiting jobs stay cancelled; document merge must not run."""
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.status = JobStatus.cancelled
        job.completed_at = datetime.now(UTC)
        job.updated_at = job.completed_at
        session.commit()
        part_id = job.document_part_id
        assert part_id is not None

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 204

    job = _get_job(product_job_id)
    assert job.status == JobStatus.cancelled
    assert job.result is None

    with sync_system_session() as session:
        lines = list(session.execute(select(Line).where(Line.part_id == part_id)).scalars().all())
    assert lines == []


def test_parallel_callback_replays_merge_once(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    payload = _segment_done_payload(
        product_job_id=product_job_id, inference_job_id=inference_job_id
    )

    def deliver() -> int:
        return client.post(CALLBACK_URL, headers=WEBHOOK_HEADERS, json=payload).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(lambda _unused: deliver(), range(2)))

    assert statuses == [204, 204]
    job = _get_job(product_job_id)
    assert job.status == JobStatus.done
    assert job.result is not None
    assert job.result["lines_count"] == 1


def test_callback_task_mismatch_returns_409(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job(job_type=JobType.transcribe)
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 409


def test_callback_ml_job_mismatch_returns_409(client: TestClient):
    product_job_id, seeded_inference_job_id = _seed_waiting_job()
    callback_inference_job_id = uuid.uuid4()
    assert callback_inference_job_id != seeded_inference_job_id

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id,
            inference_job_id=callback_inference_job_id,
        ),
    )
    assert response.status_code == 409


def test_callback_rejects_unbound_inference_job(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.inference_job_id = None
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 409
    assert _get_job(product_job_id).status == JobStatus.waiting


def test_callback_requires_waiting_job_state(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.status = JobStatus.running
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 409
    assert _get_job(product_job_id).status == JobStatus.running


def test_callback_claimed_replay_is_not_merged_twice(client: TestClient):
    product_job_id, inference_job_id = _seed_waiting_job()
    with sync_system_session() as session:
        job = session.get(Job, product_job_id)
        assert job is not None
        job.callback_claimed_at = datetime.now(UTC)
        session.commit()

    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )

    assert response.status_code == 204
    job = _get_job(product_job_id)
    assert job.status == JobStatus.waiting
    assert job.result is None


# --- Merge/finalize atomicity ---
# Tests a raise after the document merge takes the merge down with it. Does not
# test the claim, which commits in its own transaction by design.


def test_finalize_failure_rolls_back_the_merged_lines(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Merge + finalize are one transaction: neither the lines nor a done job survive."""
    product_job_id, inference_job_id = _seed_waiting_job()
    part_id = _get_job(product_job_id).document_part_id
    assert part_id is not None

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("finalize exploded")

    monkeypatch.setattr(job_callback_service, "_mark_done_from_callback_sync", _boom)

    with pytest.raises(RuntimeError, match="finalize exploded"):
        client.post(
            CALLBACK_URL,
            headers=WEBHOOK_HEADERS,
            json=_segment_done_payload(
                product_job_id=product_job_id, inference_job_id=inference_job_id
            ),
        )

    with sync_system_session() as session:
        lines = list(session.execute(select(Line).where(Line.part_id == part_id)).scalars().all())
    assert lines == []

    job = _get_job(product_job_id)
    assert job.status != JobStatus.done
    assert job.result is None
    # The claim committed separately, so it still needs a compensating write -
    # otherwise the job hangs claimed until the stale-claim sweep notices.
    assert job.status == JobStatus.failed
    assert job.callback_claimed_at is None
    assert job.completed_at is not None


def test_callback_unknown_job_returns_404(client: TestClient):
    product_job_id = uuid.uuid4()
    inference_job_id = uuid.uuid4()
    response = client.post(
        CALLBACK_URL,
        headers=WEBHOOK_HEADERS,
        json=_segment_done_payload(
            product_job_id=product_job_id, inference_job_id=inference_job_id
        ),
    )
    assert response.status_code == 404
