"""Unit tests for the run instruction a claimed page carries.

A fake session stands in for Postgres: ``get`` answers from a dict keyed by ORM
class and id, and the part's lines come from a patched ``load_lines``. Does not
test the claim transaction around the build, see test_job_claim_service.py.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from backend.document.application.transcribe_merge_service import TranscribeJobHandlerError
from backend.document.infrastructure.orm_models import DocumentPart
from backend.jobs.application import inference_dispatcher
from backend.jobs.application.inference_dispatcher import build_page_run
from backend.jobs.infrastructure.orm_models import JobType
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask


class _FakeSession:
    def __init__(self, rows: dict[tuple[type, uuid.UUID], object]) -> None:
        self._rows = rows

    def get(self, model: type, row_id: uuid.UUID) -> object | None:
        return self._rows.get((model, row_id))


def _line(index: int) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), points=[[index, 0], [index, 10]])


@pytest.fixture
def page(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """One part with three lines, reachable through the patched session."""
    document_id = uuid.uuid4()
    part = SimpleNamespace(id=uuid.uuid4(), document_id=document_id, image_key="pages/p1.jpg")
    lines = [_line(index) for index in range(3)]
    rows: dict[tuple[type, uuid.UUID], object] = {(DocumentPart, part.id): part}
    sessions: list[_FakeSession] = []

    @contextmanager
    def _factory():
        session = _FakeSession(rows)
        sessions.append(session)
        yield session

    monkeypatch.setattr(inference_dispatcher, "sync_system_session", _factory)
    monkeypatch.setattr(
        inference_dispatcher.TranscribeMergeService,
        "load_lines",
        staticmethod(lambda _session, part_id: list(lines) if part_id == part.id else []),
    )
    return SimpleNamespace(
        document_id=document_id, part=part, lines=lines, rows=rows, sessions=sessions
    )


def _job(page: SimpleNamespace, job_type: JobType, **overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "id": uuid.uuid4(),
        "type": job_type,
        "document_id": page.document_id,
        "document_part_id": page.part.id,
        "model_id": None,
        "payload": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _model(page: SimpleNamespace, task: InferenceTask, artifact_ref: str) -> uuid.UUID:
    model_id = uuid.uuid4()
    page.rows[(InferenceModel, model_id)] = SimpleNamespace(task=task, artifact_ref=artifact_ref)
    return model_id


# --- Transcribe ---
# Tests the registry model, the line list and the params an agent is handed.


def test_a_transcribe_job_without_a_model_falls_back_to_the_default(page: SimpleNamespace):
    run = build_page_run(_job(page, JobType.transcribe))

    assert (run.request.registry_model_id, run.request.registry_tag) == (
        "syriac-calamari-v2",
        "stable",
    )
    assert run.request.task.value == "transcribe"
    assert run.image_key == "pages/p1.jpg"
    assert len(page.sessions) == 1


def test_every_line_is_sent_in_reading_order(page: SimpleNamespace):
    run = build_page_run(_job(page, JobType.transcribe))

    assert run.request.params["lines"] == [
        {"line_id": str(line.id), "line_index": index, "points": line.points}
        for index, line in enumerate(page.lines)
    ]


def test_line_ids_narrow_the_page_and_renumber_it(page: SimpleNamespace):
    chosen = page.lines[2]
    job = _job(page, JobType.transcribe, payload={"line_ids": [str(chosen.id)]})

    run = build_page_run(job)

    assert run.request.params["lines"] == [
        {"line_id": str(chosen.id), "line_index": 0, "points": chosen.points}
    ]


def test_line_ids_that_match_nothing_are_refused(page: SimpleNamespace):
    job = _job(page, JobType.transcribe, payload={"line_ids": [str(uuid.uuid4())]})

    with pytest.raises(TranscribeJobHandlerError, match="No matching lines"):
        build_page_run(job)


def test_ml_params_are_merged_under_the_line_list(page: SimpleNamespace):
    job = _job(
        page,
        JobType.transcribe,
        payload={"ml_params": {"beam_width": 4, "lines": "ignored"}},
    )

    params = build_page_run(job).request.params

    assert params["beam_width"] == 4
    assert isinstance(params["lines"], list)
    assert len(params["lines"]) == 3


def test_a_selected_model_names_its_registry_entry(page: SimpleNamespace):
    model_id = _model(page, InferenceTask.transcribe, "registry://greek-trocr?tag=v3")

    run = build_page_run(_job(page, JobType.transcribe, model_id=model_id))

    assert (run.request.registry_model_id, run.request.registry_tag) == ("greek-trocr", "v3")


@pytest.mark.parametrize("artifact_ref", ["registry://greek-trocr", "registry://greek-trocr?tag="])
def test_a_registry_ref_without_a_tag_means_stable(page: SimpleNamespace, artifact_ref: str):
    model_id = _model(page, InferenceTask.transcribe, artifact_ref)

    run = build_page_run(_job(page, JobType.transcribe, model_id=model_id))

    assert run.request.registry_tag == "stable"


@pytest.mark.parametrize(
    "artifact_ref", ["hf://greek-trocr", "registry://", "registry://greek-trocr/extra"]
)
def test_a_non_registry_ref_is_refused(page: SimpleNamespace, artifact_ref: str):
    model_id = _model(page, InferenceTask.transcribe, artifact_ref)

    with pytest.raises(ValueError, match="artifact_ref must be registry://"):
        build_page_run(_job(page, JobType.transcribe, model_id=model_id))


def test_a_missing_model_is_refused(page: SimpleNamespace):
    with pytest.raises(ValueError, match="Selected inference model not found"):
        build_page_run(_job(page, JobType.transcribe, model_id=uuid.uuid4()))


def test_a_model_for_another_task_is_refused(page: SimpleNamespace):
    model_id = _model(page, InferenceTask.segment, "registry://blla-segment")

    with pytest.raises(ValueError, match="does not support transcribe"):
        build_page_run(_job(page, JobType.transcribe, model_id=model_id))


@pytest.mark.parametrize("missing", ["document_id", "document_part_id"])
def test_a_transcribe_job_without_its_target_is_refused(page: SimpleNamespace, missing: str):
    job = _job(page, JobType.transcribe, **{missing: None})

    with pytest.raises(TranscribeJobHandlerError, match="missing its target document part"):
        build_page_run(job)


def test_a_part_of_another_document_is_refused(page: SimpleNamespace):
    job = _job(page, JobType.transcribe, document_id=uuid.uuid4())

    with pytest.raises(TranscribeJobHandlerError, match="Document part not found"):
        build_page_run(job)


def test_a_deleted_part_is_refused(page: SimpleNamespace):
    job = _job(page, JobType.transcribe, document_part_id=uuid.uuid4())

    with pytest.raises(TranscribeJobHandlerError, match="Document part not found"):
        build_page_run(job)


# --- Segment and unknown types ---


def test_a_segment_job_carries_its_params_and_image_key(page: SimpleNamespace):
    job = _job(page, JobType.segment, payload={"ml_params": {"text_direction": "rl"}})

    run = build_page_run(job)

    assert run.request.task.value == "segment"
    assert (run.request.registry_model_id, run.request.registry_tag) == ("blla-segment", "stable")
    assert run.request.params == {"text_direction": "rl"}
    assert run.image_key == "pages/p1.jpg"


def test_a_segment_job_without_a_part_is_refused(page: SimpleNamespace):
    with pytest.raises(ValueError, match="Segment job is missing"):
        build_page_run(_job(page, JobType.segment, document_part_id=None))


def test_a_segment_job_on_a_deleted_part_is_refused(page: SimpleNamespace):
    with pytest.raises(ValueError, match="Document part not found"):
        build_page_run(_job(page, JobType.segment, document_part_id=uuid.uuid4()))


def test_a_job_type_with_no_run_instruction_is_refused(page: SimpleNamespace):
    other = next(
        job_type for job_type in JobType if job_type not in (JobType.segment, JobType.transcribe)
    )

    with pytest.raises(ValueError, match="No inference run instruction"):
        build_page_run(_job(page, other))
    assert page.sessions == []
