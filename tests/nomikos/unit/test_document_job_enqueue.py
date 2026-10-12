"""``DocumentJobEnqueueService`` — turning a request into a claimable ``pending`` row.

The interface is two methods returning a :class:`Job`, but almost everything interesting
happens between authorization and ``session.add``:

* **model resolution.** An explicit ``model_id`` wins and contributes only that model's
  defaults. Otherwise the nearest binding — part, then document, then project — supplies
  both the model and its effective params. For segment, *no* binding is not an error: the
  resolver's ``NotFoundError`` is swallowed and the job goes out with a null ``model_id``
  for the worker's own default to fill in. For transcribe it is a 422, because the
  worker's default is one script's model. Both are pinned here.
* **parameter precedence.** ``segment`` accepts caller params and layers them *over* the
  model or binding params; ``transcribe`` accepts none at all. Reversing either would
  silently ignore what the user asked for.
* **refusals.** A part with no lines cannot be transcribed (409), a line id that is not on
  the part is absent (404), and an explicitly empty selection is invalid (422). Each maps
  to a different status, so the exception *types* are the contract.

The real :class:`DocumentAccess` is constructed over fake repositories rather than
stubbed, so "a non-member cannot enqueue work" is an outcome rather than a call
assertion. The ML catalog and the session are faked: both are recorded so the composed
job row can be inspected without Postgres.

Not covered here: the resolver's own precedence walk (``backend/ml``, exercised in
``tests/nomikos/integration/test_inference_catalog.py``), and what the worker does with
the row afterwards (``tests/nomikos/integration/test_jobs.py``).
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.exceptions import (
    AccessDeniedError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from backend.document.application.document_access import DocumentAccess
from backend.document.application.document_job_enqueue import DocumentJobEnqueueService
from backend.document.infrastructure.orm_models import Document, DocumentPart, Line
from backend.jobs.infrastructure.orm_models import JobStatus, JobType
from backend.ml.application.model_service import ResolvedModelBinding
from backend.ml.domain.execution import ExecutionRequest, ExecutionTarget
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask, ModelBinding
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User

# Capacity is an input to submission, not a collaborator of it: the route reads it
# once and hands it down as data. These tests are about model resolution and the
# composed row, so they pass the one reading that keeps every case on the cloud -
# preferred cloud, cloud available. The gating itself is exercised end to end in
# tests/nomikos/integration/test_execution_target.py, against real device rows.
CLOUD_AVAILABLE = ExecutionRequest(
    preferred=ExecutionTarget.cloud,
    available=frozenset({ExecutionTarget.cloud}),
)


class _Session:
    """Records the staged job instead of writing it; ``refresh`` is a no-op here."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits = 0
        self.refreshed: list[object] = []

    def add(self, item: object) -> None:
        self.added.append(item)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, item: object) -> None:
        self.refreshed.append(item)


class _ProjectRepository:
    def __init__(self, project: Project | None) -> None:
        self._project = project

    async def get_by_id(self, _session, project_id):
        if self._project is None or self._project.id != project_id:
            return None
        return self._project


class _DocumentRepository:
    def __init__(
        self, document: Document, part: DocumentPart, lines: list[Line] | None = None
    ) -> None:
        self._document = document
        self._part = part
        self._lines = lines or []

    async def get_by_id(self, _session, document_id):
        return self._document if self._document.id == document_id else None

    async def get_part(self, _session, part_id):
        return self._part if self._part.id == part_id else None

    async def list_part_lines(self, _session, part_id):
        return [line for line in self._lines if line.part_id == part_id]


class _InferenceModels:
    """Stands in for the ML catalog.

    ``resolve_for_part`` raises ``NotFoundError`` unless a binding was supplied, which is
    the production behaviour of the real resolver when nothing is bound at any scope.
    """

    def __init__(
        self,
        *,
        model: InferenceModel | None = None,
        resolved: ResolvedModelBinding | None = None,
    ) -> None:
        self._model = model
        self._resolved = resolved
        self.lookups: list[tuple[uuid.UUID, InferenceTask]] = []
        self.resolutions: list[InferenceTask] = []

    async def get_model_for_task(self, _session, model_id, task):
        self.lookups.append((model_id, task))
        if self._model is None or self._model.id != model_id:
            raise NotFoundError("Inference model not found")
        if self._model.task != task:
            raise ValidationError("Model task does not match binding task")
        return self._model

    async def resolve_for_part(self, _session, _user, _project_id, _document_id, _part_id, *, task):
        self.resolutions.append(task)
        if self._resolved is None:
            raise NotFoundError(f"No {task.value} model binding found")
        return self._resolved


def _user(user_id=None) -> User:
    return User(
        id=user_id or uuid.uuid4(),
        email="scribe@example.org",
        username="scribe",
        hashed_password="x",
    )


def _model(task: InferenceTask, defaults: dict | None = None) -> InferenceModel:
    return InferenceModel(
        id=uuid.uuid4(),
        name=f"{task.value}-model",
        provider="kraken",
        task=task,
        artifact_ref="ref",
        default_params=defaults if defaults is not None else {},
    )


def _binding(model: InferenceModel, effective: dict) -> ResolvedModelBinding:
    binding = ModelBinding(id=uuid.uuid4(), task=model.task, model_id=model.id)
    return ResolvedModelBinding(binding=binding, model=model, effective_params=effective)


def _bound_transcribe_model() -> ResolvedModelBinding:
    """A project transcription binding: transcribe refuses to run without a model."""
    return _binding(_model(InferenceTask.transcribe), {})


def _fixture(
    *,
    owner_id: uuid.UUID | None = None,
    line_count: int = 0,
    model: InferenceModel | None = None,
    resolved: ResolvedModelBinding | None = None,
):
    owner_id = owner_id or uuid.uuid4()
    project = Project(id=uuid.uuid4(), name="Codices", owner_id=owner_id)
    project.shared_users = []
    document = Document(id=uuid.uuid4(), project_id=project.id, name="MS 1")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="p0.webp")
    lines = [
        Line(id=uuid.uuid4(), part_id=part.id, order=index, baseline={})
        for index in range(line_count)
    ]
    documents = _DocumentRepository(document, part, lines)
    projects = _ProjectRepository(project)
    inference = _InferenceModels(model=model, resolved=resolved)
    service = DocumentJobEnqueueService(
        documents=documents,
        projects=projects,
        access=DocumentAccess(documents=documents, projects=projects),
        inference_models=inference,
    )
    return service, project, document, part, lines, inference


# --- Authorization: no job row is staged for a caller who may not ask for one ---
# Tests that the part chain is walked before any catalog lookup or write. Does not re-test
# the 404/403 split, which belongs to test_document_access_seam.


@pytest.mark.parametrize("method", ["enqueue_segment_part", "enqueue_transcribe_part"])
async def test_a_non_member_cannot_enqueue_and_nothing_is_staged(method: str) -> None:
    service, project, document, part, _lines, inference = _fixture(line_count=2)
    session = _Session()

    with pytest.raises(AccessDeniedError):
        await getattr(service, method)(
            session, _user(), project.id, document.id, part.id, execution=CLOUD_AVAILABLE
        )

    assert session.added == []
    assert session.commits == 0
    # Nor did an unauthorized caller get to probe the ML catalog through the resolver.
    assert inference.resolutions == []


# --- Transcribe refusals: three inputs, three statuses ---
# Tests the exception type for each bad request, because the type is what selects the HTTP
# status. Does not test the router's request schema.


async def test_a_part_with_no_lines_cannot_be_transcribed() -> None:
    """409, not 404 or 422: the part exists and the request is well formed, the page is
    just not segmented yet, so retrying after segmentation is the correct client move."""
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, line_count=0)
    session = _Session()

    with pytest.raises(ConflictError):
        await service.enqueue_transcribe_part(
            session,
            _user(owner_id),
            project.id,
            document.id,
            part.id,
            execution=CLOUD_AVAILABLE,
        )
    assert session.added == []
    assert session.commits == 0


async def test_a_line_id_from_another_page_is_not_found() -> None:
    owner_id = uuid.uuid4()
    service, project, document, part, lines, _inference = _fixture(owner_id=owner_id, line_count=2)
    session = _Session()

    with pytest.raises(NotFoundError, match="Line not found"):
        await service.enqueue_transcribe_part(
            session,
            _user(owner_id),
            project.id,
            document.id,
            part.id,
            line_ids=[lines[0].id, uuid.uuid4()],
            execution=CLOUD_AVAILABLE,
        )
    assert session.added == []


async def test_an_explicitly_empty_selection_is_invalid_rather_than_a_whole_page() -> None:
    """``None`` means the whole page; ``[]`` is a caller who selected nothing.

    Treating them alike would turn a mis-built request into a full-page transcription
    charge, so the empty list has to be refused instead of falling back.
    """
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, line_count=2)
    session = _Session()

    with pytest.raises(ValidationError):
        await service.enqueue_transcribe_part(
            session,
            _user(owner_id),
            project.id,
            document.id,
            part.id,
            line_ids=[],
            execution=CLOUD_AVAILABLE,
        )
    assert session.added == []


# --- The composed job row ---
# Tests the fields the worker and the callback path read back off the row. Does not test
# the worker itself.


async def test_transcribe_stages_a_pending_cloud_job_bound_to_the_caller_and_page() -> None:
    owner_id = uuid.uuid4()
    user = _user(owner_id)
    service, project, document, part, _lines, _inference = _fixture(
        owner_id=owner_id, line_count=2, resolved=_bound_transcribe_model()
    )
    session = _Session()

    job = await service.enqueue_transcribe_part(
        session, user, project.id, document.id, part.id, execution=CLOUD_AVAILABLE
    )

    assert (job.type, job.status) == (JobType.transcribe, JobStatus.pending)
    assert (job.user_id, job.document_id, job.document_part_id) == (user.id, document.id, part.id)
    assert job.payload["execution"] == "cloud"
    # Committed and refreshed, so the caller can read the server-assigned id straight back
    # into the 202 response.
    assert session.added == [job]
    assert session.commits == 1
    assert session.refreshed == [job]


async def test_a_whole_page_transcribe_carries_no_line_selection() -> None:
    """Absent, not empty: the worker distinguishes "every line" from "these lines"."""
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(
        owner_id=owner_id, line_count=2, resolved=_bound_transcribe_model()
    )

    job = await service.enqueue_transcribe_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        execution=CLOUD_AVAILABLE,
    )

    assert "line_ids" not in job.payload


async def test_a_selective_transcribe_carries_the_ids_as_strings_in_caller_order() -> None:
    owner_id = uuid.uuid4()
    service, project, document, part, lines, _inference = _fixture(
        owner_id=owner_id, line_count=3, resolved=_bound_transcribe_model()
    )
    selection = [lines[2].id, lines[0].id]

    job = await service.enqueue_transcribe_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        line_ids=selection,
        execution=CLOUD_AVAILABLE,
    )

    # JSONB cannot hold UUID objects, so the payload has to be serialisable as written.
    assert job.payload["line_ids"] == [str(line_id) for line_id in selection]


async def test_segment_stages_a_pending_cloud_job_without_needing_lines() -> None:
    """Segmentation is what *produces* lines, so the transcribe precondition must not apply."""
    owner_id = uuid.uuid4()
    user = _user(owner_id)
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, line_count=0)
    session = _Session()

    job = await service.enqueue_segment_part(
        session, user, project.id, document.id, part.id, execution=CLOUD_AVAILABLE
    )

    assert (job.type, job.status) == (JobType.segment, JobStatus.pending)
    assert (job.user_id, job.document_id, job.document_part_id) == (user.id, document.id, part.id)
    assert job.payload == {"ml_params": {}, "execution": "cloud"}
    assert session.commits == 1


# --- Model resolution: explicit id, nearest binding, or neither ---
# Tests which model and binding land on the row and where the params came from. Does not
# test the resolver's scope walk, which lives in backend/ml.


async def test_an_explicit_model_contributes_its_defaults_and_no_binding() -> None:
    owner_id = uuid.uuid4()
    model = _model(InferenceTask.transcribe, {"beam": 4})
    service, project, document, part, _lines, inference = _fixture(
        owner_id=owner_id, line_count=1, model=model
    )

    job = await service.enqueue_transcribe_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        model_id=model.id,
        execution=CLOUD_AVAILABLE,
    )

    assert job.model_id == model.id
    # A model chosen by id was not reached through a binding, so attributing one would
    # misreport where the parameters came from.
    assert job.binding_id is None
    assert job.payload["ml_params"] == {"beam": 4}
    assert inference.lookups == [(model.id, InferenceTask.transcribe)]
    # An explicit choice short-circuits resolution entirely.
    assert inference.resolutions == []


async def test_an_explicit_model_is_checked_against_the_task_it_was_asked_for() -> None:
    """A segmentation model on a transcribe job would fail deep inside the worker."""
    owner_id = uuid.uuid4()
    segmenter = _model(InferenceTask.segment)
    service, project, document, part, _lines, _inference = _fixture(
        owner_id=owner_id, line_count=1, model=segmenter
    )
    session = _Session()

    with pytest.raises(ValidationError):
        await service.enqueue_transcribe_part(
            session,
            _user(owner_id),
            project.id,
            document.id,
            part.id,
            model_id=segmenter.id,
            execution=CLOUD_AVAILABLE,
        )
    assert session.added == []


async def test_an_unknown_model_id_is_not_found() -> None:
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(
        owner_id=owner_id, line_count=1, model=_model(InferenceTask.transcribe)
    )
    session = _Session()

    with pytest.raises(NotFoundError):
        await service.enqueue_segment_part(
            session,
            _user(owner_id),
            project.id,
            document.id,
            part.id,
            model_id=uuid.uuid4(),
            execution=CLOUD_AVAILABLE,
        )
    assert session.added == []


@pytest.mark.parametrize(
    ("method", "task"),
    [
        ("enqueue_segment_part", InferenceTask.segment),
        ("enqueue_transcribe_part", InferenceTask.transcribe),
    ],
)
async def test_the_nearest_binding_supplies_the_model_and_its_effective_params(
    method: str, task: InferenceTask
) -> None:
    owner_id = uuid.uuid4()
    model = _model(task, {"ignored": True})
    resolved = _binding(model, {"threshold": 0.3})
    service, project, document, part, _lines, inference = _fixture(
        owner_id=owner_id, line_count=1, resolved=resolved
    )

    job = await getattr(service, method)(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        execution=CLOUD_AVAILABLE,
    )

    assert job.model_id == model.id
    # The binding id is kept so a later reader can see which override produced the run.
    assert job.binding_id == resolved.binding.id
    # ``effective_params`` already merged the model defaults with the binding overrides;
    # re-deriving them from ``default_params`` here would drop the overrides.
    assert job.payload["ml_params"] == {"threshold": 0.3}
    assert inference.resolutions == [task]


@pytest.mark.parametrize(
    ("method", "task"),
    [
        ("enqueue_segment_part", InferenceTask.segment),
        ("enqueue_transcribe_part", InferenceTask.transcribe),
    ],
)
async def test_a_project_scoped_binding_is_used_for_both_tasks_when_no_model_is_given(
    method: str, task: InferenceTask
) -> None:
    """The scope walk itself lives in backend/ml; this pins that a project-scoped
    row the resolver returns is honored by both enqueue paths, so a default a
    member saved on the project actually selects the model for new jobs."""
    owner_id = uuid.uuid4()
    model = _model(task, {"beam": 4})
    service, project, document, part, _lines, inference = _fixture(
        owner_id=owner_id, line_count=1, model=model
    )
    # The fixture mints the project id internally, so the project scope can only
    # be attached after it exists; the fake holds whatever binding it is given.
    binding = ModelBinding(id=uuid.uuid4(), task=task, model_id=model.id, project_id=project.id)
    inference._resolved = ResolvedModelBinding(
        binding=binding, model=model, effective_params={"beam": 4}
    )

    job = await getattr(service, method)(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        execution=CLOUD_AVAILABLE,
    )

    assert job.model_id == model.id
    assert job.binding_id == binding.id
    assert job.payload["ml_params"] == {"beam": 4}
    assert inference.resolutions == [task]


async def test_segment_with_no_binding_anywhere_still_enqueues_with_a_null_model() -> None:
    """Not an error for segmentation: one segmenter serves every script, and the resolver's
    404 says only that nobody pinned a model."""
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, line_count=1)
    session = _Session()

    job = await service.enqueue_segment_part(
        session, _user(owner_id), project.id, document.id, part.id, execution=CLOUD_AVAILABLE
    )

    assert job.model_id is None
    assert job.binding_id is None
    assert session.commits == 1


async def test_transcribe_with_no_model_and_no_binding_is_refused_and_nothing_is_staged() -> None:
    """The worker's fallback is one script's model; a Greek page must not run it silently."""
    owner_id = uuid.uuid4()
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, line_count=1)
    session = _Session()

    with pytest.raises(ValidationError) as refused:
        await service.enqueue_transcribe_part(
            session, _user(owner_id), project.id, document.id, part.id, execution=CLOUD_AVAILABLE
        )

    assert str(refused.value) == (
        "No transcription model is bound to this project. "
        "Pick a model in the menu or bind one to the project."
    )
    assert session.added == []
    assert session.commits == 0


# --- Segment parameter precedence: the caller's request wins ---
# Tests that per-request tuning layers over whatever the model or binding supplies.


async def test_request_params_override_an_explicit_models_defaults() -> None:
    owner_id = uuid.uuid4()
    model = _model(InferenceTask.segment, {"heatmap_threshold": 0.17, "model_default": True})
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, model=model)

    job = await service.enqueue_segment_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        model_id=model.id,
        ml_params={"heatmap_threshold": 0.5},
        execution=CLOUD_AVAILABLE,
    )

    assert job.payload["ml_params"] == {"heatmap_threshold": 0.5, "model_default": True}


async def test_request_params_override_a_bindings_effective_params() -> None:
    owner_id = uuid.uuid4()
    model = _model(InferenceTask.segment)
    resolved = _binding(model, {"heatmap_threshold": 0.17, "binding_default": 80})
    service, project, document, part, _lines, _inference = _fixture(
        owner_id=owner_id, resolved=resolved
    )

    job = await service.enqueue_segment_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        ml_params={"heatmap_threshold": 0.5},
        execution=CLOUD_AVAILABLE,
    )

    assert job.payload["ml_params"] == {"heatmap_threshold": 0.5, "binding_default": 80}


async def test_the_payload_params_are_a_copy_of_the_catalog_row() -> None:
    """The job payload is a per-run snapshot; mutating it must not edit the shared model."""
    owner_id = uuid.uuid4()
    model = _model(InferenceTask.segment, {"heatmap_threshold": 0.17})
    service, project, document, part, _lines, _inference = _fixture(owner_id=owner_id, model=model)

    job = await service.enqueue_segment_part(
        _Session(),
        _user(owner_id),
        project.id,
        document.id,
        part.id,
        model_id=model.id,
        execution=CLOUD_AVAILABLE,
    )
    job.payload["ml_params"]["heatmap_threshold"] = 0.1

    assert model.default_params == {"heatmap_threshold": 0.17}
