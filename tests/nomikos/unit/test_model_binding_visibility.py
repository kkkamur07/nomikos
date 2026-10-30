"""Authz predicate tests: a binding is visible only inside its own project.

``_binding_visible_to_project`` gates binding reads, updates and deletes, so
every scope branch - project, document, part, missing rows, foreign projects -
is pinned here against in-memory doubles.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.ml.application.model_service import InferenceModelService
from backend.ml.infrastructure.model_repository import MlRepository
from backend.ml.infrastructure.orm_models import InferenceTask, ModelBinding
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def _user() -> User:
    return User(
        id=uuid.uuid4(),
        email="owner@example.org",
        username="owner",
        hashed_password="x",
    )


def _project(owner: User) -> Project:
    project = Project(id=uuid.uuid4(), name="Codex", slug="codex", owner_id=owner.id)
    project.shared_users = []
    return project


def _binding(**kwargs) -> ModelBinding:
    values: dict = {
        "task": InferenceTask.transcribe,
        "model_id": uuid.uuid4(),
        "project_id": None,
        "document_id": None,
        "document_part_id": None,
        "overrides": {},
    }
    values.update(kwargs)
    return ModelBinding(id=uuid.uuid4(), **values)


class _FakeProjects:
    def __init__(self, projects: list[Project]) -> None:
        self._projects = {p.id: p for p in projects}

    async def get_by_id(self, session, project_id):
        return self._projects.get(project_id)


class _FakeDocuments:
    def __init__(self, docs: list[Document], parts: list[DocumentPart]) -> None:
        self._docs = {d.id: d for d in docs}
        self._parts = {p.id: p for p in parts}

    async def get_by_id(self, session, document_id):
        return self._docs.get(document_id)

    async def get_part(self, session, part_id):
        return self._parts.get(part_id)


class _FakeInference:
    def __init__(self, bindings: list[ModelBinding]) -> None:
        self._bindings = {b.id: b for b in bindings}

    async def get_binding(self, session, binding_id):
        return self._bindings.get(binding_id)


def _service(
    projects: list[Project],
    docs: list[Document] | None = None,
    parts: list[DocumentPart] | None = None,
    bindings: list[ModelBinding] | None = None,
) -> InferenceModelService:
    return InferenceModelService(
        inference=_FakeInference(bindings or []),
        projects=_FakeProjects(projects),
        documents=_FakeDocuments(docs or [], parts or []),
    )


async def test_project_scoped_binding_is_visible() -> None:
    owner = _user()
    project = _project(owner)
    binding = _binding(project_id=project.id)
    service = _service([project], bindings=[binding])

    assert await service._binding_visible_to_project(None, owner, project.id, binding.id) is binding


async def test_non_member_cannot_see_any_binding() -> None:
    owner, stranger = _user(), _user()
    project = _project(owner)
    binding = _binding(project_id=project.id)
    service = _service([project], bindings=[binding])

    with pytest.raises(AccessDeniedError):
        await service._binding_visible_to_project(None, stranger, project.id, binding.id)


async def test_missing_binding_is_not_found() -> None:
    owner = _user()
    project = _project(owner)
    service = _service([project])

    with pytest.raises(NotFoundError):
        await service._binding_visible_to_project(None, owner, project.id, uuid.uuid4())


async def test_document_scoped_binding_is_visible_in_its_project() -> None:
    owner = _user()
    project = _project(owner)
    doc = Document(id=uuid.uuid4(), project_id=project.id, name="d")
    binding = _binding(document_id=doc.id)
    service = _service([project], docs=[doc], bindings=[binding])

    assert await service._binding_visible_to_project(None, owner, project.id, binding.id) is binding


async def test_document_scoped_binding_is_hidden_from_other_projects() -> None:
    owner = _user()
    project, other = _project(owner), _project(owner)
    doc = Document(id=uuid.uuid4(), project_id=project.id, name="d")
    binding = _binding(document_id=doc.id)
    service = _service([project, other], docs=[doc], bindings=[binding])

    with pytest.raises(NotFoundError):
        await service._binding_visible_to_project(None, owner, other.id, binding.id)


async def test_binding_for_a_deleted_document_is_not_found() -> None:
    owner = _user()
    project = _project(owner)
    binding = _binding(document_id=uuid.uuid4())
    service = _service([project], bindings=[binding])

    with pytest.raises(NotFoundError):
        await service._binding_visible_to_project(None, owner, project.id, binding.id)


async def test_part_scoped_binding_resolves_through_its_document() -> None:
    owner = _user()
    project = _project(owner)
    doc = Document(id=uuid.uuid4(), project_id=project.id, name="d")
    part = DocumentPart(id=uuid.uuid4(), document_id=doc.id, order=0, image_key="k")
    binding = _binding(document_part_id=part.id)
    service = _service([project], docs=[doc], parts=[part], bindings=[binding])

    assert await service._binding_visible_to_project(None, owner, project.id, binding.id) is binding


async def test_part_scoped_binding_is_hidden_from_other_projects() -> None:
    owner = _user()
    project, other = _project(owner), _project(owner)
    doc = Document(id=uuid.uuid4(), project_id=project.id, name="d")
    part = DocumentPart(id=uuid.uuid4(), document_id=doc.id, order=0, image_key="k")
    binding = _binding(document_part_id=part.id)
    service = _service([project, other], docs=[doc], parts=[part], bindings=[binding])

    with pytest.raises(NotFoundError):
        await service._binding_visible_to_project(None, owner, other.id, binding.id)


async def test_binding_for_a_deleted_part_is_not_found() -> None:
    owner = _user()
    project = _project(owner)
    binding = _binding(document_part_id=uuid.uuid4())
    service = _service([project], bindings=[binding])

    with pytest.raises(NotFoundError):
        await service._binding_visible_to_project(None, owner, project.id, binding.id)


class _RacySession:
    def __init__(self) -> None:
        self.rollbacks = 0
        self.added = []

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        raise IntegrityError("INSERT INTO model_bindings", {}, Exception("boom"))

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def refresh(self, row) -> None:
        raise AssertionError("must not refresh a rolled-back row")


async def test_concurrent_binding_create_is_a_conflict_not_a_500() -> None:
    repo = MlRepository()

    with pytest.raises(ConflictError, match="already exists"):
        await repo.create_binding(
            _RacySession(),
            task=InferenceTask.transcribe,
            model_id=uuid.uuid4(),
            project_id=uuid.uuid4(),
        )
