"""Project slug races and share-privacy regression tests.

Two concurrent creates with the same slug must both end in 409, never a 500:
the pre-check keeps the common case cheap and the ``IntegrityError`` handler
turns the race loser into the same conflict. Sharing with an unknown address
is ignored (204 at the route) so an owner cannot probe which emails hold
accounts. Guidelines are bounded at the API so one request cannot store
megabytes of text.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from backend.core.exceptions import ConflictError
from backend.project.api.schemas import ProjectCreateRequest, ProjectUpdateRequest
from backend.project.application.project_service import ProjectService
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def _user(username: str = "owner") -> User:
    return User(
        id=uuid.uuid4(),
        email=f"{username}@example.org",
        username=username,
        hashed_password="x",
    )


class _FakeSession:
    def __init__(self) -> None:
        self.rollbacks = 0

    async def rollback(self) -> None:
        self.rollbacks += 1


class _RacyProjectRepository:
    """First create wins; a concurrent second create hits the unique index."""

    def __init__(self, owner_id=None) -> None:
        self._owner_id = owner_id
        self._slugs: set[str] = set()

    async def get_by_slug(self, session, slug: str):
        return None

    async def create(self, session, *, name: str, slug: str, guidelines, owner_id):
        if slug in self._slugs:
            raise IntegrityError("INSERT INTO projects", {"slug": slug}, Exception("boom"))
        self._slugs.add(slug)
        return Project(id=uuid.uuid4(), name=name, slug=slug, owner_id=owner_id)

    async def get_by_id(self, session, project_id):
        project = Project(id=project_id, name="Codex", slug="codex", owner_id=self._owner_id)
        project.shared_users = []
        return project

    async def update(self, session, project, **fields):
        if "slug" in fields and fields["slug"] in self._slugs:
            raise IntegrityError("UPDATE projects", {"slug": fields["slug"]}, Exception("boom"))
        for key, value in fields.items():
            setattr(project, key, value)
        return project


async def test_concurrent_creates_with_the_same_slug_conflict() -> None:
    service = ProjectService(project_repo=_RacyProjectRepository())
    owner, session = _user(), _FakeSession()

    first = await service.create_project(session, owner, name="Codex", slug="codex")
    assert first.slug == "codex"
    with pytest.raises(ConflictError, match="slug already taken"):
        await service.create_project(session, owner, name="Codex", slug="codex")
    assert session.rollbacks == 1


async def test_concurrent_slug_update_conflicts() -> None:
    owner, session = _user(), _FakeSession()
    repo = _RacyProjectRepository(owner_id=owner.id)
    repo._slugs.add("taken")
    service = ProjectService(project_repo=repo)

    project = await service.get_project(session, owner, uuid.uuid4())
    with pytest.raises(ConflictError, match="slug already taken"):
        await service.update_project(session, owner, project.id, slug="taken")
    assert session.rollbacks == 1


def test_guidelines_are_bounded() -> None:
    ok = ProjectCreateRequest(name="n", slug="s", guidelines="g" * 10000)
    assert ok.guidelines is not None and len(ok.guidelines) == 10000

    with pytest.raises(ValidationError):
        ProjectCreateRequest(name="n", slug="s", guidelines="g" * 10001)
    with pytest.raises(ValidationError):
        ProjectUpdateRequest(guidelines="g" * 10001)
