"""Unit tests for sharing a project by email or username."""

import uuid

import pytest

from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError
from backend.project.api.schemas import ShareUserRequest
from backend.project.application.project_service import ProjectService
from backend.project.infrastructure.orm_models import Project
from backend.users.infrastructure.orm_models import User


def _user(username: str, email: str) -> User:
    return User(id=uuid.uuid4(), email=email, username=username, hashed_password="x")


class _StubUserRepository:
    def __init__(self, users: list[User]) -> None:
        self._users = users

    async def get_by_email(self, session, email):
        return next((u for u in self._users if u.email.lower() == email.lower()), None)

    async def get_by_username(self, session, username):
        return next((u for u in self._users if u.username == username), None)


class _StubProjectRepository:
    def __init__(self, project: Project) -> None:
        self._project = project
        self.added: list[User] = []
        self.removed: list[User] = []

    async def get_by_id(self, session, project_id):
        return self._project if project_id == self._project.id else None

    async def add_shared_user(self, session, project, user):
        project.shared_users.append(user)
        self.added.append(user)

    async def remove_shared_user(self, session, project, user):
        if user not in project.shared_users:
            return False
        project.shared_users.remove(user)
        self.removed.append(user)
        return True


@pytest.fixture
def owner() -> User:
    return _user("owner", "owner@example.org")


@pytest.fixture
def friend() -> User:
    return _user("friend", "Friend@Example.org")


@pytest.fixture
def project(owner: User) -> Project:
    project = Project(id=uuid.uuid4(), name="Codex", slug="codex", owner_id=owner.id)
    project.shared_users = []
    return project


def _service(project: Project, users: list[User]) -> tuple[ProjectService, _StubProjectRepository]:
    projects = _StubProjectRepository(project)
    return ProjectService(project_repo=projects, user_repo=_StubUserRepository(users)), projects


# --- Sharing by email ---
# Tests the email path resolves an account. Does not test persistence.


async def test_share_by_email_adds_the_collaborator(owner, friend, project) -> None:
    service, projects = _service(project, [owner, friend])

    await service.share_project(None, owner, project.id, email="friend@example.org")

    assert projects.added == [friend]


async def test_share_by_email_ignores_case(owner, friend, project) -> None:
    service, projects = _service(project, [owner, friend])

    await service.share_project(None, owner, project.id, email="FRIEND@EXAMPLE.ORG")

    assert projects.added == [friend]


async def test_share_by_unknown_email_names_the_address(owner, project) -> None:
    service, _ = _service(project, [owner])

    with pytest.raises(NotFoundError, match="nobody@example.org"):
        await service.share_project(None, owner, project.id, email="nobody@example.org")


async def test_share_with_own_email_is_a_conflict(owner, project) -> None:
    service, _ = _service(project, [owner])

    with pytest.raises(ConflictError):
        await service.share_project(None, owner, project.id, email=owner.email)


async def test_share_twice_by_email_is_a_conflict(owner, friend, project) -> None:
    service, _ = _service(project, [owner, friend])
    await service.share_project(None, owner, project.id, email=friend.email)

    with pytest.raises(ConflictError):
        await service.share_project(None, owner, project.id, username=friend.username)


async def test_share_by_username_still_works(owner, friend, project) -> None:
    service, projects = _service(project, [owner, friend])

    await service.share_project(None, owner, project.id, username="friend")

    assert projects.added == [friend]


# --- The single box: identifier resolution ---
# Tests email-or-username resolved server-side. Does not test the HTTP layer.


async def test_identifier_resolves_an_email(owner, friend, project) -> None:
    service, projects = _service(project, [owner, friend])

    await service.share_project(None, owner, project.id, identifier="FRIEND@example.org")

    assert projects.added == [friend]


async def test_identifier_resolves_a_username(owner, friend, project) -> None:
    service, projects = _service(project, [owner, friend])

    await service.share_project(None, owner, project.id, identifier="friend")

    assert projects.added == [friend]


async def test_identifier_resolves_a_username_containing_an_at_sign(owner, project) -> None:
    """Registration constrains only a username's length, so `greek@corpus` is a
    real account name. Classifying it as an email loses it."""
    odd = _user("greek@corpus", "curator@example.org")
    service, projects = _service(project, [owner, odd])

    await service.share_project(None, owner, project.id, identifier="greek@corpus")

    assert projects.added == [odd]


async def test_identifier_prefers_the_email_owner_over_a_username_squatter(owner, project) -> None:
    """A username is unique but unrestricted, so someone can register the
    username `victim@example.org` while the real owner of that address holds it
    as their email. The address must resolve to the account that proved it."""
    squatter = _user("victim@example.org", "squatter@example.org")
    victim = _user("victim", "victim@example.org")
    service, projects = _service(project, [owner, squatter, victim])

    await service.share_project(None, owner, project.id, identifier="victim@example.org")

    assert projects.added == [victim]


async def test_unknown_identifier_names_what_was_typed(owner, project) -> None:
    service, _ = _service(project, [owner])

    with pytest.raises(NotFoundError, match="ghost"):
        await service.share_project(None, owner, project.id, identifier="ghost")


async def test_share_needs_exactly_one_identifier(owner, friend, project) -> None:
    service, _ = _service(project, [owner, friend])

    with pytest.raises(ValueError):
        await service.share_project(None, owner, project.id)
    with pytest.raises(ValueError):
        await service.share_project(
            None, owner, project.id, username="friend", email="friend@example.org"
        )
    with pytest.raises(ValueError):
        await service.share_project(None, owner, project.id, username="friend", identifier="friend")


# --- Removing a collaborator ---
# Tests removal addressed by id. Does not test the URL, which is what made it necessary.


async def test_unshare_removes_the_collaborator_by_id(owner, friend, project) -> None:
    project.shared_users = [friend]
    service, projects = _service(project, [owner, friend])

    await service.unshare_project(None, owner, project.id, user_id=friend.id)

    assert projects.removed == [friend]
    assert project.shared_users == []


async def test_unshare_a_user_who_is_not_a_collaborator_is_not_found(
    owner, friend, project
) -> None:
    service, _ = _service(project, [owner, friend])

    with pytest.raises(NotFoundError):
        await service.unshare_project(None, owner, project.id, user_id=friend.id)


async def test_unshare_is_owner_only(owner, friend, project) -> None:
    other = _user("other", "other@example.org")
    project.shared_users = [friend, other]
    service, _ = _service(project, [owner, friend, other])

    with pytest.raises(AccessDeniedError):
        await service.unshare_project(None, friend, project.id, user_id=other.id)


# --- Collaborator list ---
# Tests who may read the list and its order. Does not test HTTP serialisation.


async def test_collaborators_are_listed_for_the_owner_sorted_by_username(
    owner, friend, project
) -> None:
    zed = _user("zed", "zed@example.org")
    project.shared_users = [zed, friend]
    service, _ = _service(project, [owner, friend, zed])

    listed = await service.list_collaborators(None, owner, project.id)

    assert [u.username for u in listed] == ["friend", "zed"]


async def test_collaborators_are_hidden_from_non_owners(owner, friend, project) -> None:
    project.shared_users = [friend]
    service, _ = _service(project, [owner, friend])

    with pytest.raises(AccessDeniedError):
        await service.list_collaborators(None, friend, project.id)


# --- Request schema ---
# Tests the body accepts one identifier. Does not test the route.


def test_share_request_accepts_email_and_normalises_it() -> None:
    body = ShareUserRequest(email=" Friend@Example.org ")
    assert body.email == "friend@example.org"
    assert body.username is None


def test_share_request_accepts_username() -> None:
    assert ShareUserRequest(username="friend").username == "friend"


def test_share_request_accepts_a_bare_identifier() -> None:
    body = ShareUserRequest(identifier="  greek@corpus  ")
    assert body.identifier == "greek@corpus"
    assert body.email is None and body.username is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"username": "friend", "email": "friend@example.org"},
        {"identifier": "friend", "email": "friend@example.org"},
        {"identifier": "   "},
        {"email": "not-an-email"},
    ],
)
def test_share_request_rejects_ambiguous_or_malformed_bodies(payload) -> None:
    with pytest.raises(ValueError):
        ShareUserRequest(**payload)
