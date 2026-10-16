"""Project CRUD and sharing use cases."""

import logging
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, ConflictError, NotFoundError
from backend.project.domain.access import has_owner, is_member, is_owner
from backend.project.infrastructure.orm_models import Project
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User
from backend.users.infrastructure.user_repository import UserRepository

logger = logging.getLogger(__name__)


class ProjectService:
    def __init__(
        self,
        project_repo: ProjectRepository | None = None,
        user_repo: UserRepository | None = None,
    ) -> None:
        self._projects = project_repo or ProjectRepository()
        self._users = user_repo or UserRepository()

    async def list_projects(
        self,
        session: AsyncSession,
        user: User,
        *,
        limit: int = 50,
        cursor=None,
    ) -> list[Project]:
        return await self._projects.list_for_user(session, user.id, limit=limit, cursor=cursor)

    async def create_project(
        self,
        session: AsyncSession,
        user: User,
        *,
        name: str,
        slug: str,
        guidelines: str | None = None,
    ) -> Project:
        if await self._projects.get_by_slug(session, slug):
            raise ConflictError("Project slug already taken")
        try:
            return await self._projects.create(
                session,
                name=name,
                slug=slug,
                guidelines=guidelines,
                owner_id=user.id,
            )
        except IntegrityError as exc:
            # Check-then-insert lost a concurrent create with the same slug.
            # The pre-check above keeps the common case a 409 without a
            # rollback; this turns the race loser into the same 409.
            await session.rollback()
            raise ConflictError("Project slug already taken") from exc

    async def get_project(self, session: AsyncSession, user: User, project_id: UUID) -> Project:
        project = await self._load_or_404(session, project_id)
        self._require_member(project, user.id)
        return project

    async def update_project(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        **updates: str | None,
    ) -> Project:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        slug = updates.get("slug")
        if slug is not None and slug != project.slug:
            existing = await self._projects.get_by_slug(session, slug)
            if existing is not None:
                raise ConflictError("Project slug already taken")
        try:
            return await self._projects.update(session, project, **updates)
        except IntegrityError as exc:
            await session.rollback()
            raise ConflictError("Project slug already taken") from exc

    async def delete_project(self, session: AsyncSession, user: User, project_id: UUID) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        await self._projects.delete(session, project)

    async def list_collaborators(
        self, session: AsyncSession, user: User, project_id: UUID
    ) -> list[User]:
        """The users a project is shared with. Owner-only: it exposes emails."""
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        return sorted(project.shared_users, key=lambda shared: shared.username.lower())

    async def share_project(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        *,
        username: str | None = None,
        email: str | None = None,
        identifier: str | None = None,
    ) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        try:
            collaborator = await self._find_collaborator(
                session, username=username, email=email, identifier=identifier
            )
        except NotFoundError:
            # Always 204: a 404 for an unknown email would let any project
            # owner probe which addresses hold accounts. Unknown addresses are
            # ignored rather than invited so no information leaves this branch.
            logger.info("share_project_unknown_identifier project_id=%s", project_id)
            return
        if collaborator.id == project.owner_id:
            raise ConflictError("Cannot share project with the owner")
        if any(shared.id == collaborator.id for shared in project.shared_users):
            raise ConflictError("User already has access to this project")
        await self._projects.add_shared_user(session, project, collaborator)

    async def unshare_project(
        self, session: AsyncSession, user: User, project_id: UUID, *, user_id: UUID
    ) -> None:
        project = await self._load_or_404(session, project_id)
        self._require_owner(project, user.id)
        # Taken from the project's own collaborators rather than looked up
        # globally: "is not shared on this project" and "is not a user at all"
        # are the same answer to the caller, and this way it is one query.
        collaborator = next(
            (shared for shared in project.shared_users if shared.id == user_id), None
        )
        if collaborator is None:
            raise NotFoundError("User is not shared on this project")
        await self._projects.remove_shared_user(session, project, collaborator)

    async def _find_collaborator(
        self,
        session: AsyncSession,
        *,
        username: str | None = None,
        email: str | None = None,
        identifier: str | None = None,
    ) -> User:
        given = [f for f in (username, email, identifier) if f is not None]
        if len(given) != 1:
            raise ValueError("share_project needs exactly one of username, email or identifier")

        if identifier is not None:
            # A single box cannot know which kind of name it was handed: nothing
            # stops a username containing an "@", so "looks like an email" is a
            # guess, and a wrong guess is a 422 the owner cannot act on.
            #
            # Email is tried first, and that order is load-bearing rather than
            # arbitrary. Usernames are unique but unrestricted, so someone can
            # register the username "victim@example.org" while the real owner of
            # that address holds it as their email. Username-first would hand
            # that impostor every project meant for the real account; email-first
            # means a real address always resolves to the account that proved it.
            collaborator = await self._users.get_by_email(session, identifier)
            if collaborator is None:
                collaborator = await self._users.get_by_username(session, identifier)
            if collaborator is None:
                raise NotFoundError(f"No account found for {identifier}")
            return collaborator

        if email is not None:
            collaborator = await self._users.get_by_email(session, email)
            if collaborator is None:
                raise NotFoundError(f"No account is registered under {email}")
            return collaborator

        collaborator = await self._users.get_by_username(session, username or "")
        if collaborator is None:
            raise NotFoundError("User not found")
        return collaborator

    async def _load_or_404(self, session: AsyncSession, project_id: UUID) -> Project:
        project = await self._projects.get_by_id(session, project_id)
        if project is None:
            raise NotFoundError("Project not found")
        return project

    def _require_member(self, project: Project, user_id: UUID) -> None:
        if not is_member(project, user_id):
            raise AccessDeniedError("You do not have access to this project")

    def _require_owner(self, project: Project, user_id: UUID) -> None:
        if not has_owner(project):
            raise AccessDeniedError("This project has no owner; owner-only actions are unavailable")
        if not is_owner(project, user_id):
            raise AccessDeniedError("Only the project owner can perform this action")
