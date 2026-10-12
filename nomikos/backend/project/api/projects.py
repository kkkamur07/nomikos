"""Project CRUD and sharing routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, paginate_rows
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.jobs.api.schemas import JobPageResponse, job_response_from_orm
from backend.jobs.application.job_service import JobService
from backend.project.api.responses import project_response
from backend.project.api.schemas import (
    ProjectCollaboratorResponse,
    ProjectCreateRequest,
    ProjectPageResponse,
    ProjectResponse,
    ProjectUpdateRequest,
    ShareUserRequest,
)
from backend.project.application.project_service import ProjectService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects", tags=["projects"])
_service = ProjectService()
_document_repo = DocumentRepository()


def _job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)


async def _projects_with_counts(db: AsyncSession, projects: list) -> list[ProjectResponse]:
    document_counts = await _document_repo.count_documents_by_project_ids(
        db, [project.id for project in projects]
    )
    # The Shared section names whose project it is, so one lookup of the owners.
    owner_ids = {project.owner_id for project in projects if project.owner_id is not None}
    usernames: dict[UUID, str] = {}
    if owner_ids:
        rows = await db.execute(select(User.id, User.username).where(User.id.in_(owner_ids)))
        usernames = dict(rows.all())
    return [
        project_response(
            project,
            document_count=document_counts.get(project.id, 0),
            owner_username=usernames.get(project.owner_id) if project.owner_id else None,
        )
        for project in projects
    ]


@router.get("", response_model=ProjectPageResponse)
async def list_projects(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=MAX_CURSOR_LENGTH),
) -> ProjectPageResponse:
    page_cursor = decode_cursor(cursor) if cursor else None
    projects = await _service.list_projects(
        db,
        current_user,
        limit=limit + 1,
        cursor=page_cursor,
    )
    page, next_cursor = paginate_rows(
        projects,
        limit=limit,
        created_at_getter=lambda project: project.created_at,
        id_getter=lambda project: project.id,
    )
    items = await _projects_with_counts(db, page)
    return ProjectPageResponse(items=items, next_cursor=next_cursor)


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    project = await _service.create_project(
        db,
        current_user,
        name=body.name,
        slug=body.slug,
        guidelines=body.guidelines,
    )
    return project_response(project, document_count=0)


async def _project_with_count(db: AsyncSession, project) -> ProjectResponse:
    document_counts = await _document_repo.count_documents_by_project_ids(db, [project.id])
    return project_response(project, document_count=document_counts.get(project.id, 0))


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    project = await _service.get_project(db, current_user, project_id)
    return await _project_with_count(db, project)


@router.get("/{project_id}/jobs", response_model=JobPageResponse)
async def list_project_jobs(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    job_service: JobService = Depends(_job_service),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=MAX_CURSOR_LENGTH),
) -> JobPageResponse:
    await _service.get_project(db, current_user, project_id)
    page_cursor = decode_cursor(cursor) if cursor else None
    jobs = await job_service.list_project_jobs(
        project_id,
        limit=limit + 1,
        cursor=page_cursor,
    )
    page, next_cursor = paginate_rows(
        jobs,
        limit=limit,
        created_at_getter=lambda job: job.created_at,
        id_getter=lambda job: job.id,
    )
    return JobPageResponse(
        items=[job_response_from_orm(job) for job in page],
        next_cursor=next_cursor,
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    body: ProjectUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    updates = body.model_dump(exclude_unset=True)
    project = await _service.update_project(db, current_user, project_id, **updates)
    return await _project_with_count(db, project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.delete_project(db, current_user, project_id)


@router.get("/{project_id}/share", response_model=list[ProjectCollaboratorResponse])
async def list_project_collaborators(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ProjectCollaboratorResponse]:
    collaborators = await _service.list_collaborators(db, current_user, project_id)
    return [ProjectCollaboratorResponse.model_validate(user) for user in collaborators]


@router.post("/{project_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def share_project(
    project_id: UUID,
    body: ShareUserRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _service.share_project(
        db,
        current_user,
        project_id,
        username=body.username,
        email=body.email,
        identifier=body.identifier,
    )


@router.delete("/{project_id}/share/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unshare_project(
    project_id: UUID,
    user_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove a collaborator, addressed by id.

    By id rather than by username because a username is a path segment here and
    registration constrains only its length: a username containing a ``/`` splits
    into two segments, matches no route, and answers 404 however carefully the
    client percent-encodes it. An id has no character that means anything to a
    URL. ``GET /projects/{id}/share`` hands the caller the ids to use.
    """
    await _service.unshare_project(db, current_user, project_id, user_id=user_id)
