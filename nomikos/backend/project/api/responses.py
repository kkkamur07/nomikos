"""Shared response builders for project API routers."""

from backend.project.api.schemas import ProjectResponse
from backend.project.infrastructure.orm_models import Project


def project_response(
    project: Project, *, document_count: int = 0, owner_username: str | None = None
) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        name=project.name,
        slug=project.slug,
        guidelines=project.guidelines,
        owner_id=project.owner_id,
        owner_username=owner_username,
        document_count=document_count,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
