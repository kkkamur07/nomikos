"""Document routes: list, read, create, rename, share-token rotation, delete."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, paginate_rows
from backend.document.api.responses import (
    document_response,
    document_with_parts_response,
)
from backend.document.api.schemas import (
    DocumentCreateRequest,
    DocumentPageResponse,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentWithPartsResponse,
)
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_catalog = DocumentCatalog()
_parts = DocumentPartService()
_document_repo = DocumentRepository()


@router.get("", response_model=DocumentPageResponse)
async def list_documents(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_archived: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=MAX_CURSOR_LENGTH),
) -> DocumentPageResponse:
    page_cursor = decode_cursor(cursor) if cursor else None
    documents = await _catalog.list_documents(
        db,
        current_user,
        project_id,
        include_archived=include_archived,
        limit=limit + 1,
        cursor=page_cursor,
    )
    page, next_cursor = paginate_rows(
        documents,
        limit=limit,
        created_at_getter=lambda document: document.created_at,
        id_getter=lambda document: document.id,
    )
    part_counts = await _document_repo.count_parts_by_document_ids(
        db, [document.id for document in page]
    )
    owns = await _catalog.owns_project(db, current_user, project_id)
    return DocumentPageResponse(
        items=[
            document_response(
                document,
                part_count=part_counts.get(document.id, 0),
                share_token=document.public_share_token if owns else None,
            )
            for document in page
        ],
        next_cursor=next_cursor,
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    project_id: UUID,
    body: DocumentCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    document = await _catalog.create_document(db, current_user, project_id, name=body.name)
    return document_response(document, part_count=0)


@router.get("/{document_id}", response_model=DocumentWithPartsResponse)
async def get_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentWithPartsResponse:
    document = await _catalog.get_document(db, current_user, project_id, document_id)
    await _parts.backfill_part_dimensions(db, document.parts)
    owns = await _catalog.owns_project(db, current_user, project_id)
    return document_with_parts_response(
        document, share_token=document.public_share_token if owns else None
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    project_id: UUID,
    document_id: UUID,
    body: DocumentUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    updates = body.model_dump(exclude_unset=True)
    document = await _catalog.update_document(db, current_user, project_id, document_id, **updates)
    part_counts = await _document_repo.count_parts_by_document_ids(db, [document.id])
    owns = await _catalog.owns_project(db, current_user, project_id)
    return document_response(
        document,
        part_count=part_counts.get(document.id, 0),
        share_token=document.public_share_token if owns else None,
    )


@router.post("/{document_id}/share-token/rotate", response_model=DocumentResponse)
async def rotate_document_share_token(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentResponse:
    """Mint a fresh share token, owner-only. Every link built from the old one 404s
    from here on - see ``DocumentCatalog.rotate_share_token`` - regardless of who it
    was shared with or how long ago.
    """
    document = await _catalog.rotate_share_token(db, current_user, project_id, document_id)
    part_counts = await _document_repo.count_parts_by_document_ids(db, [document.id])
    # Ownership is the gate on rotation itself, so reaching this line has already
    # established it; there is nothing left to ask.
    return document_response(
        document,
        part_count=part_counts.get(document.id, 0),
        share_token=document.public_share_token,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _catalog.delete_document(db, current_user, project_id, document_id)
