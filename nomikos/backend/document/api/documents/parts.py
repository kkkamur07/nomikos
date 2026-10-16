"""DocumentPart routes: upload, direct-upload begin and finalize, reorder, publish, review, delete."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ValidationError
from backend.document.api.responses import (
    part_response,
)
from backend.document.api.schemas import (
    DocumentPartResponse,
    DocumentPartUpdateRequest,
    PartsPublishedUpdateRequest,
    PartUploadBeginRequest,
    PartUploadBeginResponse,
    PartUploadFinalizeRequest,
    ReorderPartsRequest,
)
from backend.document.application.part_service import DocumentPartService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_parts = DocumentPartService()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@router.post(
    "/{document_id}/parts",
    response_model=DocumentPartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_part(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
) -> DocumentPartResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ValidationError("Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError("File exceeds the 100 MB upload limit")
    # Validation happens inside the single bounded decode the service performs to encode
    # the stored WebP; decoding here as well doubled the peak memory of every upload.
    part = await _parts.upload_part(
        db,
        current_user,
        project_id,
        document_id,
        data=data,
        filename=file.filename,
    )
    return part_response(part)


@router.post(
    "/{document_id}/parts/upload",
    response_model=PartUploadBeginResponse,
    status_code=status.HTTP_201_CREATED,
)
async def begin_part_upload(
    project_id: UUID,
    document_id: UUID,
    body: PartUploadBeginRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PartUploadBeginResponse:
    """Name the file and get a presigned storage URL for the browser to PUT to.

    Vercel Functions cap a request body at 4.5 MB, so a manuscript page scan cannot
    be POSTed through the platform directly. This minted URL lets the browser stream
    the bytes to Supabase Storage instead; the caller then posts
    ``/{part_id}/finalize`` to persist the key and dimensions.

    On a backend that cannot presign (``STORAGE_BACKEND=local``) the URL and token are
    ``null`` and the caller falls back to the multipart :func:`upload_part`.
    """
    part, image_key, upload_url, token = await _parts.begin_upload(
        db,
        current_user,
        project_id,
        document_id,
        filename=body.filename,
    )
    return PartUploadBeginResponse(
        part_id=part.id if part is not None else None,
        image_key=image_key,
        upload_url=upload_url,
        token=token,
    )


@router.post(
    "/{document_id}/parts/{part_id}/finalize",
    response_model=DocumentPartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def finalize_part_upload(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: PartUploadFinalizeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentPartResponse:
    """Seal a direct upload after the browser has PUT the bytes to storage."""
    part = await _parts.finalize_upload(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        image_key=body.image_key,
        width=body.width,
        height=body.height,
    )
    return part_response(part)


@router.patch("/{document_id}/parts/reorder", response_model=list[DocumentPartResponse])
async def reorder_parts(
    project_id: UUID,
    document_id: UUID,
    body: ReorderPartsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentPartResponse]:
    parts = await _parts.reorder_parts(db, current_user, project_id, document_id, body.part_ids)
    return [part_response(p) for p in parts]


@router.patch("/{document_id}/parts/published", response_model=list[DocumentPartResponse])
async def update_parts_published(
    project_id: UUID,
    document_id: UUID,
    body: PartsPublishedUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DocumentPartResponse]:
    """Owner-only: which pages of a published document are actually reachable publicly.

    One request for the whole batch a UI toggles at once, rather than one PATCH per
    page - see ``DocumentPartService.update_parts_published``.
    """
    parts = await _parts.update_parts_published(
        db,
        current_user,
        project_id,
        document_id,
        updates=[(item.part_id, item.published) for item in body.parts],
    )
    return [part_response(p) for p in parts]


@router.patch("/{document_id}/parts/{part_id}", response_model=DocumentPartResponse)
async def update_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: DocumentPartUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentPartResponse:
    part = await _parts.update_part_review_status(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        reviewed=body.reviewed,
    )
    return part_response(part)


@router.delete("/{document_id}/parts/{part_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _parts.delete_part(db, current_user, project_id, document_id, part_id)
