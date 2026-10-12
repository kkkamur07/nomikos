"""Public read-only routes for published documents."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.page_xml_export_service import PageXmlExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.core.api.content_disposition import attachment_disposition
from backend.core.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, paginate_rows
from backend.document.api.line_responses import line_transcription_response
from backend.document.api.public_rate_limit import throttle_public_export, throttle_public_read
from backend.document.api.responses import public_document_with_parts_response
from backend.document.api.schemas import (
    DEFAULT_PUBLIC_LAYOUT_LINES,
    MAX_PUBLIC_LAYOUT_LINES,
    MAX_SHARE_TOKEN_LENGTH,
    PublicBlockResponse,
    PublicDocumentWithPartsResponse,
    PublicLayoutResponse,
    PublicLineResponse,
    PublicTranscriptionLayerResponse,
)
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.document_export_docx import (
    DOCX_MEDIA_TYPE,
    DocumentDocxExportService,
)
from infrastructure.db import get_db

#: Every route here answers an unauthenticated caller, so the per-client read budget is
#: mounted on the router rather than route by route: a route added later is metered by
#: existing, not by whoever adds it remembering to.
router = APIRouter(
    prefix="/public",
    tags=["public"],
    dependencies=[Depends(throttle_public_read)],
    responses={429: {"description": "Rate limit exceeded"}},
)
_service = DocumentCatalog()
_transcription_pdf_service = TranscriptionPdfService()
_page_xml_export_service = PageXmlExportService()
_docx_export_service = DocumentDocxExportService()

PDF_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Transcription PDF bytes",
    }
}
XML_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/xml": {"schema": {"type": "string", "format": "binary"}}},
        "description": "PAGE XML bytes",
    }
}
DOCX_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {DOCX_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        "description": "Word transcription of one published page",
    }
}
ZIP_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Zip of the PAGE XML and the full-resolution page image it describes",
    }
}


def _public_line_response(line) -> PublicLineResponse:
    return PublicLineResponse(
        id=line.id,
        part_id=line.part_id,
        order=line.order,
        points=line.points,
        line_transcriptions=[
            line_transcription_response(transcription) for transcription in line.transcriptions
        ],
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}",
    response_model=PublicDocumentWithPartsResponse,
)
async def get_published_document(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> PublicDocumentWithPartsResponse:
    # No dimension backfill here, deliberately. It is a *write* - up to 25 blob
    # downloads and a `session.commit()` - and this route answers anyone with the URL.
    # Parts uploaded since migration 004 carry their dimensions already; the legacy ones
    # are filled by the member read of the same document (`GET .../documents/{id}`),
    # which is the path that has a caller to attribute the work to. Until then the
    # response carries `width: null`, which its schema has always allowed.
    document = await _service.get_document_public(db, project_id, document_id, token=t)
    return public_document_with_parts_response(document)


@router.get(
    "/projects/{project_id}/documents/{document_id}/layout",
    response_model=PublicLayoutResponse,
)
async def get_published_layout(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_PUBLIC_LAYOUT_LINES, ge=1, le=MAX_PUBLIC_LAYOUT_LINES),
    cursor: str | None = Query(default=None, max_length=MAX_CURSOR_LENGTH),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> PublicLayoutResponse:
    page_cursor = decode_cursor(cursor) if cursor else None
    # The catalog probes one row past ``limit`` on both axes: for lines the extra row
    # becomes the cursor below, for blocks it becomes ``blocks_truncated``.
    layout = await _service.list_document_layout_public(
        db,
        project_id,
        document_id,
        limit=limit,
        cursor=page_cursor,
        token=t,
    )
    page, next_cursor = paginate_rows(
        layout.lines,
        limit=limit,
        created_at_getter=lambda line: line.created_at,
        id_getter=lambda line: line.id,
    )
    return PublicLayoutResponse(
        blocks=[
            PublicBlockResponse(
                id=block.id,
                part_id=block.part_id,
                order=block.order,
                box=block.box,
            )
            for block in layout.blocks
        ],
        blocks_truncated=layout.blocks_truncated,
        lines=[_public_line_response(line) for line in page],
        next_cursor=next_cursor,
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/transcriptions",
    response_model=list[PublicTranscriptionLayerResponse],
)
async def list_published_transcriptions(
    project_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> list[PublicTranscriptionLayerResponse]:
    transcriptions = await _service.list_transcriptions_public(db, project_id, document_id, token=t)
    return [PublicTranscriptionLayerResponse.model_validate(row) for row in transcriptions]


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
    # Charged on top of the router's read budget: this is a full fpdf2 render over
    # every line on the page, not a row read.
    dependencies=[Depends(throttle_public_export)],
)
async def get_published_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> Response:
    pdf_bytes = await _transcription_pdf_service.generate_part_pdf_public(
        db,
        project_id,
        document_id,
        part_id,
        token=t,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="transcription.pdf"'},
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/page-xml",
    response_class=Response,
    responses=XML_RESPONSE,
    dependencies=[Depends(throttle_public_export)],
)
async def get_published_page_xml(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> Response:
    xml_bytes = await _page_xml_export_service.export_part_public(
        db,
        project_id,
        document_id,
        part_id,
        token=t,
    )
    return Response(
        content=xml_bytes,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="page.xml"'},
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/page-xml-bundle",
    response_class=Response,
    responses=ZIP_RESPONSE,
    dependencies=[Depends(throttle_public_export)],
)
async def get_published_page_xml_bundle(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> Response:
    """The PAGE XML zipped next to the full-resolution page image it describes."""
    bundle = await _page_xml_export_service.export_part_bundle_public(
        db,
        project_id,
        document_id,
        part_id,
        token=t,
    )
    zip_bytes = await asyncio.to_thread(bundle.to_zip)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": attachment_disposition(bundle.zip_filename)},
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}/parts/{part_id}/docx",
    response_class=Response,
    responses=DOCX_RESPONSE,
    dependencies=[Depends(throttle_public_export)],
)
async def get_published_docx(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: AsyncSession = Depends(get_db),
    t: str | None = Query(default=None, max_length=MAX_SHARE_TOKEN_LENGTH),
) -> Response:
    """One published page's transcription as a Word file."""
    export = await _docx_export_service.export_part_docx_public(
        db,
        project_id,
        document_id,
        part_id,
        token=t,
    )
    return Response(
        content=export.content,
        media_type=DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )
