"""Export routes: line artifacts, per-page PDF and PAGE XML, and whole-document exports."""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.export_service import AnnotationExportService, ExportResult
from backend.annotation.application.page_xml_export_service import PageXmlExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.core.api.content_disposition import attachment_disposition
from backend.document.api.openapi_responses import (
    DOCUMENT_ZIP_RESPONSE,
    DOCX_RESPONSE,
    PDF_RESPONSE,
    TEXT_RESPONSE,
    XML_RESPONSE,
    ZIP_RESPONSE,
)
from backend.document.api.schemas import (
    ExportArtifactResponse,
    ExportResponse,
    ExportWarningsResponse,
)
from backend.document.application.document_export_docx import (
    DOCX_MEDIA_TYPE,
    DocumentDocxExportService,
)
from backend.document.application.document_export_page_xml import DocumentPageXmlExportService
from backend.document.application.document_export_pdf import DocumentTranscriptionPdfService
from backend.document.application.document_export_text import DocumentTextExportService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_export_service = AnnotationExportService()
_page_xml_export_service = PageXmlExportService()
_transcription_pdf_service = TranscriptionPdfService()
_document_page_xml_export_service = DocumentPageXmlExportService()
_document_pdf_export_service = DocumentTranscriptionPdfService()
_document_text_export_service = DocumentTextExportService()
_document_docx_export_service = DocumentDocxExportService()
REVIEWED_ONLY_DESCRIPTION = (
    "Cover only pages marked reviewed. 404 when that leaves nothing to export."
)


def _export_response(result: ExportResult) -> ExportResponse:
    return ExportResponse(
        exported_count=result.exported_count,
        artifacts=[
            ExportArtifactResponse(
                line_id=artifact.line_id,
                segment_number=artifact.segment_number,
                image_filename=artifact.image_filename,
                transcription_filename=artifact.transcription_filename,
                transcription_text=artifact.transcription_text,
                image_base64=artifact.image_base64,
            )
            for artifact in result.artifacts
        ],
        warnings=ExportWarningsResponse(
            unpaired_segments=result.warnings.unpaired_segments,
            unused_text_lines=result.warnings.unused_text_lines,
        ),
        steps=result.steps,
    )


@router.post("/{document_id}/parts/{part_id}/export", response_model=ExportResponse)
async def export_approved_line_artifacts(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ExportResponse:
    result = await _export_service.export_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return _export_response(result)


async def _member_transcription_pdf_response(
    db: AsyncSession,
    current_user: User,
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
) -> Response:
    pdf_bytes = await _transcription_pdf_service.generate_part_pdf(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="transcription.pdf"'},
    )


@router.get(
    "/{document_id}/parts/{part_id}/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
)
async def get_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    return await _member_transcription_pdf_response(
        db, current_user, project_id, document_id, part_id
    )


@router.post(
    "/{document_id}/parts/{part_id}/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
)
async def generate_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    return await _member_transcription_pdf_response(
        db, current_user, project_id, document_id, part_id
    )


@router.get(
    "/{document_id}/parts/{part_id}/docx",
    response_class=Response,
    responses=DOCX_RESPONSE,
)
async def export_part_docx(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """One page's transcription as a Word file, for the page editor."""
    export = await _document_docx_export_service.export_part_docx(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=export.content,
        media_type=DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )


@router.get(
    "/{document_id}/parts/{part_id}/page-xml",
    response_class=Response,
    responses=XML_RESPONSE,
)
async def export_part_page_xml(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    xml_bytes = await _page_xml_export_service.export_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    return Response(
        content=xml_bytes,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="page.xml"'},
    )


@router.get(
    "/{document_id}/parts/{part_id}/page-xml-bundle",
    response_class=Response,
    responses=ZIP_RESPONSE,
)
async def export_part_page_xml_bundle(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """The PAGE XML zipped next to the full-resolution page image it describes."""
    bundle = await _page_xml_export_service.export_part_bundle(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
    )
    # Zipping a full manuscript scan is a memcpy plus a deflate of the XML; small,
    # but off the event loop like every other byte-shuffling export here.
    zip_bytes = await asyncio.to_thread(bundle.to_zip)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": attachment_disposition(bundle.zip_filename)},
    )


# --- Whole-document exports ---
#
# The same three artifacts the per-part routes above serve, for every page at once.
# Getting an eighteen-page chapter out of the per-part routes means opening eighteen
# pages and pressing the same button eighteen times, which is not a workflow anybody
# completes without losing a page. Membership is checked exactly as it is above, and
# ``reviewed_only`` is the one filter: it narrows the export to pages a human has signed
# off, which is what a chapter is usually released as.


@router.get(
    "/{document_id}/export/page-xml",
    response_class=StreamingResponse,
    responses=DOCUMENT_ZIP_RESPONSE,
)
async def export_document_page_xml(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    reviewed_only: bool = Query(default=False, description=REVIEWED_ONLY_DESCRIPTION),
) -> StreamingResponse:
    """Every page as PAGE XML next to the full-resolution image it describes."""
    export = await _document_page_xml_export_service.export_document(
        db,
        current_user,
        project_id,
        document_id,
        reviewed_only=reviewed_only,
    )
    # Streamed rather than returned whole: the archive holds a full manuscript scan per
    # page, and the generator is written so that only one of them is in memory at a time.
    return StreamingResponse(
        export.chunks,
        media_type="application/zip",
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )


@router.get(
    "/{document_id}/export/transcription-pdf",
    response_class=Response,
    responses=PDF_RESPONSE,
)
async def export_document_transcription_pdf(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    reviewed_only: bool = Query(default=False, description=REVIEWED_ONLY_DESCRIPTION),
) -> Response:
    """The whole document's transcription as one PDF, pages in reading order."""
    export = await _document_pdf_export_service.generate_document_pdf(
        db,
        current_user,
        project_id,
        document_id,
        reviewed_only=reviewed_only,
    )
    return Response(
        content=export.content,
        media_type="application/pdf",
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )


@router.get(
    "/{document_id}/export/text",
    response_class=Response,
    responses=TEXT_RESPONSE,
)
async def export_document_text(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    reviewed_only: bool = Query(default=False, description=REVIEWED_ONLY_DESCRIPTION),
) -> Response:
    """The whole document's transcription as plain text, one ``[p.N]`` block per page."""
    export = await _document_text_export_service.export_document_text(
        db,
        current_user,
        project_id,
        document_id,
        reviewed_only=reviewed_only,
    )
    return Response(
        content=export.text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )


@router.get(
    "/{document_id}/export/docx",
    response_class=Response,
    responses=DOCX_RESPONSE,
)
async def export_document_docx(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    reviewed_only: bool = Query(default=False, description=REVIEWED_ONLY_DESCRIPTION),
) -> Response:
    """The whole document's transcription as a Word file, one heading per page."""
    export = await _document_docx_export_service.export_document_docx(
        db,
        current_user,
        project_id,
        document_id,
        reviewed_only=reviewed_only,
    )
    return Response(
        content=export.content,
        media_type=DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": attachment_disposition(export.filename)},
    )
