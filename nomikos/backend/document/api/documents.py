"""Document and DocumentPart routes under projects."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.export_service import AnnotationExportService
from backend.annotation.application.page_xml_export_service import PageXmlExportService
from backend.annotation.application.transcription_pdf_service import TranscriptionPdfService
from backend.core.api.content_disposition import attachment_disposition
from backend.core.api.pagination import MAX_CURSOR_LENGTH, decode_cursor, paginate_rows
from backend.core.exceptions import ValidationError
from backend.document.api.line_responses import line_response
from backend.document.api.responses import (
    document_response,
    document_with_parts_response,
    part_response,
)
from backend.document.api.schemas import (
    BlockCreateRequest,
    BlockPatchRequest,
    BlockResponse,
    CopyToGroundTruthRequest,
    CopyToGroundTruthResponse,
    DocumentCreateRequest,
    DocumentPageResponse,
    DocumentPartResponse,
    DocumentPartUpdateRequest,
    DocumentResponse,
    DocumentUpdateRequest,
    DocumentWithPartsResponse,
    ExportArtifactResponse,
    ExportResponse,
    ExportWarningsResponse,
    LayoutResetRequest,
    LayoutResponse,
    LineCreateRequest,
    LinePatchRequest,
    LineResponse,
    LinesReplaceRequest,
    LineTranscriptionPatchRequest,
    LineTranscriptionResponse,
    PagePairingResponse,
    PageTranscriptionImportRequest,
    PageTranscriptionTextLineResponse,
    PairingProgressResponse,
    PairTextLineRequest,
    PartsPublishedUpdateRequest,
    PartUploadBeginRequest,
    PartUploadBeginResponse,
    PartUploadFinalizeRequest,
    ReorderPartsRequest,
    SegmentPartRequest,
    TranscribePartRequest,
    TranscriptionLayerResponse,
)
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.document_export_docx import (
    DOCX_MEDIA_TYPE,
    DocumentDocxExportService,
)
from backend.document.application.document_export_page_xml import DocumentPageXmlExportService
from backend.document.application.document_export_pdf import DocumentTranscriptionPdfService
from backend.document.application.document_export_text import DocumentTextExportService
from backend.document.application.document_job_enqueue import DocumentJobEnqueueService
from backend.document.application.layout_service import LayoutService
from backend.document.application.part_service import DocumentPartService
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Block, TranscriptionKind
from backend.jobs.api.schemas import EnqueueJobResponse, enqueue_job_response_from_orm
from backend.ml.application.capacity_service import InferenceCapacityService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_catalog = DocumentCatalog()
_parts = DocumentPartService()
_layout = LayoutService()
_transcriptions = TranscriptionService()
_enqueue = DocumentJobEnqueueService()
_capacity = InferenceCapacityService()
_document_repo = DocumentRepository()
_export_service = AnnotationExportService()
_page_xml_export_service = PageXmlExportService()
_transcription_pdf_service = TranscriptionPdfService()
_document_page_xml_export_service = DocumentPageXmlExportService()
_document_pdf_export_service = DocumentTranscriptionPdfService()
_document_text_export_service = DocumentTextExportService()
_document_docx_export_service = DocumentDocxExportService()
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
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
ZIP_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Zip of the PAGE XML and the full-resolution page image it describes",
    }
}
DOCUMENT_ZIP_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
        "description": "Zip of one PAGE XML and one full-resolution image per page",
    }
}
TEXT_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"text/plain": {"schema": {"type": "string"}}},
        "description": "Plain-text transcription, one page after another",
    }
}
DOCX_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {DOCX_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        "description": "Word transcription, one heading per page, script fonts and direction set",
    }
}
REVIEWED_ONLY_DESCRIPTION = (
    "Cover only pages marked reviewed. 404 when that leaves nothing to export."
)


def _block_response(block: Block) -> BlockResponse:
    return BlockResponse(
        id=block.id,
        part_id=block.part_id,
        order=block.order,
        box=block.box,
        manual_geometry=block.manual_geometry,
        label=block.label,
        created_at=block.created_at,
    )


def _pairing_response(text_lines: list, progress: dict[str, int]) -> PagePairingResponse:
    return PagePairingResponse(
        text_lines=[
            PageTranscriptionTextLineResponse(
                order=line.order,
                text=line.text,
                paired_line_id=line.paired_line_id,
            )
            for line in text_lines
        ],
        pairing_progress=PairingProgressResponse(**progress),
    )


def _export_response(result) -> ExportResponse:
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


@router.get("/{document_id}/transcriptions", response_model=list[TranscriptionLayerResponse])
async def list_transcriptions(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TranscriptionLayerResponse]:
    transcriptions = await _catalog.list_transcriptions(db, current_user, project_id, document_id)
    return [TranscriptionLayerResponse.model_validate(t) for t in transcriptions]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _catalog.delete_document(db, current_user, project_id, document_id)


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


@router.get("/{document_id}/parts/{part_id}/layout", response_model=LayoutResponse)
async def list_part_layout(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LayoutResponse:
    blocks, lines = await _layout.list_part_layout(
        db, current_user, project_id, document_id, part_id
    )
    return LayoutResponse(
        blocks=[_block_response(block) for block in blocks],
        lines=[line_response(line) for line in lines],
    )


@router.post(
    "/{document_id}/parts/{part_id}/blocks",
    response_model=BlockResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: BlockCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockResponse:
    block = await _layout.create_part_block(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        order=body.order,
        box=body.box,
        label=body.label or None,
        assign_contained=body.assign_contained,
    )
    return _block_response(block)


@router.patch("/{document_id}/parts/{part_id}/blocks/{block_id}", response_model=BlockResponse)
async def patch_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    block_id: UUID,
    body: BlockPatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockResponse:
    block = await _layout.patch_part_block(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        block_id,
        **body.model_dump(exclude_unset=True),
    )
    return _block_response(block)


@router.delete(
    "/{document_id}/parts/{part_id}/blocks/{block_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_part_block(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    block_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _layout.delete_part_block(db, current_user, project_id, document_id, part_id, block_id)


@router.get("/{document_id}/parts/{part_id}/lines", response_model=list[LineResponse])
async def list_part_lines(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _layout.list_part_lines(db, current_user, project_id, document_id, part_id)
    return [line_response(line) for line in lines]


@router.post(
    "/{document_id}/parts/{part_id}/lines",
    response_model=LineResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LineCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineResponse:
    line = await _layout.create_part_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        order=body.order,
        kind=body.kind,
        points=body.points,
        block_id=body.block_id,
        baseline=body.baseline,
        mask=body.mask,
    )
    return line_response(line)


@router.patch("/{document_id}/parts/{part_id}/lines/{line_id}", response_model=LineResponse)
async def patch_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    line_id: UUID,
    body: LinePatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineResponse:
    line = await _layout.patch_part_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_id,
        **body.model_dump(exclude_unset=True),
    )
    return line_response(line)


@router.delete(
    "/{document_id}/parts/{part_id}/lines/{line_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_part_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    line_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _layout.delete_part_line(db, current_user, project_id, document_id, part_id, line_id)


@router.put("/{document_id}/parts/{part_id}/lines", response_model=list[LineResponse])
async def replace_part_lines(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LinesReplaceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse]:
    lines = await _layout.replace_part_lines(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        lines=[line.model_dump(exclude_unset=True) for line in body.lines],
    )
    return [line_response(line) for line in lines]


@router.post("/{document_id}/parts/{part_id}/layout/reset", response_model=LayoutResponse)
async def reset_part_layout(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LayoutResetRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LayoutResponse:
    blocks, lines = await _layout.reset_part_layout(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_ids=body.line_ids,
    )
    return LayoutResponse(
        blocks=[_block_response(block) for block in blocks],
        lines=[line_response(line) for line in lines],
    )


@router.put("/{document_id}/parts/{part_id}/page-transcription", response_model=PagePairingResponse)
async def import_page_transcription(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: PageTranscriptionImportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _transcriptions.import_page_transcription(
        db, current_user, project_id, document_id, part_id, text=body.text
    )
    return _pairing_response(text_lines, progress)


@router.get("/{document_id}/parts/{part_id}/pairing", response_model=PagePairingResponse)
async def get_page_pairing(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _transcriptions.get_page_pairing(
        db, current_user, project_id, document_id, part_id
    )
    return _pairing_response(text_lines, progress)


@router.post("/{document_id}/parts/{part_id}/pairings", response_model=PagePairingResponse)
async def pair_page_text_line(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: PairTextLineRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PagePairingResponse:
    text_lines, progress = await _transcriptions.pair_page_text_line(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        line_id=body.line_id,
        text_line_order=body.text_line_order,
    )
    return _pairing_response(text_lines, progress)


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


@router.post(
    "/{document_id}/parts/{part_id}/segment",
    response_model=EnqueueJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def segment_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: SegmentPartRequest | None = None,
) -> EnqueueJobResponse:
    body = body or SegmentPartRequest()
    job = await _enqueue.enqueue_segment_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        execution=await _capacity.execution_request(db, current_user),
        model_id=body.model_id,
        ml_params=body.model_dump(exclude={"model_id"}),
    )
    return enqueue_job_response_from_orm(job)


@router.post(
    "/{document_id}/parts/{part_id}/transcribe",
    response_model=EnqueueJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def transcribe_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    body: TranscribePartRequest | None = None,
) -> EnqueueJobResponse:
    body = body or TranscribePartRequest()
    job = await _enqueue.enqueue_transcribe_part(
        db,
        current_user,
        project_id,
        document_id,
        part_id,
        execution=await _capacity.execution_request(db, current_user),
        model_id=body.model_id,
        line_ids=body.line_ids,
    )
    return enqueue_job_response_from_orm(job)


@router.post(
    "/{document_id}/transcriptions/{transcription_id}/copy-to-ground-truth",
    response_model=CopyToGroundTruthResponse,
)
async def copy_to_ground_truth(
    project_id: UUID,
    document_id: UUID,
    transcription_id: UUID,
    body: CopyToGroundTruthRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CopyToGroundTruthResponse:
    copied_line_ids = await _transcriptions.copy_to_ground_truth(
        db,
        current_user,
        project_id,
        document_id,
        transcription_id,
        line_ids=body.line_ids,
    )
    return CopyToGroundTruthResponse(copied_line_ids=copied_line_ids)


@router.patch(
    "/{document_id}/transcriptions/{transcription_id}/lines/{line_id}",
    response_model=LineTranscriptionResponse,
)
async def patch_ground_truth_line_text(
    project_id: UUID,
    document_id: UUID,
    transcription_id: UUID,
    line_id: UUID,
    body: LineTranscriptionPatchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LineTranscriptionResponse:
    line_transcription = await _transcriptions.patch_ground_truth_line_text(
        db,
        current_user,
        project_id,
        document_id,
        transcription_id,
        line_id,
        text=body.text,
        previous_text=body.previous_text,
    )
    return LineTranscriptionResponse(
        id=line_transcription.id,
        transcription_id=line_transcription.transcription_id,
        transcription_kind=TranscriptionKind.ground_truth,
        text=line_transcription.text,
        confidence=line_transcription.confidence,
    )


@router.delete("/{document_id}/parts/{part_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_part(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _parts.delete_part(db, current_user, project_id, document_id, part_id)
