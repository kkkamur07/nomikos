"""Transcription routes: layers, page text pairing, ground truth, and the transcription job."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.api.schemas import (
    CopyToGroundTruthRequest,
    CopyToGroundTruthResponse,
    LineTranscriptionPatchRequest,
    LineTranscriptionResponse,
    PagePairingResponse,
    PageTranscriptionImportRequest,
    PageTranscriptionTextLineResponse,
    PairingProgressResponse,
    PairTextLineRequest,
    TranscribePartRequest,
    TranscriptionLayerResponse,
)
from backend.document.application.document_catalog import DocumentCatalog
from backend.document.application.document_job_enqueue import DocumentJobEnqueueService
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.orm_models import PageTranscriptionLine, TranscriptionKind
from backend.jobs.api.schemas import EnqueueJobResponse, enqueue_job_response_from_orm
from backend.ml.application.capacity_service import InferenceCapacityService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_catalog = DocumentCatalog()
_transcriptions = TranscriptionService()
_enqueue = DocumentJobEnqueueService()
_capacity = InferenceCapacityService()


def _pairing_response(
    text_lines: list[PageTranscriptionLine], progress: dict[str, int]
) -> PagePairingResponse:
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


@router.get("/{document_id}/transcriptions", response_model=list[TranscriptionLayerResponse])
async def list_transcriptions(
    project_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TranscriptionLayerResponse]:
    transcriptions = await _catalog.list_transcriptions(db, current_user, project_id, document_id)
    return [TranscriptionLayerResponse.model_validate(t) for t in transcriptions]


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
    )
    return LineTranscriptionResponse(
        id=line_transcription.id,
        transcription_id=line_transcription.transcription_id,
        transcription_kind=TranscriptionKind.ground_truth,
        text=line_transcription.text,
        confidence=line_transcription.confidence,
    )
