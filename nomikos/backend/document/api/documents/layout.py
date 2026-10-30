"""Page layout routes: blocks, lines, layout reset, and the segmentation job."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.line_merge_service import (
    LineMergeService,
    LineMergeTextConflict,
)
from backend.document.api.line_responses import line_response
from backend.document.api.schemas import (
    BlockCreateRequest,
    BlockPatchRequest,
    BlockResponse,
    LayoutResetRequest,
    LayoutResponse,
    LineCreateRequest,
    LineMergeConflictResponse,
    LineMergeRequest,
    LinePatchRequest,
    LineResponse,
    LinesReplaceRequest,
    SegmentPartRequest,
)
from backend.document.application.document_job_enqueue import DocumentJobEnqueueService
from backend.document.application.layout_service import LayoutService
from backend.document.infrastructure.orm_models import Block
from backend.jobs.api.schemas import EnqueueJobResponse, enqueue_job_response_from_orm
from backend.ml.application.capacity_service import InferenceCapacityService
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])
_layout = LayoutService()
_line_merges = LineMergeService()
_enqueue = DocumentJobEnqueueService()
_capacity = InferenceCapacityService()


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


# The free merge sits with the other line mutations rather than under
# `segment-health`, because that router's contract is "apply the fix the report
# offered" and every one of its routes refuses an id the report did not flag.
# This one starts from what a human picked and offers nothing, so putting it
# there would mean one router with two opposite rules about where a choice comes
# from. The service behind it still lives in the annotation module, next to the
# geometry it reuses and the has-somebody-worked-on-this rule it must not
# diverge from.
@router.post(
    "/{document_id}/parts/{part_id}/lines/merge",
    response_model=list[LineResponse],
    responses={
        409: {
            "model": LineMergeConflictResponse,
            "description": "Several selected lines carry text or a pairing",
        }
    },
)
async def merge_part_lines(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    body: LineMergeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LineResponse] | JSONResponse:
    """Fold two or more of this part's lines into one, and answer with the whole list.

    The whole list, rather than the merged row, because the merge renumbers the
    rows below it: the editor already reloads this shape after a history restore
    and after every segment-health fix, so a client can replace its state rather
    than reconcile a partial answer against orders it cannot recompute.
    """
    try:
        lines = await _line_merges.merge_lines(
            db,
            current_user,
            project_id,
            document_id,
            part_id,
            body.line_ids,
            body.keep_text_from,
        )
    except LineMergeTextConflict as conflict:
        # Caught here rather than left to the global ConflictError handler, which
        # answers with a message alone. The client cannot ask "keep whose text?"
        # without the ids, so this is the one conflict that carries data.
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=jsonable_encoder(
                {
                    "error": {
                        "code": "MERGE_TEXT_CONFLICT",
                        "message": str(conflict),
                        "line_ids": conflict.line_ids,
                    }
                }
            ),
        )
    return [line_response(line) for line in lines]


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
