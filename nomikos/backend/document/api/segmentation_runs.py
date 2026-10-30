"""Segmentation provenance routes: what the model proposed, and what a human left.

Read-only, both of them. The run rows are written by the segment merge inside the
job callback's transaction and never updated, so there is nothing here to PATCH
and no way to correct a record of what a model said.

The diff response carries geometry for the two classes that need it - a
correction, where the pair is the training signal, and a deletion, where the
polygon is the only surviving copy. ``unchanged`` carries ids alone: the page's
own line list already has those polygons, and repeating them would roughly double
the response for the most common case.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.application.segmentation_run_service import SegmentationRunService
from backend.document.domain.segmentation_diff import SegmentationDiff
from backend.document.infrastructure.orm_models import SegmentationRun
from backend.users.api.dependencies import get_current_user
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(
    prefix="/projects/{project_id}/documents/{document_id}/parts/{part_id}/segmentation-runs",
    tags=["segmentation-runs"],
)
_service = SegmentationRunService()


class SegmentationRunResponse(BaseModel):
    """One recorded run. The model fields are the copies taken when it happened."""

    id: UUID
    part_id: UUID
    created_by_job_id: UUID | None
    model_name: str | None
    model_provider: str | None
    model_artifact_ref: str | None
    registry_model_id: str | None
    registry_tag: str | None
    params: dict
    image_width: int | None
    image_height: int | None
    line_count: int
    block_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class UnchangedLineResponse(BaseModel):
    external_id: str
    line_id: UUID


class CorrectedLineResponse(BaseModel):
    external_id: str
    model_points: list[list[float]]
    #: Several ids when the proposed line was split into pieces.
    line_ids: list[UUID]
    human_points: list[list[list[float]]]


class DeletedLineResponse(BaseModel):
    """A proposed line with no live counterpart: removed, or absorbed by a merge."""

    external_id: str
    order: int
    model_points: list[list[float]]


class AddedLineResponse(BaseModel):
    line_id: UUID
    order: int
    points: list[list[float]]


class SegmentationDiffResponse(BaseModel):
    run: SegmentationRunResponse
    unchanged: list[UnchangedLineResponse] = Field(default_factory=list)
    corrected: list[CorrectedLineResponse] = Field(default_factory=list)
    deleted: list[DeletedLineResponse] = Field(default_factory=list)
    added: list[AddedLineResponse] = Field(default_factory=list)


def _run_response(run: SegmentationRun) -> SegmentationRunResponse:
    return SegmentationRunResponse.model_validate(run)


def _diff_response(run: SegmentationRun, diff: SegmentationDiff) -> SegmentationDiffResponse:
    return SegmentationDiffResponse(
        run=_run_response(run),
        unchanged=[
            UnchangedLineResponse(external_id=item.external_id, line_id=item.line_id)
            for item in diff.unchanged
        ],
        corrected=[
            CorrectedLineResponse(
                external_id=item.external_id,
                model_points=item.model_points,
                line_ids=list(item.line_ids),
                human_points=[list(points) for points in item.human_points],
            )
            for item in diff.corrected
        ],
        deleted=[
            DeletedLineResponse(
                external_id=item.external_id,
                order=item.order,
                model_points=item.model_points,
            )
            for item in diff.deleted
        ],
        added=[
            AddedLineResponse(line_id=item.line_id, order=item.order, points=item.points)
            for item in diff.added
        ],
    )


@router.get("", response_model=list[SegmentationRunResponse])
async def list_segmentation_runs(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[SegmentationRunResponse]:
    runs = await _service.list_runs(db, current_user, project_id, document_id, part_id)
    return [_run_response(run) for run in runs]


@router.get("/diff", response_model=SegmentationDiffResponse)
async def segmentation_run_diff(
    project_id: UUID,
    document_id: UUID,
    part_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    run_id: Annotated[
        UUID | None,
        Query(description="Which recorded run to compare against. Defaults to the latest."),
    ] = None,
) -> SegmentationDiffResponse:
    run, diff = await _service.diff(
        db, current_user, project_id, document_id, part_id, run_id=run_id
    )
    return _diff_response(run, diff)
