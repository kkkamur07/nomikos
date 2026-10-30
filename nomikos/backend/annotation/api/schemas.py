from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from backend.annotation.infrastructure.orm_models import SnapshotState


class AnnotationHistorySnapshotResponse(BaseModel):
    id: UUID
    part_id: UUID
    state: SnapshotState
    line_count: int
    paired_line_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AnnotationHistorySnapshotSummaryResponse(BaseModel):
    id: UUID
    part_id: UUID
    line_count: int
    paired_line_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentSuspectResponse(BaseModel):
    """A segment that looks like noise. Flagged for a human, never auto-deleted."""

    line_id: UUID
    reasons: list[str]


class SegmentSplitResponse(BaseModel):
    """A segment that swallowed two columns, with the gutter cuts already found."""

    line_id: UUID
    cuts: list[float]
    piece_count: int


class SegmentMergeResponse(BaseModel):
    """A fragment and the line it broke off. The primary keeps its id."""

    primary_id: UUID
    fragment_id: UUID


class SegmentOverlapResponse(BaseModel):
    """Two masks sharing enough area that one of them is wrong."""

    upper_id: UUID
    lower_id: UUID
    ratio: float
    cut: float
    #: Fraction of each outline a trim would remove, so the cost is on screen
    #: before anyone agrees to it.
    upper_loss: float
    lower_loss: float
    #: One line drawn twice rather than two that bleed into each other. No trim
    #: is offered for these: cutting between the baselines halves a duplicate.
    duplicate: bool


class SegmentHealthResponse(BaseModel):
    part_id: UUID
    page_width: float
    page_height: float
    #: False when the part stores no dimensions and the page was measured from
    #: the segments instead. The findings still hold, but the column bands are
    #: derived from where the ink is rather than where the page ends.
    measured_page: bool
    line_count: int
    considered_count: int
    finding_count: int
    suspects: list[SegmentSuspectResponse]
    spanning: list[SegmentSplitResponse]
    fragments: list[SegmentMergeResponse]
    overlaps: list[SegmentOverlapResponse]


class SegmentSplitRequest(BaseModel):
    line_id: UUID


class SegmentMergeRequest(BaseModel):
    primary_id: UUID
    fragment_id: UUID


class SegmentTrimRequest(BaseModel):
    upper_id: UUID
    lower_id: UUID


class SegmentDeleteRequest(BaseModel):
    line_id: UUID
