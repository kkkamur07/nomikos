"""Pydantic DTOs for documents and parts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from backend.document.application.part_service import (
    MAX_PART_UPLOAD_BYTES as _MAX_PART_UPLOAD_BYTES,
)
from backend.document.infrastructure.orm_models import (
    DocumentWorkflow,
    LineGeometryKind,
    LineSource,
    TranscriptionKind,
)

MAX_PAGE_TRANSCRIPTION_CHARS = 1_000_000
MAX_PAGE_TRANSCRIPTION_LINES = 10_000
MAX_REPLACE_PART_LINES = 10_000
# The largest page scan a direct-to-storage upload may declare up front. The same
# bound is re-enforced by finalize against the blob actually stored, since the direct
# PUT bypasses the API entirely (see MAX_PART_UPLOAD_BYTES in part_service).
MAX_PART_UPLOAD_BYTES = _MAX_PART_UPLOAD_BYTES
# Mirrors INFERENCE_MAX_GEOMETRY_POINTS (nomikos_inference/admission.py). Geometry above this
# bound is refused by the inference runtime, so the platform must never store it either.
MAX_LINE_GEOMETRY_POINTS = 256
MAX_LINE_TEXT_CHARS = 10_000
# Bulk id lists. A request has to be able to name every row it could legitimately act
# on - `ReorderPartsRequest` must name *all* parts or it is rejected - so these sit at
# the largest set the platform can produce rather than at some tighter operational
# number. What they rule out is the ten-million-element list, which otherwise reaches
# `.in_(...)` or drives one UPDATE per element before anything looks at it.
MAX_LINE_IDS_PER_REQUEST = MAX_REPLACE_PART_LINES
MAX_PART_IDS_PER_REQUEST = 10_000
# Unauthenticated layout reads are keyset paginated like every other list endpoint, so
# the page size caps how much work one anonymous request can ask for, not how much
# geometry a document may have. The cursor is how a client reads the rest.
DEFAULT_PUBLIC_LAYOUT_LINES = 500
MAX_PUBLIC_LAYOUT_LINES = 500
# Matches ``documents.public_share_token``'s column width. Bounding the query parameter
# here means an oversized ``t`` is a 422 before it ever reaches a comparison.
MAX_SHARE_TOKEN_LENGTH = 64

#: Optional list of line ids, bounded. Declared once so a new route cannot accept an
#: unbounded one by forgetting a per-route check - the same reason ``GeometryPoints``
#: carries its own bound.
OptionalLineIds = Annotated[list[UUID] | None, Field(max_length=MAX_LINE_IDS_PER_REQUEST)]


def _validate_point_pairs(value: list[list[float]] | None) -> list[list[float]] | None:
    if value is None:
        return value
    if any(len(point) != 2 for point in value):
        raise ValueError("each point must contain x and y")
    return value


# Shared geometry types: the point-count bound lives on the schema so no future route
# can accept an unbounded polygon by forgetting a per-route check.
GeometryPoints = Annotated[
    list[list[float]],
    Field(min_length=4, max_length=MAX_LINE_GEOMETRY_POINTS),
    AfterValidator(_validate_point_pairs),
]
OptionalGeometryPoints = Annotated[
    list[list[float]] | None,
    Field(max_length=MAX_LINE_GEOMETRY_POINTS),
    AfterValidator(_validate_point_pairs),
]
# Ordering column on a PATCH body: optional, never null (the column is NOT NULL).
PatchOrder = Annotated[int, Field(ge=0)] | SkipJsonSchema[None]


class DocumentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)


class DocumentUpdateRequest(BaseModel):
    """Partial document update.

    ``extra="forbid"`` because the handler applies ``model_dump(exclude_unset=True)``
    verbatim with ``setattr``: a key it does not recognise cannot be written, so
    dropping it silently would let a client believe it had.
    """

    model_config = {"extra": "forbid"}

    name: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    workflow: DocumentWorkflow | SkipJsonSchema[None] = None

    @field_validator("name", "workflow", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class DocumentResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    workflow: DocumentWorkflow
    part_count: int = 0
    # Owner-facing only. ``responses.document_response`` is the single place that
    # decides whether this is populated or forced to ``None`` - see its ``public`` flag.
    # ``None`` until a document is first published, and again after nothing has
    # regenerated it; never re-derive "is this shared" from anything but this field.
    public_share_token: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentPageResponse(BaseModel):
    items: list[DocumentResponse]
    next_cursor: str | None = None


class DocumentPartResponse(BaseModel):
    id: UUID
    document_id: UUID
    order: int
    image_url: str
    width: int | None
    height: int | None
    reviewed: bool
    #: Read-only, written only by the review toggle. Null on a page nobody has
    #: reviewed and on one reviewed before these columns existed - there is no
    #: backfill, because the answer is not recoverable.
    reviewed_at: datetime | None = None
    #: Suppressed on the public read: an anonymous reader of a shared document has
    #: no business learning which account signed the page off.
    reviewed_by: UUID | None = None
    published: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentPartUpdateRequest(BaseModel):
    reviewed: bool

    @field_validator("reviewed", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class PartPublishedUpdate(BaseModel):
    part_id: UUID
    published: bool


class PartsPublishedUpdateRequest(BaseModel):
    """Bulk per-part publish flag: one call for the pages a chapter goes live with."""

    parts: list[PartPublishedUpdate] = Field(min_length=1, max_length=MAX_PART_IDS_PER_REQUEST)


class PartUploadBeginRequest(BaseModel):
    """Begin a direct-to-storage part upload: name the file, get a presigned URL."""

    filename: str = Field(min_length=1, max_length=130)
    size: int = Field(gt=0, le=MAX_PART_UPLOAD_BYTES)


class PartUploadBeginResponse(BaseModel):
    part_id: UUID | None = None
    image_key: str
    upload_url: str | None = None
    token: str | None = None


class PartUploadFinalizeRequest(BaseModel):
    """Seal a direct upload after the browser has PUT the bytes to storage."""

    image_key: str = Field(min_length=1, max_length=1024)
    width: int | None = None
    height: int | None = None


class DocumentWithPartsResponse(DocumentResponse):
    parts: list[DocumentPartResponse]


class PublicDocumentWithPartsResponse(BaseModel):
    """What an anonymous reader is told a published document is.

    A separate model rather than ``DocumentWithPartsResponse`` with the token stripped
    at serialisation time: ``response_model_exclude`` keeps the value off the wire but
    not out of the schema, so the generated OpenAPI - and every client generated from
    it - went on advertising ``public_share_token`` on a body that never carries it.
    The public contract should be readable as what it is.
    """

    id: UUID
    project_id: UUID
    name: str
    workflow: DocumentWorkflow
    part_count: int = 0
    parts: list[DocumentPartResponse]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReorderPartsRequest(BaseModel):
    part_ids: list[UUID] = Field(min_length=1, max_length=MAX_PART_IDS_PER_REQUEST)


class TranscriptionLayerResponse(BaseModel):
    id: UUID
    document_id: UUID
    name: str
    kind: TranscriptionKind
    created_by_job_id: UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class LineTranscriptionResponse(BaseModel):
    id: UUID
    transcription_id: UUID
    transcription_kind: TranscriptionKind
    text: str
    confidence: float | None


class LineResponse(BaseModel):
    id: UUID
    part_id: UUID
    block_id: UUID | None
    order: int
    baseline: dict
    mask: dict | None
    kind: LineGeometryKind
    points: list[list[float]]
    source: LineSource
    source_metadata: dict[str, object] | None
    kraken_ceiling: list[list[float]] | None
    manual_geometry: bool
    line_transcriptions: list[LineTranscriptionResponse] = Field(default_factory=list)
    created_at: datetime


class LineUpsertRequest(BaseModel):
    id: UUID | None = None
    order: int = Field(ge=0)
    kind: LineGeometryKind = LineGeometryKind.polygon
    points: GeometryPoints
    block_id: UUID | None = None
    source: LineSource = LineSource.manual
    source_metadata: dict[str, object] | None = None
    kraken_ceiling: OptionalGeometryPoints = None
    baseline: dict | None = None
    mask: dict | None = None
    approved_text: str | None = Field(default=None, max_length=MAX_LINE_TEXT_CHARS)


class LinesReplaceRequest(BaseModel):
    lines: list[LineUpsertRequest] = Field(default_factory=list, max_length=MAX_REPLACE_PART_LINES)


class BlockResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    box: dict
    manual_geometry: bool
    created_at: datetime


class BlockCreateRequest(BaseModel):
    order: int = Field(ge=0)
    box: dict


class BlockPatchRequest(BaseModel):
    """Partial block update.

    ``order``/``box`` back NOT NULL columns, so an explicit ``null`` is rejected rather
    than silently ignored. Routes send ``model_dump(exclude_unset=True)``, which is what
    separates "field omitted" from "field explicitly set". An unrecognised key is
    refused for the same reason a null is: it cannot be written, so accepting the
    request would report a write that did not happen.
    """

    model_config = {"extra": "forbid"}

    order: PatchOrder = None
    box: dict | SkipJsonSchema[None] = None

    @field_validator("order", "box", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class LineCreateRequest(BaseModel):
    order: int = Field(ge=0)
    kind: LineGeometryKind = LineGeometryKind.polygon
    points: GeometryPoints
    block_id: UUID | None = None
    baseline: dict | None = None
    mask: dict | None = None


class LinePatchRequest(BaseModel):
    """Partial line update.

    ``block_id`` and ``mask`` back nullable columns, so an explicit ``null`` clears them.
    ``order``/``baseline``/``points`` back NOT NULL columns and reject an explicit null.
    An unrecognised key is refused rather than dropped, so a client is never told it
    wrote a field the handler ignored.
    """

    model_config = {"extra": "forbid"}

    order: PatchOrder = None
    block_id: UUID | None = None
    baseline: dict | SkipJsonSchema[None] = None
    mask: dict | None = None
    points: OptionalGeometryPoints = None

    @field_validator("order", "baseline", "points", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class LayoutResetRequest(BaseModel):
    line_ids: OptionalLineIds = None


class TranscribePartRequest(BaseModel):
    model_id: UUID | None = None
    line_ids: OptionalLineIds = None


class SegmentPartRequest(BaseModel):
    model_id: UUID | None = None


class LayoutResponse(BaseModel):
    blocks: list[BlockResponse]
    lines: list[LineResponse]


class PageTranscriptionImportRequest(BaseModel):
    text: str = Field(max_length=MAX_PAGE_TRANSCRIPTION_CHARS)

    @field_validator("text")
    @classmethod
    def validate_line_count(cls, value: str) -> str:
        line_count = sum(1 for line in value.splitlines() if line.strip())
        if line_count > MAX_PAGE_TRANSCRIPTION_LINES:
            raise ValueError(f"text cannot exceed {MAX_PAGE_TRANSCRIPTION_LINES} non-empty lines")
        return value


class PageTranscriptionTextLineResponse(BaseModel):
    order: int
    text: str
    paired_line_id: UUID | None


class PairingProgressResponse(BaseModel):
    paired_lines: int
    total_lines: int
    percent: int


class PagePairingResponse(BaseModel):
    text_lines: list[PageTranscriptionTextLineResponse]
    pairing_progress: PairingProgressResponse


class ExportWarningsResponse(BaseModel):
    unpaired_segments: list[int]
    unused_text_lines: list[int]


class ExportArtifactResponse(BaseModel):
    line_id: UUID
    segment_number: int
    image_filename: str
    transcription_filename: str
    transcription_text: str
    image_base64: str


class ExportResponse(BaseModel):
    exported_count: int
    artifacts: list[ExportArtifactResponse]
    warnings: ExportWarningsResponse
    steps: list[str]


class PairTextLineRequest(BaseModel):
    line_id: UUID
    text_line_order: int = Field(ge=0)


class CopyToGroundTruthRequest(BaseModel):
    line_ids: OptionalLineIds = None


class CopyToGroundTruthResponse(BaseModel):
    copied_line_ids: list[UUID]


class LineTranscriptionPatchRequest(BaseModel):
    # The column behind this is an unbounded ``Text``, and this was the only text field
    # in the module without a cap - so the one route that writes a single line's
    # transcription accepted a body larger than a whole page import.
    text: str = Field(max_length=MAX_LINE_TEXT_CHARS)


class PublicBlockResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    box: dict


class PublicLineResponse(BaseModel):
    id: UUID
    part_id: UUID
    order: int
    points: list[list[float]]
    line_transcriptions: list[LineTranscriptionResponse] = Field(default_factory=list)


class PublicLayoutResponse(BaseModel):
    blocks: list[PublicBlockResponse] = Field(default_factory=list)
    #: ``next_cursor`` paginates lines only, so blocks report their overflow as a flag.
    #: Without it a page holding exactly ``limit`` blocks is indistinguishable from a
    #: document that has more, and the rest are simply gone with nothing to notice.
    blocks_truncated: bool = False
    lines: list[PublicLineResponse] = Field(default_factory=list)
    next_cursor: str | None = None


class PublicTranscriptionLayerResponse(BaseModel):
    id: UUID
    name: str
    kind: TranscriptionKind

    model_config = {"from_attributes": True}
