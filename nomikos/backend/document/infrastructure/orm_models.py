from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infrastructure.db import Base

if TYPE_CHECKING:
    from backend.ml.infrastructure.orm_models import ModelBinding
    from backend.project.infrastructure.orm_models import Project


class DocumentWorkflow(StrEnum):
    draft = "draft"
    published = "published"
    archived = "archived"


class TranscriptionKind(StrEnum):
    ground_truth = "ground_truth"
    model = "model"


class LineGeometryKind(StrEnum):
    polygon = "polygon"
    rectangle = "rectangle"


class LineSource(StrEnum):
    manual = "manual"
    kraken = "kraken"
    model = "model"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(512))
    workflow: Mapped[DocumentWorkflow] = mapped_column(
        Enum(DocumentWorkflow, name="document_workflow"),
        default=DocumentWorkflow.draft,
    )
    # Minted on first publish, never at creation - a draft has nothing to link to yet.
    # Nullable and unique rather than a default-empty string, so "no link exists" and
    # "this is the live link" can never be confused with each other or with a collision.
    public_share_token: Mapped[str | None] = mapped_column(
        String(64), unique=True, index=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    project: Mapped[Project] = relationship("Project", back_populates="documents")
    parts: Mapped[list[DocumentPart]] = relationship(
        "DocumentPart", back_populates="document", cascade="all, delete-orphan"
    )
    transcriptions: Mapped[list[Transcription]] = relationship(
        "Transcription", back_populates="document", cascade="all, delete-orphan"
    )
    model_bindings: Mapped[list[ModelBinding]] = relationship(
        "ModelBinding", back_populates="document"
    )


class DocumentPart(Base):
    __tablename__ = "document_parts"
    __table_args__ = (
        UniqueConstraint("document_id", "order", name="uq_document_parts_document_order"),
        Index("ix_document_parts_document_order", "document_id", "order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    order: Mapped[int] = mapped_column(Integer, default=0)
    image_key: Mapped[str] = mapped_column(String(1024))
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    # Who signed the page off and when. Both nullable and both written together with
    # ``reviewed``: a page reviewed before this column existed keeps nulls, and there
    # is no backfill because the answer is not recoverable. Unmarking clears them
    # rather than keeping the last reviewer, so "reviewed is false" can never be read
    # alongside a name that seems to contradict it.
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Defaults true (and every existing row is backfilled true by the migration) so
    # publishing a document does not silently change behaviour for pages that were
    # already live before per-page holdback existed.
    published: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="parts")
    blocks: Mapped[list[Block]] = relationship(
        "Block", back_populates="part", cascade="all, delete-orphan"
    )
    lines: Mapped[list[Line]] = relationship(
        "Line", back_populates="part", cascade="all, delete-orphan"
    )
    page_transcription_lines: Mapped[list[PageTranscriptionLine]] = relationship(
        "PageTranscriptionLine", back_populates="part", cascade="all, delete-orphan"
    )
    segmentation_runs: Mapped[list[SegmentationRun]] = relationship(
        "SegmentationRun", back_populates="part", cascade="all, delete-orphan"
    )
    model_bindings: Mapped[list[ModelBinding]] = relationship(
        "ModelBinding", back_populates="document_part"
    )


class MediaDeletionIntent(Base):
    """Durable outbox record for eventually deleting an object-store key."""

    __tablename__ = "media_deletion_intents"
    __table_args__ = (
        Index(
            "ix_media_deletion_intents_pending",
            "created_at",
            postgresql_where="completed_at IS NULL",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_key: Mapped[str] = mapped_column(String(1024), unique=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Block(Base):
    __tablename__ = "blocks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), index=True
    )
    box: Mapped[dict] = mapped_column(JSONB)
    manual_geometry: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    part: Mapped[DocumentPart] = relationship("DocumentPart", back_populates="blocks")
    lines: Mapped[list[Line]] = relationship("Line", back_populates="block")


class SegmentationRun(Base):
    """What one segmenter run proposed for one page, exactly as it proposed it.

    The live ``lines`` rows are the *current* state of a page: every human edit
    overwrites geometry in place, flips ``source`` to manual and, for a deletion,
    removes the row entirely. So by the time a page is worth training on, what the
    model actually said about it is gone. This table is the other half of the pair
    the platform already has on the transcription side, where each recogniser run
    gets its own immutable ``Transcription`` layer and the human works in a separate
    ground-truth layer: one row here per segment job per part, written in the same
    transaction that inserts the live lines and never updated afterwards.

    Model identity is **denormalised at write time**, not left as a join. The only
    durable link from a merged page back to its model is ``jobs.model_id``, and jobs
    are user-deletable (``clear_project_job_history``) with the FK set to SET NULL,
    so a join would evaporate on the first history clear and take the training
    signal with it. The columns below are copies, deliberately, and they record what
    was true when the run happened rather than what the registry says today.

    Retention: keep every run. A row is a few tens of kilobytes for a normal page,
    the same order as the ``lines`` rows it produced, and it is the measurement
    itself - "what did this model version propose before anyone touched it" is not a
    question a later run can answer. Pruning is a part-level decision if it is ever
    needed; the rows cascade with the part they describe.
    """

    __tablename__ = "segmentation_runs"
    __table_args__ = (Index("ix_segmentation_runs_part_created", "part_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), index=True
    )
    # SET NULL, like every other job reference: the run outlives the job that made it.
    created_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ``registry://<registry_model_id>?tag=<tag>`` as it stood at submission, plus the
    # two parts split out so a query can filter on them without parsing a URL.
    model_artifact_ref: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    registry_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    registry_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: The effective parameters the job ran with (model defaults, binding overrides
    #: and the caller's ``ml_params`` already merged), e.g. ``heatmap_threshold``.
    params: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    #: The page size the coordinates below are in. Copied from the part because the
    #: part can be replaced by a re-upload at a different resolution, after which the
    #: stored polygons would be silently mis-scaled against the new image.
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    block_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    #: The canonical output verbatim: one entry per proposed line, carrying
    #: ``external_id``, ``order``, ``baseline``, ``points``, ``mask``,
    #: ``kraken_ceiling``, ``block_external_id`` and the adapter's own metadata.
    lines: Mapped[list[dict]] = mapped_column(JSONB, default=list, server_default="[]")
    blocks: Mapped[list[dict]] = mapped_column(JSONB, default=list, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    part: Mapped[DocumentPart] = relationship("DocumentPart", back_populates="segmentation_runs")


class Line(Base):
    __tablename__ = "lines"
    __table_args__ = (
        Index("ix_lines_part_order", "part_id", "order", "created_at"),
        Index("ix_lines_segmentation_run_external", "segmentation_run_id", "run_external_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), index=True
    )
    block_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("blocks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    baseline: Mapped[dict] = mapped_column(JSONB)
    mask: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    kind: Mapped[LineGeometryKind] = mapped_column(
        Enum(LineGeometryKind, name="line_geometry_kind"),
        default=LineGeometryKind.polygon,
        server_default=LineGeometryKind.polygon.value,
    )
    points: Mapped[list[list[float]]] = mapped_column(JSONB, default=list, server_default="[]")
    source: Mapped[LineSource] = mapped_column(
        Enum(LineSource, name="line_source"),
        default=LineSource.manual,
        server_default=LineSource.manual.value,
    )
    source_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Where this line came from, as two real columns rather than keys in
    # ``source_metadata``. Three reasons. They are what the provenance diff joins and
    # groups on, so they want a plain composite index instead of a JSONB expression
    # index. ``source_metadata`` is client-writable through ``LineUpsertRequest``, so
    # a bulk save could rewrite the origin of every line on the page; these two are in
    # no request schema and in no PATCH field list, which is what makes "the human
    # edited it" and "the model proposed it" impossible to confuse. And a real column
    # survives every in-place mutation for free: nothing has to remember to copy it.
    segmentation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("segmentation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    #: The ``external_id`` this line had in that run's output. Kept even once the line
    #: has been redrawn by hand: it is the join key back to what the model proposed.
    run_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kraken_ceiling: Mapped[list[list[float]] | None] = mapped_column(JSONB, nullable=True)
    manual_geometry: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # ``onupdate`` rather than a write in each service: the mutation paths are spread
    # over three modules and a fourth would be added without anybody remembering this.
    # The price is that it moves on any UPDATE, including the pure renumbering segment
    # health does to close a gap, so read it as "last written", not "last redrawn".
    # Nullable with no server default, so rows that predate this simply have none
    # rather than all claiming to have been edited at migration time.
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )
    #: The human who last edited this line's geometry through the editor. Set only by
    #: the paths that have a user in hand and are that person's own act; a server-side
    #: fix or a re-segment leaves it alone.
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    part: Mapped[DocumentPart] = relationship("DocumentPart", back_populates="lines")
    block: Mapped[Block | None] = relationship("Block", back_populates="lines")
    transcriptions: Mapped[list[LineTranscription]] = relationship(
        "LineTranscription", back_populates="line", cascade="all, delete-orphan"
    )


class Transcription(Base):
    __tablename__ = "transcriptions"
    __table_args__ = (
        Index(
            "uq_transcriptions_one_ground_truth",
            "document_id",
            unique=True,
            postgresql_where="kind = 'ground_truth'",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[TranscriptionKind] = mapped_column(
        Enum(TranscriptionKind, name="transcription_kind")
    )
    created_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    # Denormalised for the same reason as on ``SegmentationRun``: the only route from
    # a model layer to the model that wrote it ran document -> transcription -> job ->
    # inference_model, and both of those FKs are SET NULL. Clearing job history left a
    # layer called "Model transcription 3f2a9c1b" and no way to say whose output it
    # was, which is precisely the thing a retraining run needs to know. Null on the
    # ground-truth layer and on anything created before this shipped.
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_artifact_ref: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    registry_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    registry_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_params: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="transcriptions")
    line_transcriptions: Mapped[list[LineTranscription]] = relationship(
        "LineTranscription",
        back_populates="transcription",
        cascade="all, delete-orphan",
        foreign_keys="LineTranscription.transcription_id",
    )


class LineTranscription(Base):
    __tablename__ = "line_transcriptions"
    __table_args__ = (
        UniqueConstraint("line_id", "transcription_id", name="uq_line_transcriptions_line_layer"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    line_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lines.id", ondelete="CASCADE"), index=True
    )
    transcription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcriptions.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: On a ground-truth row: the model layer this text was accepted from, recorded by
    #: ``copy_to_ground_truth`` and kept through every later human edit of the same
    #: row. Without it, "a human typed this from scratch" and "a human accepted the
    #: model's line and fixed two characters" are the same record, and the second is
    #: the more valuable of the two to measure. It dies only with the row itself,
    #: which is what ``GroundTruthText.write(text=None)`` does. SET NULL rather than
    #: CASCADE: deleting the model layer must not take the corrected text with it.
    seeded_from_transcription_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcriptions.id", ondelete="SET NULL"), nullable=True
    )

    line: Mapped[Line] = relationship("Line", back_populates="transcriptions")
    # ``foreign_keys`` is required now that two columns point at ``transcriptions``:
    # without it the mapper cannot tell which one owns this relationship and refuses
    # to configure at all. ``seeded_from_transcription_id`` gets no relationship of
    # its own on purpose - it is a provenance note, not a navigable edge, and giving
    # it one would invite a lazy load per line on every page read.
    transcription: Mapped[Transcription] = relationship(
        "Transcription",
        back_populates="line_transcriptions",
        foreign_keys=[transcription_id],
    )


class PageTranscriptionLine(Base):
    __tablename__ = "page_transcription_lines"
    __table_args__ = (
        UniqueConstraint("part_id", "order", name="uq_page_transcription_lines_part_order"),
        UniqueConstraint("paired_line_id", name="uq_page_transcription_lines_paired_line"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), index=True
    )
    order: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    paired_line_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("lines.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    part: Mapped[DocumentPart] = relationship(
        "DocumentPart", back_populates="page_transcription_lines"
    )
    paired_line: Mapped[Line | None] = relationship("Line")
