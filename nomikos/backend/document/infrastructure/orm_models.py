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
    # What the researcher calls this region, typically a folio ("4v"). Exports print it
    # as the region's heading (ADR 0011).
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    part: Mapped[DocumentPart] = relationship("DocumentPart", back_populates="blocks")
    lines: Mapped[list[Line]] = relationship("Line", back_populates="block")


class Line(Base):
    __tablename__ = "lines"
    __table_args__ = (Index("ix_lines_part_order", "part_id", "order", "created_at"),)

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
    kraken_ceiling: Mapped[list[list[float]] | None] = mapped_column(JSONB, nullable=True)
    manual_geometry: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="transcriptions")
    line_transcriptions: Mapped[list[LineTranscription]] = relationship(
        "LineTranscription", back_populates="transcription", cascade="all, delete-orphan"
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

    line: Mapped[Line] = relationship("Line", back_populates="transcriptions")
    transcription: Mapped[Transcription] = relationship(
        "Transcription", back_populates="line_transcriptions"
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
