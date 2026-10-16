from __future__ import annotations

import uuid
from datetime import datetime
from typing import NotRequired

from sqlalchemy import DateTime, ForeignKey, Index, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import TypedDict

from infrastructure.db import Base


class SnapshotLineState(TypedDict):
    """One line as stored in a history snapshot's ``lines`` list."""

    id: str
    block_id: str | None
    order: int
    kind: str
    points: object
    baseline: NotRequired[object]
    mask: NotRequired[object]
    source: str
    source_metadata: object
    kraken_ceiling: object
    approved_text: str | None


class SnapshotState(TypedDict):
    """The restorable page state stored on a history snapshot row."""

    lines: list[SnapshotLineState]


class AnnotationHistorySnapshot(Base):
    """Compact restorable Page annotation state."""

    __tablename__ = "annotation_history_snapshots"
    __table_args__ = (
        Index(
            "ix_annotation_history_snapshots_part_created",
            "part_id",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    part_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[SnapshotState] = mapped_column(JSONB)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    paired_line_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
