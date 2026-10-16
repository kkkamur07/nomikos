from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, Text, event, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.orm.base import NEVER_SET, NO_VALUE

from backend.core.exceptions import ConflictError
from backend.ml.domain.execution import ExecutionTarget
from infrastructure.db import Base

if TYPE_CHECKING:
    from backend.ml.infrastructure.orm_models import InferenceModel, ModelBinding


class JobStatus(StrEnum):
    pending = "pending"
    waiting = "waiting"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class JobType(StrEnum):
    segment = "segment"
    transcribe = "transcribe"
    binarize = "binarize"
    pipeline = "pipeline"


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_payload_gin", "payload", postgresql_using="gin"),
        # The platform worker's claim: ordered by age over the pending rows.
        Index(
            "ix_jobs_claim_pending",
            "created_at",
            "id",
            postgresql_where="status = 'pending'",
        ),
        # The **agent** claim (``job_claim_service._claimable_job_query``), which
        # the one above does not serve. That query adds ``execution_target`` and,
        # for a device token, ``user_id`` - both far more selective than the
        # partial predicate - so with only the index above every long poll scanned
        # the whole pending queue in ``created_at`` order and discarded rows for
        # other hosts and other accounts. Every connected agent runs this once a
        # second, so it is the hottest query in the schema.
        #
        # Column order is the two equality predicates first, then the sort key, so
        # one index range answers both the filter and the ORDER BY without a sort.
        Index(
            "ix_jobs_claim_target_pending",
            "execution_target",
            "user_id",
            "created_at",
            "id",
            postgresql_where="status = 'pending'",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inference_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    type: Mapped[JobType] = mapped_column(Enum(JobType, name="job_type"))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.pending, index=True
    )
    # The **inference host** this job runs on, fixed at submission. Never written
    # again - see ``_execution_target_is_fixed`` below and the database trigger
    # in migration 006.
    execution_target: Mapped[ExecutionTarget] = mapped_column(
        Enum(ExecutionTarget, name="execution_target"),
        nullable=False,
        server_default=ExecutionTarget.cloud.value,
        default=ExecutionTarget.cloud,
    )
    # The host the account setting asked for. Equal to ``execution_target``
    # unless the preferred host had no **capacity**, which is exactly the case
    # the researcher has to be told about - so it is recorded on the job rather
    # than announced once and lost.
    preferred_execution_target: Mapped[ExecutionTarget] = mapped_column(
        Enum(ExecutionTarget, name="execution_target"),
        nullable=False,
        server_default=ExecutionTarget.cloud.value,
        default=ExecutionTarget.cloud,
    )
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("inference_models.id", ondelete="SET NULL"), nullable=True
    )
    binding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_bindings.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    document_part_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_parts.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How many claims on this page have been abandoned. Incremented by whichever
    # sweep takes the claim back, reset by a successful terminal write. Without
    # it a page that reliably kills whatever runs it cycles pending -> waiting ->
    # pending forever and never reaches a terminal status; see
    # ``job_sweeps.MAX_CLAIM_ATTEMPTS``.
    claim_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", default=0
    )
    # Write-only today: the claim and the platform worker both stamp it, nothing
    # reads it. It is kept because the two lease timeouts are sized on the
    # assumption that a stalled run is invisible, and this is the column a
    # heartbeat would land in. Do not treat it as liveness - nothing checks it.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    callback_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model: Mapped[InferenceModel | None] = relationship("InferenceModel", back_populates="jobs")
    binding: Mapped[ModelBinding | None] = relationship("ModelBinding", back_populates="jobs")

    @property
    def execution_target_substituted(self) -> bool:
        """Whether the preferred host had no **capacity** at submission."""
        return self.execution_target != self.preferred_execution_target


EXECUTION_TARGET_IS_FIXED = (
    "A job's execution target is fixed when the job is submitted and cannot be changed"
)


@event.listens_for(Job.execution_target, "set", active_history=True)
def _execution_target_is_fixed(
    _target: Job, value: ExecutionTarget, oldvalue: object, _initiator: object
) -> ExecutionTarget:
    """Refuse to move a job between **inference host**s after submission.

    "Fixed at submission and never changed afterwards" is the property the whole
    of ADR 0002 rests on: it is what makes a missing local agent an ordinary,
    announced state instead of a job silently rerouted or left unclaimed. A
    property nothing enforces is a comment, so this raises rather than logs.

    ``active_history`` loads the stored value before the assignment lands, so an
    expired instance cannot slip a change past on a ``NO_VALUE`` old value. The
    first write - the constructor, and loading from the database - is not a
    change and passes through. Migration 006 puts the same rule in Postgres, so
    it also holds for statements that never touch this mapper.
    """
    if oldvalue is NO_VALUE or oldvalue is NEVER_SET or oldvalue is None:
        return value
    if value != oldvalue:
        raise ConflictError(EXECUTION_TARGET_IS_FIXED)
    return value
