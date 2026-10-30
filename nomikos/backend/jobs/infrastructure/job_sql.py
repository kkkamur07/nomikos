"""SQL fragments shared by the claim, sweep and terminal writes on ``jobs``."""

from __future__ import annotations

from sqlalchemy import ColumnElement

import infrastructure.models  # noqa: F401 - register all ORM mappers
from backend.jobs.infrastructure.orm_models import Job, JobStatus

NON_TERMINAL_STATUSES = (JobStatus.pending, JobStatus.running, JobStatus.waiting)

# Prefix ``jobs.claimed_by`` carries while an inference agent holds a page,
# written by ``job_claim_service.agent_claim_owner``. The platform worker's own
# identity may never take this shape, see ``job_claim_engine.worker_identity``.
AGENT_CLAIM_PREFIX = "agent:"


def held_by_agent() -> ColumnElement[bool]:
    """SQL for "an inference agent holds this page"."""
    return Job.claimed_by.startswith(AGENT_CLAIM_PREFIX)
