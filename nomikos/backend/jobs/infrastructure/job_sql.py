"""SQL fragments shared by the claim, sweep and terminal writes on ``jobs``."""

from __future__ import annotations

from typing import cast

from sqlalchemy import ColumnElement, CursorResult, Result, or_

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


def not_held_by_agent() -> ColumnElement[bool]:
    """SQL for the complement, NULL included.

    ``NOT LIKE`` alone evaluates to NULL for an unclaimed row and would silently
    exempt every job the platform itself dispatched, which is the whole
    population the waiting timeout exists for.
    """
    return or_(Job.claimed_by.is_(None), ~Job.claimed_by.startswith(AGENT_CLAIM_PREFIX))


def rows_affected(result: Result[tuple[object, ...]]) -> int:
    """Rows an UPDATE or DELETE touched.

    ``Session.execute`` is typed as returning ``Result`` for every statement,
    but a DML statement always yields a ``CursorResult``, which is where
    ``rowcount`` lives.
    """
    return cast("CursorResult[tuple[object, ...]]", result).rowcount or 0
