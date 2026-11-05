"""Health check routes."""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.schema_revision import (
    SchemaState,
    code_revisions,
    compare_schema_revision,
    read_database_revision,
)
from backend.core.schemas.health import HealthResponse
from backend.core.settings.job import get_job_settings
from backend.jobs.infrastructure.job_repository import JobRepository
from infrastructure.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


async def _oldest_pending_job_seconds(db: AsyncSession) -> float | None:
    """Read the queue's head age and warn when it is past the stall threshold.

    The WARNING is the part that actually alerts. Nobody reads a probe's response
    body, so a number alone would be as silent as the queue it describes; a
    WARNING line lands in log-based alerting with no extra infrastructure. It is
    emitted on every probe past the threshold rather than once per crossing,
    deliberately: alerting wants a rate, and a warning that stops arriving must
    mean the queue drained, not that a process-local latch is still set.

    A failure to read the queue is logged with its traceback and reported as
    ``None`` rather than raised. ``SELECT 1`` has already succeeded by the time
    this runs, so the probe's own subject - can this process reach Postgres - is
    answered; failing the request here would take the API out of rotation over a
    supplementary number.
    """
    try:
        age_seconds = await JobRepository(db).seconds_since_oldest_pending_job()
    except SQLAlchemyError:
        logger.exception("health check could not read the pending job queue")
        return None
    threshold = get_job_settings().job_queue_stall_warning_seconds
    if age_seconds is not None and age_seconds >= threshold:
        logger.warning(
            "oldest pending job has waited %.0fs (threshold %.0fs): "
            "no platform worker or inference agent is claiming the queue",
            age_seconds,
            threshold,
        )
    return age_seconds


#: The last ``(state, database revision, code head)`` this process logged. The
#: schema line is written only when this changes, so an unauthenticated client
#: looping ``/health`` cannot turn a lasting state into a log flood.
_last_schema_report: tuple[SchemaState, str | None, str | None] | None = None


def _log_schema_transition(state: SchemaState, db_revision: str | None, head: str | None) -> None:
    global _last_schema_report
    report = (state, db_revision, head)
    if report == _last_schema_report:
        return
    previous = _last_schema_report
    _last_schema_report = report
    if state == "behind":
        logger.error(
            "database schema %s is behind code head %s: apply the migration SQL",
            db_revision,
            head,
        )
    elif state == "ahead":
        logger.warning(
            "database schema %s is not a revision this code knows (head %s): "
            "expected only while a release's SQL is applied ahead of its deploy",
            db_revision,
            head,
        )
    elif state == "unknown":
        logger.warning(
            "database schema could not be compared (database %s, code head %s)",
            db_revision,
            head,
        )
    elif previous is not None:
        logger.info("database schema %s matches code head %s again", db_revision, head)


async def _schema_check(db: AsyncSession) -> SchemaState:
    """Compare ``alembic_version`` with the shipped head; log the state on a change.

    Only ``behind`` is an ERROR: it is the one state in which this process cannot
    serve requests. ``ahead`` is the documented release order (SQL applied before
    merge, old code still running) and ``unknown`` means one side could not be
    read; both warn and are otherwise reported only. The log names both revisions;
    the public body carries only the state, so the probe does not fingerprint the
    deployed release.
    """
    code = code_revisions()
    db_revision = await read_database_revision(db)
    state = compare_schema_revision(db_revision, code)
    _log_schema_transition(state, db_revision, code.head if code is not None else None)
    return state


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {
            "model": HealthResponse,
            "description": "Database unreachable, or its schema is behind the code's alembic head",
        }
    },
)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse | JSONResponse:
    """Liveness/readiness (``SELECT 1``, schema revision) plus the oldest pending job's age.

    Dev-user seeding lives in app lifespan.

    **The queue age never changes the status code.** A pending job piles up
    because a *different* host stopped claiming - the platform worker
    (``JOB_WORKER_ENABLED``, which the API deployment sets false) or a
    researcher's inference agent. Returning 503 for that would pull a perfectly
    healthy API out of rotation, or restart it, and neither brings the missing
    consumer back. The status code answers "can this process serve requests";
    the queue age is reported next to it, and the WARNING in
    ``_oldest_pending_job_seconds`` is what pages someone.

    It rides on ``/health`` rather than a separate ``/health/jobs`` because the
    thing that has to happen on a schedule is the threshold check, and ``/health``
    is the only route infrastructure already polls on a schedule. A route nobody
    is configured to call would put the queue's one alarm behind another manual
    deployment step - the same class of gap that leaves the queue unclaimed in
    the first place. The cost is one ``min(created_at)`` against the
    ``ix_jobs_claim_pending`` partial index per probe.

    **A schema behind the code is the one other 503.** Production never runs
    alembic; migration SQL is applied by hand. When the code ships a revision
    the database does not have yet, every query touching the new columns fails,
    so this process genuinely cannot serve requests, and that is exactly what the
    status code answers. Unlike the queue age, the fix belongs to this deploy:
    a 503 keeps the new container from going healthy, so the image's
    HEALTHCHECK holds the rollout instead of letting it 500 on real traffic. A
    database *ahead* of the code is the documented release order (SQL before
    merge) and stays 200; an unreadable revision on either side stays 200 too,
    because "could not check" is not "cannot serve". The revision read runs last
    so a failure there cannot poison the queue query's transaction.
    """
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        body = HealthResponse(status="degraded", database="error")
        return JSONResponse(status_code=503, content=body.model_dump(by_alias=True))

    oldest_pending_job_seconds = await _oldest_pending_job_seconds(db)
    schema_state = await _schema_check(db)
    if schema_state == "behind":
        body = HealthResponse(
            status="degraded",
            database="ok",
            oldest_pending_job_seconds=oldest_pending_job_seconds,
            schema_state=schema_state,
        )
        return JSONResponse(status_code=503, content=body.model_dump(by_alias=True))

    return HealthResponse(
        status="ok",
        database="ok",
        oldest_pending_job_seconds=oldest_pending_job_seconds,
        schema_state=schema_state,
    )
