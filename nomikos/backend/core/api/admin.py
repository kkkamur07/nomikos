"""Admin dashboard: platform usage and database load, for ``users.is_admin`` only.

One route, ``GET /admin/overview``, so the dashboard is one request and the
statements behind it run on one pooled connection. Everything is an aggregate;
nothing here reads a page image, a transcription text, or another user's document.
See ADR "Admin dashboard and request metrics".
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.api import request_metrics
from backend.core.application import admin_stats_service, feedback_service
from backend.users.api.dependencies import require_admin
from backend.users.infrastructure.orm_models import User
from infrastructure.db import engine, get_db

router = APIRouter(prefix="/admin", tags=["admin"])

#: How long one window's Postgres cards are reused, so reloading the dashboard or
#: flipping the 7/30/90 selector does not rescan ``lines`` and the history table.
OVERVIEW_CACHE_SECONDS = 60.0
#: Ceiling for each overview statement; a slow dashboard must not hold a connection.
OVERVIEW_STATEMENT_TIMEOUT = "5s"

_overview_cache: dict[int, tuple[float, dict[str, Any]]] = {}


def clear_overview_cache() -> None:
    _overview_cache.clear()


class WeekCount(BaseModel):
    week: date
    count: int


class AdminUserRow(BaseModel):
    id: UUID
    email: str
    username: str
    is_admin: bool
    created_at: datetime
    jobs: int
    lines_approved: int
    last_seen: datetime | None = None


class UsersOverview(BaseModel):
    total: int
    admins: int
    new_in_window: int
    active_7d: int
    active_30d: int
    active_in_window: int
    new_per_week: list[WeekCount]
    top_users: list[AdminUserRow]


class JobDayCount(BaseModel):
    day: date
    kind: str
    status: str
    count: int


class DayCount(BaseModel):
    day: date
    count: int


class AnnotationDayCount(BaseModel):
    day: date
    corrections: int
    approved: int


class ActivityOverview(BaseModel):
    jobs: list[JobDayCount]
    pages: list[DayCount]
    lines: list[DayCount]
    annotations: list[AnnotationDayCount]


class ModelUsage(BaseModel):
    name: str
    task: str
    provider: str | None = None
    jobs: int
    done: int
    failed: int
    success_rate: float | None = None
    median_seconds: float | None = None


class ClientFailureEntry(BaseModel):
    at: datetime
    path: str | None = None
    status: int | None = None
    source: str | None = None
    message: str


class ClientFailuresOverview(BaseModel):
    started_at: datetime
    per_day: list[DayCount]
    recent: list[ClientFailureEntry]


class TableStats(BaseModel):
    name: str
    inserts: int
    updates: int
    deletes: int
    seq_scan: int
    idx_scan: int
    live_rows: int
    total_bytes: int


class DatabaseOverview(BaseModel):
    name: str
    size_bytes: int
    numbackends: int
    active_queries: int
    xact_commit: int
    xact_rollback: int
    blks_hit: int
    blks_read: int
    tup_returned: int
    tup_fetched: int
    tup_inserted: int
    tup_updated: int
    tup_deleted: int
    stats_reset: datetime | None = None
    pool_status: str | None = None
    busiest_tables: list[TableStats]
    seq_scan_tables: list[TableStats]
    largest_tables: list[TableStats]


class RouteMetric(BaseModel):
    method: str
    route: str
    status_class: str
    count: int
    avg_ms: float
    max_ms: float


class RequestsOverview(BaseModel):
    started_at: datetime
    total_requests: int
    routes: list[RouteMetric]


class AdminOverview(BaseModel):
    days: int
    generated_at: datetime
    users: UsersOverview
    activity: ActivityOverview
    models: list[ModelUsage]
    client_failures: ClientFailuresOverview
    database: DatabaseOverview
    requests: RequestsOverview


def _pool_status() -> str | None:
    status = getattr(engine.pool, "status", None)
    return status() if callable(status) else None


async def _postgres_cards(db: AsyncSession, *, days: int) -> dict[str, Any]:
    """The four cards that query Postgres, cached per ``days`` in this process."""
    now = time.monotonic()
    cached = _overview_cache.get(days)
    if cached is not None and now - cached[0] < OVERVIEW_CACHE_SECONDS:
        return cached[1]
    # LOCAL: ends with this request's transaction, so the pooled connection
    # goes back without it.
    await db.execute(text(f"SET LOCAL statement_timeout = '{OVERVIEW_STATEMENT_TIMEOUT}'"))
    cards = {
        "users": await admin_stats_service.users_stats(db, days=days),
        "activity": await admin_stats_service.activity_stats(db, days=days),
        "models": await admin_stats_service.model_stats(db, days=days),
        "database": await admin_stats_service.database_stats(db),
    }
    _overview_cache[days] = (now, cards)
    return cards


@router.get("/overview", response_model=AdminOverview)
async def admin_overview(
    _admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> AdminOverview:
    """Everything the dashboard shows, for the last ``days`` days.

    Per-route request counts, client failures and the pool are this API process's
    own, since ``started_at``, and always current; the rest comes from Postgres and
    may be up to 60 seconds old.
    """
    cards = await _postgres_cards(db, days=days)
    return AdminOverview.model_validate(
        {
            "days": days,
            "generated_at": admin_stats_service.window_start(0),
            "users": cards["users"],
            "activity": cards["activity"],
            "models": cards["models"],
            "client_failures": request_metrics.client_failures_snapshot(days=days),
            "database": {**cards["database"], "pool_status": _pool_status()},
            "requests": request_metrics.snapshot(),
        }
    )


class FeedbackRow(BaseModel):
    id: UUID
    created_at: datetime
    email: str
    username: str
    message: str
    page_path: str | None = None


@router.get("/feedback", response_model=list[FeedbackRow])
async def admin_feedback(
    _admin: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[FeedbackRow]:
    """Newest feedback notes with their authors, for the admin dashboard."""
    rows = await feedback_service.list_feedback(db, limit=limit)
    return [FeedbackRow.model_validate(row) for row in rows]
