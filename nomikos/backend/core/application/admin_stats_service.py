"""Aggregate queries behind the admin dashboard.

One async function per dashboard card, and each issues **one** SQL statement. A card
that needs several shapes (a total, a weekly series, a top-20 list) gets them as JSON
sub-selects of that one statement rather than as follow-up queries, so opening the
dashboard costs a fixed handful of round trips whatever the data looks like - no
per-user or per-model loop.

Every window is ``created_at >= since`` with ``since = now - days``, and every day
bucket is the UTC date. Nothing here writes.

Two things the schema cannot answer exactly, and what is used instead:

* **Lines approved / corrected.** Lines carry no approval timestamp. Each page save
  writes an ``annotation_history_snapshots`` row holding ``paired_line_count`` (lines
  with approved text at that moment), so a save counts as one correction and the rise
  in ``paired_line_count`` since the page's previous snapshot counts as newly approved
  lines. History keeps only the last few snapshots per page
  (``HISTORY_RETENTION_LIMIT``) and a page's oldest kept snapshot has nothing to
  compare with, so its approvals are not counted: older days under-count.
* **Who approved them.** Snapshots record no user. Approvals are credited to the
  owner of the project the page belongs to.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

TOP_USERS = 20
TOP_TABLES = 10

# Approval deltas per snapshot. The window function runs over every snapshot of a
# page (the previous one may predate the window) and the outer query filters. A page's
# oldest kept snapshot has no predecessor (history is pruned), so it counts as a
# correction with no approvals rather than crediting every line it holds as new;
# ``greatest`` ignores the NULL ``lag`` and yields 0.
_APPROVAL_DELTAS = """
    SELECT s.part_id, s.created_at,
           greatest(
               s.paired_line_count
               - lag(s.paired_line_count) OVER (PARTITION BY s.part_id ORDER BY s.created_at),
               0
           ) AS approved
    FROM annotation_history_snapshots s
"""

# A browser session is rotated in place on refresh, so ``updated_at`` is its last use.
_SESSION_LAST_USED = "greatest(created_at, updated_at)"

_USERS_SQL = text(
    f"""
    WITH approvals AS (
        SELECT p.owner_id AS user_id, sum(a.approved) AS lines_approved
        FROM ({_APPROVAL_DELTAS}) a
        JOIN document_parts dp ON dp.id = a.part_id
        JOIN documents d ON d.id = dp.document_id
        JOIN projects p ON p.id = d.project_id
        WHERE a.created_at >= :since AND p.owner_id IS NOT NULL
        GROUP BY p.owner_id
    ),
    job_counts AS (
        SELECT user_id, count(*) AS jobs
        FROM jobs
        WHERE created_at >= :since AND user_id IS NOT NULL
        GROUP BY user_id
    ),
    last_seen AS (
        SELECT user_id, max({_SESSION_LAST_USED}) AS last_seen
        FROM auth_sessions
        GROUP BY user_id
    )
    SELECT
        (SELECT count(*) FROM users) AS total,
        (SELECT count(*) FROM users WHERE is_admin) AS admins,
        (SELECT count(*) FROM users WHERE created_at >= :since) AS new_in_window,
        (SELECT count(DISTINCT user_id) FROM auth_sessions
            WHERE {_SESSION_LAST_USED} >= now() - interval '7 days') AS active_7d,
        (SELECT count(DISTINCT user_id) FROM auth_sessions
            WHERE {_SESSION_LAST_USED} >= now() - interval '30 days') AS active_30d,
        (SELECT count(DISTINCT user_id) FROM auth_sessions
            WHERE {_SESSION_LAST_USED} >= :since) AS active_in_window,
        (SELECT coalesce(json_agg(w ORDER BY w.week), '[]'::json) FROM (
            SELECT date_trunc('week', created_at AT TIME ZONE 'UTC')::date AS week,
                   count(*) AS count
            FROM users WHERE created_at >= :since GROUP BY 1
        ) w) AS new_per_week,
        (SELECT coalesce(json_agg(t), '[]'::json) FROM (
            SELECT u.id, u.email, u.username, u.is_admin, u.created_at,
                   coalesce(j.jobs, 0) AS jobs,
                   coalesce(a.lines_approved, 0) AS lines_approved,
                   s.last_seen
            FROM users u
            LEFT JOIN job_counts j ON j.user_id = u.id
            LEFT JOIN approvals a ON a.user_id = u.id
            LEFT JOIN last_seen s ON s.user_id = u.id
            ORDER BY coalesce(j.jobs, 0) DESC, coalesce(a.lines_approved, 0) DESC,
                     s.last_seen DESC NULLS LAST, u.email
            LIMIT {TOP_USERS}
        ) t) AS top_users
    """  # noqa: S608 - only module constants are interpolated, never input
)

_ACTIVITY_SQL = text(
    f"""
    SELECT
        (SELECT coalesce(json_agg(j ORDER BY j.day, j.kind, j.status), '[]'::json) FROM (
            SELECT (created_at AT TIME ZONE 'UTC')::date AS day,
                   type::text AS kind, status::text AS status, count(*) AS count
            FROM jobs WHERE created_at >= :since GROUP BY 1, 2, 3
        ) j) AS jobs,
        (SELECT coalesce(json_agg(p ORDER BY p.day), '[]'::json) FROM (
            SELECT (created_at AT TIME ZONE 'UTC')::date AS day, count(*) AS count
            FROM document_parts WHERE created_at >= :since GROUP BY 1
        ) p) AS pages,
        (SELECT coalesce(json_agg(l ORDER BY l.day), '[]'::json) FROM (
            SELECT (created_at AT TIME ZONE 'UTC')::date AS day, count(*) AS count
            FROM lines WHERE created_at >= :since GROUP BY 1
        ) l) AS lines,
        (SELECT coalesce(json_agg(a ORDER BY a.day), '[]'::json) FROM (
            SELECT (created_at AT TIME ZONE 'UTC')::date AS day,
                   count(*) AS corrections, sum(approved) AS approved
            FROM ({_APPROVAL_DELTAS}) deltas
            WHERE created_at >= :since GROUP BY 1
        ) a) AS annotations
    """  # noqa: S608 - only module constants are interpolated, never input
)

_MODELS_SQL = text(
    """
    SELECT coalesce(m.name, '(no model)') AS name,
           coalesce(m.task::text, j.type::text) AS task,
           m.provider,
           count(*) AS jobs,
           count(*) FILTER (WHERE j.status = 'done') AS done,
           count(*) FILTER (WHERE j.status = 'failed') AS failed,
           percentile_cont(0.5) WITHIN GROUP (
               ORDER BY extract(epoch FROM j.completed_at - j.created_at)
           ) FILTER (WHERE j.status = 'done' AND j.completed_at IS NOT NULL)
               AS median_seconds
    FROM jobs j
    LEFT JOIN inference_models m ON m.id = j.model_id
    WHERE j.created_at >= :since
    GROUP BY 1, 2, 3
    ORDER BY jobs DESC, name
    """
)

_TABLE_COLUMNS = """
    relname AS name, n_tup_ins AS inserts, n_tup_upd AS updates, n_tup_del AS deletes,
    seq_scan, coalesce(idx_scan, 0) AS idx_scan, n_live_tup AS live_rows,
    pg_total_relation_size(relid) AS total_bytes
"""

_DATABASE_SQL = text(
    f"""
    SELECT d.datname AS name,
           pg_database_size(d.datname) AS size_bytes,
           d.numbackends, d.xact_commit, d.xact_rollback, d.blks_hit, d.blks_read,
           d.tup_returned, d.tup_fetched, d.tup_inserted, d.tup_updated, d.tup_deleted,
           d.stats_reset,
           (SELECT count(*) FROM pg_stat_activity
               WHERE datname = d.datname AND state = 'active') AS active_queries,
           (SELECT coalesce(json_agg(t), '[]'::json) FROM (
               SELECT {_TABLE_COLUMNS} FROM pg_stat_user_tables
               ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC, relname LIMIT {TOP_TABLES}
           ) t) AS busiest_tables,
           (SELECT coalesce(json_agg(t), '[]'::json) FROM (
               SELECT {_TABLE_COLUMNS} FROM pg_stat_user_tables
               ORDER BY seq_scan DESC, relname LIMIT {TOP_TABLES}
           ) t) AS seq_scan_tables,
           (SELECT coalesce(json_agg(t), '[]'::json) FROM (
               SELECT {_TABLE_COLUMNS} FROM pg_stat_user_tables
               ORDER BY pg_total_relation_size(relid) DESC, relname LIMIT {TOP_TABLES}
           ) t) AS largest_tables
    FROM pg_stat_database d
    WHERE d.datname = current_database()
    """  # noqa: S608 - only module constants are interpolated, never input
)


def window_start(days: int, *, now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) - timedelta(days=days)


def _json(value: Any) -> Any:
    """The asyncpg dialect decodes ``json`` columns; a raw driver hands back text."""
    if isinstance(value, str):
        return json.loads(value)
    return value if value is not None else []


async def users_stats(db: AsyncSession, *, days: int = 30) -> dict[str, Any]:
    row = (await db.execute(_USERS_SQL, {"since": window_start(days)})).mappings().one()
    return {
        "total": row["total"],
        "admins": row["admins"],
        "new_in_window": row["new_in_window"],
        "active_7d": row["active_7d"],
        "active_30d": row["active_30d"],
        "active_in_window": row["active_in_window"],
        "new_per_week": _json(row["new_per_week"]),
        "top_users": _json(row["top_users"]),
    }


async def activity_stats(db: AsyncSession, *, days: int = 30) -> dict[str, Any]:
    row = (await db.execute(_ACTIVITY_SQL, {"since": window_start(days)})).mappings().one()
    return {
        "jobs": _json(row["jobs"]),
        "pages": _json(row["pages"]),
        "lines": _json(row["lines"]),
        "annotations": _json(row["annotations"]),
    }


async def model_stats(db: AsyncSession, *, days: int = 30) -> list[dict[str, Any]]:
    rows = (await db.execute(_MODELS_SQL, {"since": window_start(days)})).mappings().all()
    models = []
    for row in rows:
        finished = row["done"] + row["failed"]
        models.append(
            {
                **row,
                "success_rate": round(row["done"] / finished, 4) if finished else None,
                "median_seconds": (
                    round(float(row["median_seconds"]), 2)
                    if row["median_seconds"] is not None
                    else None
                ),
            }
        )
    return models


async def database_stats(db: AsyncSession, *, pool_status: str | None = None) -> dict[str, Any]:
    row = dict((await db.execute(_DATABASE_SQL)).mappings().one())
    for key in ("busiest_tables", "seq_scan_tables", "largest_tables"):
        row[key] = _json(row[key])
    row["pool_status"] = pool_status
    return row
