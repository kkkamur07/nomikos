"""The admin dashboard's aggregates: one statement per card, admins only.

The SQL itself runs against Postgres in the integration lane; what is pinned here is
the part a refactor would quietly break: each card stays a single round trip, the
rows are shaped the way the response model reads them, and only ``is_admin`` gets in.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime

import pytest
from fastapi import HTTPException

from backend.core.application import admin_stats_service
from backend.users.api.dependencies import require_admin
from backend.users.api.schemas import UserResponse
from backend.users.infrastructure.orm_models import User


class _Result:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self) -> _Result:
        return self

    def one(self) -> dict:
        assert len(self._rows) == 1
        return self._rows[0]

    def all(self) -> list[dict]:
        return self._rows


class _Session:
    """Answers every ``execute`` with the same canned rows and counts the calls."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.statements: list[tuple[object, dict | None]] = []

    async def execute(self, statement: object, params: dict | None = None) -> _Result:
        self.statements.append((statement, params))
        return _Result(self.rows)


async def test_users_card_is_one_statement_and_decodes_json_columns() -> None:
    user_id = uuid.uuid4()
    session = _Session(
        [
            {
                "total": 3,
                "admins": 1,
                "new_in_window": 2,
                "active_7d": 1,
                "active_30d": 2,
                "active_in_window": 2,
                # A driver without the asyncpg json codec hands back text.
                "new_per_week": json.dumps([{"week": "2026-10-05", "count": 2}]),
                "top_users": [{"id": str(user_id), "email": "a@x.org", "jobs": 4}],
            }
        ]
    )

    users = await admin_stats_service.users_stats(session, days=7)

    assert len(session.statements) == 1
    assert session.statements[0][1]["since"] <= datetime.now(UTC)
    assert users["new_per_week"] == [{"week": "2026-10-05", "count": 2}]
    assert users["top_users"][0]["jobs"] == 4
    assert users["total"] == 3


async def test_activity_card_is_one_statement() -> None:
    session = _Session(
        [
            {
                "jobs": [],
                "pages": [{"day": "2026-10-11", "count": 5}],
                "lines": None,
                "annotations": [],
            }
        ]
    )

    activity = await admin_stats_service.activity_stats(session, days=30)

    assert len(session.statements) == 1
    assert activity["pages"] == [{"day": "2026-10-11", "count": 5}]
    assert activity["lines"] == []


async def test_models_card_derives_success_rate_from_finished_jobs_only() -> None:
    session = _Session(
        [
            {
                "name": "kraken",
                "task": "segment",
                "provider": "kraken",
                "jobs": 10,
                "done": 6,
                "failed": 2,
                "median_seconds": 12.3456,
            },
            {
                "name": "(no model)",
                "task": "pipeline",
                "provider": None,
                "jobs": 3,
                "done": 0,
                "failed": 0,
                "median_seconds": None,
            },
        ]
    )

    models = await admin_stats_service.model_stats(session, days=30)

    assert len(session.statements) == 1
    # Pending and running jobs are not failures: 6 of the 8 that finished.
    assert models[0]["success_rate"] == 0.75
    assert models[0]["median_seconds"] == 12.35
    assert models[1]["success_rate"] is None
    assert models[1]["median_seconds"] is None


async def test_database_card_is_one_statement_and_carries_the_pool_status() -> None:
    session = _Session(
        [
            {
                "name": "nomikos",
                "xact_commit": 10,
                "busiest_tables": "[]",
                "seq_scan_tables": [],
                "largest_tables": None,
            }
        ]
    )

    database = await admin_stats_service.database_stats(session, pool_status="Pool size: 5")

    assert len(session.statements) == 1
    assert database["busiest_tables"] == []
    assert database["largest_tables"] == []
    assert database["pool_status"] == "Pool size: 5"


def test_window_start_counts_back_whole_days() -> None:
    now = datetime(2026, 10, 11, 12, tzinfo=UTC)
    assert admin_stats_service.window_start(30, now=now).date() == date(2026, 9, 11)


def _user(*, is_admin: bool) -> User:
    return User(
        id=uuid.uuid4(),
        email="someone@example.com",
        username="someone",
        hashed_password="x",
        prefer_local_inference=False,
        is_admin=is_admin,
        created_at=datetime.now(UTC),
    )


async def test_require_admin_refuses_a_signed_in_non_admin_with_403() -> None:
    with pytest.raises(HTTPException) as refused:
        await require_admin(_user(is_admin=False))
    assert refused.value.status_code == 403


async def test_require_admin_lets_an_admin_through() -> None:
    admin = _user(is_admin=True)
    assert await require_admin(admin) is admin


def test_current_user_response_exposes_is_admin() -> None:
    assert UserResponse.model_validate(_user(is_admin=True)).is_admin is True
    assert UserResponse.model_validate(_user(is_admin=False)).is_admin is False
