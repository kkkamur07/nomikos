"""``GET /admin/overview`` reads Postgres at most once a minute per window, under a timeout."""

from __future__ import annotations

import pytest

from backend.core.api import admin


class _Session:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: object, params: dict | None = None) -> None:
        self.statements.append(str(statement))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None]]:
    admin.clear_overview_cache()
    seen: list[tuple[str, int | None]] = []

    def card(name: str):
        async def read(db, *, days: int | None = None, pool_status: str | None = None):
            seen.append((name, days))
            return {"card": name}

        return read

    for name in ("users_stats", "activity_stats", "model_stats", "database_stats"):
        monkeypatch.setattr(admin.admin_stats_service, name, card(name))
    yield seen
    admin.clear_overview_cache()


async def test_statements_run_under_a_local_statement_timeout(calls) -> None:
    session = _Session()

    await admin._postgres_cards(session, days=30)

    assert session.statements == ["SET LOCAL statement_timeout = '5s'"]
    assert [name for name, _ in calls] == [
        "users_stats",
        "activity_stats",
        "model_stats",
        "database_stats",
    ]


async def test_a_second_load_within_a_minute_is_served_from_the_cache(calls, monkeypatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(admin.time, "monotonic", lambda: clock[0])

    first = await admin._postgres_cards(_Session(), days=30)
    clock[0] += 59
    session = _Session()
    second = await admin._postgres_cards(session, days=30)

    assert second == first
    assert session.statements == []
    assert len(calls) == 4

    clock[0] += 2
    await admin._postgres_cards(_Session(), days=30)
    assert len(calls) == 8


async def test_the_cache_is_keyed_by_window(calls) -> None:
    await admin._postgres_cards(_Session(), days=30)
    await admin._postgres_cards(_Session(), days=7)

    assert [days for name, days in calls if name == "users_stats"] == [30, 7]
