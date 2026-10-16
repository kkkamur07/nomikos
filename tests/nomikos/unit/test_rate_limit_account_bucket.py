"""The account bucket is charged under its lock and refunded on success.

The per-account login bucket used to be read without a lock and charged only
after a failed password check, so N parallel attempts all saw the bucket under
the cap and a burst slipped the limit. It is now charged up front by
``consume_rate_limit`` - which holds the key's advisory lock across the
check-and-charge - and ``refund_rate_limit`` returns the charge when the
attempt succeeds, so only failures spend budget.

These tests drive the real primitives against an in-memory stand-in for the
attempt table: the fake interprets the actual SQLAlchemy statements
(advisory-lock ``text()``, expiry ``delete()``, count ``select()``), so the
accounting, expiry and lock-taking asserted here are the queries' own
predicates, not a reimplementation of them.
"""

from __future__ import annotations

import hashlib
import operator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import Delete, Select, TextClause
from sqlalchemy.sql.elements import BooleanClauseList
from starlette.requests import Request

from backend.core.exceptions import InvalidCredentialsError
from backend.core.settings import get_auth_settings, reset_settings_caches
from backend.users.api import rate_limit
from backend.users.infrastructure.orm_models import AuthRateLimitAttempt

STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


def _criteria(stmt: Select | Delete) -> list:
    clause = stmt.whereclause
    if clause is None:
        return []
    if isinstance(clause, BooleanClauseList) and clause.operator is operator.and_:
        return list(clause.clauses)
    return [clause]


def _matches(stmt: Select | Delete):
    criteria = _criteria(stmt)

    def matches(row: AuthRateLimitAttempt) -> bool:
        for crit in criteria:
            actual = getattr(row, crit.left.key)
            expected = crit.right.value
            if crit.operator == operator.eq:
                ok = actual == expected
            elif crit.operator == operator.ge:
                ok = actual >= expected
            elif crit.operator == operator.lt:
                ok = actual < expected
            else:
                raise AssertionError(f"unexpected operator {crit.operator}")
            if not ok:
                return False
        return True

    return matches


class _FakeSession:
    """Interprets the rate limiter's statements against a row list."""

    def __init__(self, store: list[AuthRateLimitAttempt]) -> None:
        self._store = store
        self.locks: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, stmt: object, params: dict | None = None) -> None:
        if isinstance(stmt, TextClause):
            assert params is not None
            self.locks.append(params["key"])
            return None
        assert isinstance(stmt, Delete)
        keep = _matches(stmt)
        self._store[:] = [row for row in self._store if not keep(row)]
        return None

    async def scalar(self, stmt: object) -> int | None:
        assert isinstance(stmt, Select)
        keep = _matches(stmt)
        rows = [row for row in self._store if keep(row)]
        descriptions = stmt.column_descriptions
        if len(descriptions) == 1 and descriptions[0]["name"] == "id":
            if not rows:
                return None
            rows.sort(key=lambda row: (row.attempted_at, row.id))
            return rows[-1].id
        return len(rows)

    def add(self, row: AuthRateLimitAttempt) -> None:
        row.id = len(self._store) + 1
        self._store.append(row)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


class _SessionFactory:
    def __init__(self) -> None:
        self.store: list[AuthRateLimitAttempt] = []
        self.sessions: list[_FakeSession] = []

    def __call__(self) -> _FakeSession:
        session = _FakeSession(self.store)
        self.sessions.append(session)
        return session


@pytest.fixture
def sessions(monkeypatch) -> _SessionFactory:
    factory = _SessionFactory()
    monkeypatch.setattr(rate_limit, "AsyncSessionLocal", factory)
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    reset_settings_caches()
    return factory


def _request() -> Request:
    request = Request(
        {
            "type": "http",
            "headers": [(b"host", b"testserver")],
            "client": None,
            "scheme": "http",
            "method": "POST",
            "path": "/auth/login",
            "query_string": b"",
            "server": ("testserver", 80),
        }
    )
    request._body = b'{"email":"victim@example.com","password":"x"}'
    return request


async def test_consume_charges_then_rejects_when_full(sessions: _SessionFactory) -> None:
    await rate_limit.consume_rate_limit(["k"], limit=2, window_seconds=60, detail="slow")
    await rate_limit.consume_rate_limit(["k"], limit=2, window_seconds=60, detail="slow")
    with pytest.raises(HTTPException) as exc:
        await rate_limit.consume_rate_limit(["k"], limit=2, window_seconds=60, detail="slow")

    assert exc.value.status_code == 429
    assert len(sessions.store) == 2
    # Every check-and-charge ran inside the key's advisory lock.
    assert [lock for session in sessions.sessions for lock in session.locks] == ["k"] * 3
    assert sum(session.rollbacks for session in sessions.sessions) == 1


async def test_refund_returns_one_charge(sessions: _SessionFactory) -> None:
    await rate_limit.consume_rate_limit(["k"], limit=5, window_seconds=60, detail="slow")
    await rate_limit.consume_rate_limit(["k"], limit=5, window_seconds=60, detail="slow")

    await rate_limit.refund_rate_limit(["k"], window_seconds=60)
    assert len(sessions.store) == 1
    await rate_limit.refund_rate_limit(["k"], window_seconds=60)
    assert sessions.store == []
    # Refunding nothing is a no-op, not an error.
    await rate_limit.refund_rate_limit(["k"], window_seconds=60)
    assert sessions.store == []


async def test_expired_attempts_do_not_count(sessions: _SessionFactory) -> None:
    stale = AuthRateLimitAttempt(key="k", attempted_at=datetime.now(UTC) - timedelta(seconds=600))
    stale.id = 1
    sessions.store.append(stale)

    await rate_limit.consume_rate_limit(["k"], limit=1, window_seconds=60, detail="slow")

    assert len(sessions.store) == 1


async def test_empty_keys_open_no_session(sessions: _SessionFactory) -> None:
    await rate_limit.consume_rate_limit([], limit=1, window_seconds=60, detail="slow")
    await rate_limit.refund_rate_limit([], window_seconds=60)

    assert sessions.sessions == []


async def test_throttle_refunds_the_account_charge_on_success(
    sessions: _SessionFactory,
) -> None:
    settings = get_auth_settings()
    digest = hashlib.sha256(b"victim@example.com").hexdigest()
    account_key = f"account:{digest}:/auth/login"

    async for _ in rate_limit.throttle_auth_attempts(_request()):
        pass

    assert sessions.store == []
    assert account_key in [lock for session in sessions.sessions for lock in session.locks]
    assert settings.auth_rate_limit_requests == 10


async def test_throttle_keeps_the_account_charge_on_failure(
    sessions: _SessionFactory,
) -> None:
    digest = hashlib.sha256(b"victim@example.com").hexdigest()
    account_key = f"account:{digest}:/auth/login"

    agen = rate_limit.throttle_auth_attempts(_request())
    await agen.__anext__()
    with pytest.raises(InvalidCredentialsError):
        await agen.athrow(InvalidCredentialsError("Invalid email or password"))

    assert [row.key for row in sessions.store] == [account_key]
