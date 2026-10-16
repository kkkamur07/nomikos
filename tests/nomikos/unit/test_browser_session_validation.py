"""Session-cookie validation: every rejection path of ``_valid_session``."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from backend.core.settings import AuthSettings, reset_settings_caches
from backend.users.application.browser_sessions import BrowserSessionService
from backend.users.infrastructure.orm_models import AuthSession, User

STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


def _settings() -> AuthSettings:
    return AuthSettings(JWT_SECRET=STRONG_SECRET, _env_file=None)


class _FakeResult:
    def __init__(self, session: AuthSession | None) -> None:
        self._session = session

    def scalar_one_or_none(self) -> AuthSession | None:
        return self._session


class _FakeDB:
    def __init__(self, session: AuthSession | None) -> None:
        self._session = session
        self.commits = 0

    async def execute(self, stmt: object) -> _FakeResult:
        return _FakeResult(self._session)

    async def commit(self) -> None:
        self.commits += 1


def _live_session(settings: AuthSettings) -> tuple[AuthSession, str, BrowserSessionService]:
    user = User(id=uuid.uuid4(), email="u@example.org", username="u", hashed_password="x")
    session = AuthSession(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash="",
        csrf_token_hash="",
        expires_at=datetime.now(UTC) + timedelta(days=settings.session_expire_days),
    )
    service = BrowserSessionService(settings)
    tokens = service._issue(user, session)
    session.user = user
    return session, tokens.session_cookie, service


async def test_missing_or_malformed_cookie_is_not_a_session() -> None:
    reset_settings_caches()
    service = BrowserSessionService(_settings())

    assert await service._valid_session(_FakeDB(None), None) is None
    assert await service._valid_session(_FakeDB(None), "no-separator-here") is None
    assert await service._valid_session(_FakeDB(None), "not-a-uuid.secret") is None


async def test_unknown_session_is_not_valid() -> None:
    reset_settings_caches()
    service = BrowserSessionService(_settings())

    assert await service._valid_session(_FakeDB(None), f"{uuid.uuid4()}.secret") is None


async def test_revoked_or_expired_session_is_not_valid() -> None:
    settings = _settings()
    service = BrowserSessionService(settings)

    session, cookie, _ = _live_session(settings)
    session.revoked_at = datetime.now(UTC)
    assert await service._valid_session(_FakeDB(session), cookie) is None

    session, cookie, _ = _live_session(settings)
    session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert await service._valid_session(_FakeDB(session), cookie) is None


async def test_wrong_secret_burns_the_session() -> None:
    settings = _settings()
    service = BrowserSessionService(settings)
    session, cookie, _ = _live_session(settings)
    session_id = cookie.split(".", 1)[0]
    db = _FakeDB(session)

    assert await service._valid_session(db, f"{session_id}.wrong-secret") is None
    assert session.revoked_at is not None
    assert db.commits == 1


async def test_matching_secret_returns_the_session_without_writing() -> None:
    settings = _settings()
    service = BrowserSessionService(settings)
    session, cookie, _ = _live_session(settings)
    db = _FakeDB(session)

    assert await service._valid_session(db, cookie) is session
    assert db.commits == 0
