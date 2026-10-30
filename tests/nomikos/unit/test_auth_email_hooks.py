"""Register enqueues the welcome mail, login enqueues the sign-in notice.

The route functions are called directly with a stubbed service layer, so no
database is involved. The scheduled background tasks run against a recording
sender patched in after the route returns, which is exactly the seam
``send_in_background`` leaves open.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, Response

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-at-least-32-bytes")

from backend.core.settings import reset_settings_caches
from backend.notifications.sender import EmailMessage
from backend.users.api.schemas import LoginRequest, RegisterRequest
from backend.users.application.browser_sessions import BrowserSessionTokens

STRONG_SECRET = "xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV"


@pytest.fixture(autouse=True)
def platform_settings(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    # APP_PUBLIC_URL and EMAIL_REPLY_TO stay unset: these tests see the defaults.
    for name in ("APP_PUBLIC_URL", "EMAIL_REPLY_TO", "RESEND_API_KEY", "EMAIL_SIGNIN_NOTICES"):
        monkeypatch.delenv(name, raising=False)
    reset_settings_caches()
    yield
    reset_settings_caches()


@pytest.fixture
def auth_routes():
    from backend.users.api import auth

    return auth


USER_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")


def _user(username: str = "ada") -> SimpleNamespace:
    return SimpleNamespace(id=USER_ID, email="ada@example.com", username=username)


def _tokens() -> BrowserSessionTokens:
    return BrowserSessionTokens(access_token="a", session_cookie="s", csrf_token="c")


def _install_fakes(auth_routes, monkeypatch, *, login_user=None, register_user=None):
    class FakeAuth:
        async def register(self, db, **kwargs):
            return register_user

        async def login(self, db, **kwargs):
            return login_user

    class FakeSessions:
        def __init__(self, settings):
            pass

        async def create(self, db, user):
            return _tokens()

    monkeypatch.setattr(auth_routes, "_auth", FakeAuth())
    monkeypatch.setattr(auth_routes, "BrowserSessionService", FakeSessions)


class _RecordingSender:
    def __init__(self) -> None:
        self.sent: list[EmailMessage] = []

    async def send(self, message: EmailMessage) -> None:
        self.sent.append(message)


async def _run(tasks: BackgroundTasks, sender: _RecordingSender, monkeypatch) -> None:
    import backend.notifications.sender as sender_module

    monkeypatch.setattr(sender_module, "get_email_sender", lambda: sender)
    await tasks()


async def test_register_schedules_the_welcome_email(auth_routes, monkeypatch) -> None:
    _install_fakes(auth_routes, monkeypatch, register_user=_user())
    tasks = BackgroundTasks()

    await auth_routes.register(
        RegisterRequest(email="ada@example.com", username="ada", password="long-enough-pass"),
        None,
        object(),
        Response(),
        tasks,
    )

    assert len(tasks.tasks) == 1
    sender = _RecordingSender()
    await _run(tasks, sender, monkeypatch)
    assert len(sender.sent) == 1
    message = sender.sent[0]
    assert message.to == "ada@example.com"
    assert message.user_id == str(USER_ID)
    assert message.subject == "Welcome to Nomikos"
    assert 'href="https://app.nomikos.app"' in message.html
    assert 'src="https://app.nomikos.app/email/nomikos-logo.png"' in message.html


async def test_welcome_email_carries_no_username(auth_routes, monkeypatch) -> None:
    username = "friendX URGENT re-confirm at https://x.example"
    _install_fakes(auth_routes, monkeypatch, register_user=_user(username))
    tasks = BackgroundTasks()

    await auth_routes.register(
        RegisterRequest(email="ada@example.com", username="ada", password="long-enough-pass"),
        None,
        object(),
        Response(),
        tasks,
    )

    sender = _RecordingSender()
    await _run(tasks, sender, monkeypatch)
    message = sender.sent[0]
    for body in (message.html, message.text):
        assert "friendX" not in body
        assert "x.example" not in body


async def test_login_schedules_the_signin_notice(auth_routes, monkeypatch) -> None:
    _install_fakes(auth_routes, monkeypatch, login_user=_user())
    tasks = BackgroundTasks()

    await auth_routes.login(
        LoginRequest(email="ada@example.com", password="whatever"),
        None,
        object(),
        Response(),
        tasks,
    )

    assert len(tasks.tasks) == 1
    sender = _RecordingSender()
    await _run(tasks, sender, monkeypatch)
    assert len(sender.sent) == 1
    message = sender.sent[0]
    assert message.to == "ada@example.com"
    assert message.user_id == str(USER_ID)
    assert message.subject == "New sign-in to your Nomikos account"
    assert "If this was not you, contact the project owner." in message.text


async def test_login_notice_names_the_configured_reply_to(auth_routes, monkeypatch) -> None:
    monkeypatch.setenv("EMAIL_REPLY_TO", "hello@nomikos.app")
    reset_settings_caches()
    _install_fakes(auth_routes, monkeypatch, login_user=_user())
    tasks = BackgroundTasks()

    await auth_routes.login(
        LoginRequest(email="ada@example.com", password="whatever"),
        None,
        object(),
        Response(),
        tasks,
    )

    sender = _RecordingSender()
    await _run(tasks, sender, monkeypatch)
    assert "If this was not you, write to hello@nomikos.app." in sender.sent[0].text


async def test_login_schedules_nothing_when_notices_are_off(auth_routes, monkeypatch) -> None:
    monkeypatch.setenv("EMAIL_SIGNIN_NOTICES", "false")
    reset_settings_caches()
    _install_fakes(auth_routes, monkeypatch, login_user=_user())
    tasks = BackgroundTasks()

    await auth_routes.login(
        LoginRequest(email="ada@example.com", password="whatever"),
        None,
        object(),
        Response(),
        tasks,
    )

    assert tasks.tasks == []


async def test_register_passes_configured_reply_to(auth_routes, monkeypatch) -> None:
    monkeypatch.setenv("EMAIL_REPLY_TO", "hello@nomikos.app")
    reset_settings_caches()
    _install_fakes(auth_routes, monkeypatch, register_user=_user())
    tasks = BackgroundTasks()

    await auth_routes.register(
        RegisterRequest(email="ada@example.com", username="ada", password="long-enough-pass"),
        None,
        object(),
        Response(),
        tasks,
    )

    sender = _RecordingSender()
    await _run(tasks, sender, monkeypatch)
    assert "Just reply to this email or write to" in sender.sent[0].html
    assert "hello@nomikos.app" in sender.sent[0].html
