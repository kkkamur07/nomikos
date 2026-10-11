"""Resend sender posts the right JSON, raises on refusal, and never breaks a request."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi import BackgroundTasks

import backend.notifications.sender as sender_module
from backend.core.settings import reset_settings_caches
from backend.notifications.sender import (
    EmailMessage,
    EmailSendError,
    LoggingEmailSender,
    ResendEmailSender,
    get_email_sender,
    send_in_background,
)


def _message() -> EmailMessage:
    return EmailMessage(to="ada@example.com", subject="Hello", html="<p>Hello</p>", text="Hello")


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_resend_posts_expected_json_with_bearer_header() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"id": "mail_123"})

    client = httpx.AsyncClient(transport=_transport(handler))
    sender = ResendEmailSender(client, api_key="re_test_key", from_address="Nomikos <x@y.z>")

    await sender.send(_message())

    assert seen["url"] == "https://api.resend.com/emails"
    assert seen["auth"] == "Bearer re_test_key"
    assert seen["json"] == {
        "from": "Nomikos <x@y.z>",
        "to": ["ada@example.com"],
        "subject": "Hello",
        "html": "<p>Hello</p>",
        "text": "Hello",
    }


async def test_resend_raises_with_body_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = httpx.AsyncClient(transport=_transport(handler))
    sender = ResendEmailSender(client, api_key="re_test_key", from_address="Nomikos <x@y.z>")

    with pytest.raises(EmailSendError, match="500.*internal error"):
        await sender.send(_message())


async def test_resend_raises_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=_transport(handler))
    sender = ResendEmailSender(client, api_key="re_test_key", from_address="Nomikos <x@y.z>")

    with pytest.raises(EmailSendError):
        await sender.send(_message())


async def test_logging_sender_logs_subject_recipient_and_body(caplog) -> None:
    sender = LoggingEmailSender()

    with caplog.at_level(logging.DEBUG, logger="backend.notifications.sender"):
        await sender.send(_message())

    assert "Hello" in caplog.text
    assert "ada@example.com" in caplog.text


def test_factory_is_log_only_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    reset_settings_caches()
    try:
        assert isinstance(get_email_sender(), LoggingEmailSender)
    finally:
        reset_settings_caches()


def test_factory_builds_resend_sender_with_a_key(monkeypatch) -> None:
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    reset_settings_caches()
    try:
        assert isinstance(get_email_sender(), ResendEmailSender)
    finally:
        reset_settings_caches()


async def test_background_helper_swallows_failure_and_logs(caplog, monkeypatch) -> None:
    class Broken:
        async def send(self, message: EmailMessage) -> None:
            raise EmailSendError("Resend is down")

    tasks = BackgroundTasks()
    monkeypatch.setattr(sender_module, "get_email_sender", lambda: Broken())
    send_in_background(tasks, _message())
    with caplog.at_level(logging.ERROR, logger="backend.notifications.sender"):
        await tasks()

    assert "email_send_failed" in caplog.text


async def test_background_helper_delivers_on_success(monkeypatch) -> None:
    delivered: list[EmailMessage] = []

    class Recording:
        async def send(self, message: EmailMessage) -> None:
            delivered.append(message)

    tasks = BackgroundTasks()
    monkeypatch.setattr(sender_module, "get_email_sender", lambda: Recording())
    send_in_background(tasks, _message())
    await tasks()

    assert [m.subject for m in delivered] == ["Hello"]


async def test_resend_includes_reply_to_only_when_configured() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"id": "mail_123"})

    sender = ResendEmailSender(
        httpx.AsyncClient(transport=_transport(handler)),
        api_key="re_test_key",
        from_address="Nomikos <x@y.z>",
        reply_to="hello@nomikos.app",
    )
    await sender.send(_message())
    assert seen[0]["reply_to"] == ["hello@nomikos.app"]

    plain = ResendEmailSender(
        httpx.AsyncClient(transport=_transport(handler)),
        api_key="re_test_key",
        from_address="Nomikos <x@y.z>",
    )
    await plain.send(_message())
    assert "reply_to" not in seen[1]
