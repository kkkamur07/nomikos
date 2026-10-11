"""Email senders: Resend over HTTP, logging fallback, background delivery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
from fastapi import BackgroundTasks

from backend.core.settings._cache import settings_cache
from backend.core.settings.email import get_email_settings

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"
RESEND_TIMEOUT_SECONDS = 10.0


@dataclass
class EmailMessage:
    """One outgoing mail. Templates leave ``to`` empty; the caller sets it from the account."""

    subject: str
    html: str
    text: str
    to: str = ""


class EmailSendError(RuntimeError):
    """Resend received the message and refused it, or never answered."""


class EmailSender(Protocol):
    async def send(self, message: EmailMessage) -> None: ...


class ResendEmailSender:
    """One POST to the Resend HTTP API per message. Raises on non-2xx."""

    def __init__(self, client: httpx.AsyncClient, api_key: str, from_address: str) -> None:
        self._client = client
        self._api_key = api_key
        self._from_address = from_address

    async def send(self, message: EmailMessage) -> None:
        try:
            response = await self._client.post(
                RESEND_API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "from": self._from_address,
                    "to": [message.to],
                    "subject": message.subject,
                    "html": message.html,
                    "text": message.text,
                },
                timeout=RESEND_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise EmailSendError(f"Resend request failed: {exc}") from exc
        if not 200 <= response.status_code < 300:
            raise EmailSendError(
                f"Resend rejected the message: HTTP {response.status_code}: {response.text}"
            )


class LoggingEmailSender:
    """The no-key sender. Visible in logs and in the dev stack, sends nothing."""

    async def send(self, message: EmailMessage) -> None:
        logger.info("email subject=%r to=%r", message.subject, message.to)
        logger.debug("email body to=%r: %s", message.to, message.text)


@settings_cache
def get_email_sender() -> EmailSender:
    """The process-wide sender, chosen by settings. No key means log-only."""
    settings = get_email_settings()
    if settings.enabled and settings.resend_api_key is not None:
        return ResendEmailSender(
            httpx.AsyncClient(),
            api_key=settings.resend_api_key,
            from_address=settings.email_from,
        )
    return LoggingEmailSender()


async def _deliver(message: EmailMessage) -> None:
    """Send one message, turning any failure into a log line. Never raises."""
    try:
        await get_email_sender().send(message)
    except Exception:
        logger.exception("email_send_failed subject=%r to=%r", message.subject, message.to)


def send_in_background(background_tasks: BackgroundTasks, message: EmailMessage) -> None:
    """Enqueue delivery after the response. Register and login succeed even if Resend is down."""
    background_tasks.add_task(_deliver, message)
