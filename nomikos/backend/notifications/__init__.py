"""Transactional email: templates, senders, and background delivery."""

from backend.notifications.sender import (
    EmailMessage,
    EmailSender,
    EmailSendError,
    LoggingEmailSender,
    ResendEmailSender,
    get_email_sender,
    send_in_background,
)
from backend.notifications.templates import signin_notice_email, welcome_email

__all__ = [
    "EmailMessage",
    "EmailSender",
    "EmailSendError",
    "LoggingEmailSender",
    "ResendEmailSender",
    "get_email_sender",
    "send_in_background",
    "signin_notice_email",
    "welcome_email",
]
