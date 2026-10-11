"""Welcome and sign-in notice templates. Plain functions, no template engine.

Each builder returns an :class:`EmailMessage` with an HTML body and a plain-text
twin. The caller fills in ``to`` from the account address; the builders only
ever interpolate the display name. User-provided strings are HTML-escaped.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from backend.notifications.sender import EmailMessage

#: The app's primary accent (``--accent`` in ``styles/theme-shell.css``).
ACCENT = "#0d1f3c"

_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _shell(title: str, body_html: str) -> str:
    return (
        "<!DOCTYPE html>"
        '<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "</head>"
        '<body style="margin:0;padding:0;background-color:#f4f5f7;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background-color:#f4f5f7;padding:24px 12px;">'
        '<tr><td align="center">'
        '<table role="presentation" width="560" cellpadding="0" cellspacing="0" '
        'style="max-width:560px;width:100%;background-color:#ffffff;border-radius:8px;overflow:hidden;">'
        '<tr><td style="padding:32px 32px 8px 32px;font-family:'
        + _FONT_STACK
        + ';color:#1a1a1a;font-size:15px;line-height:1.6;">'
        + body_html
        + "</td></tr>"
        "</table>"
        "</td></tr></table></body></html>"
    )


def _button(label: str, url: str) -> str:
    return (
        f'<a href="{html.escape(url, quote=True)}" '
        'style="display:inline-block;background-color:'
        + ACCENT
        + ";color:#ffffff;font-family:"
        + _FONT_STACK
        + ";font-size:15px;font-weight:600;text-decoration:none;"
        'padding:12px 28px;border-radius:6px;margin:16px 0;">' + html.escape(label) + "</a>"
    )


def _footer(message: str) -> str:
    return (
        '<p style="margin:24px 0 0 0;font-size:12px;color:#6b7280;">'
        + html.escape(message)
        + "</p>"
    )


def welcome_email(username: str, app_url: str) -> EmailMessage:
    """First mail after registration. The caller sets ``to`` from the account address."""
    name = html.escape(username)
    body_html = (
        '<h1 style="margin:0 0 16px 0;font-size:22px;color:#1a1a1a;">'
        f"Welcome to Nomikos, {name}</h1>"
        '<p style="margin:0 0 12px 0;">'
        "Nomikos transcribes historical manuscripts in Greek, Armenian, Coptic and Syriac. "
        "Correct the results in the editor and retrain the models on your corrections."
        "</p>"
        + _button("Open Nomikos", app_url)
        + '<p style="margin:8px 0 8px 0;font-weight:600;">Get started</p>'
        '<ul style="margin:0 0 12px 0;padding-left:20px;">'
        "<li>Create a project</li>"
        "<li>Upload page images</li>"
        "<li>Run segmentation then transcription</li>"
        "<li>Correct the result in the editor</li>"
        "<li>Export to Word or PDF</li>"
        "</ul>"
        '<p style="margin:0;">Reply to this email if you need help getting started.</p>'
        + _footer("You received this because an account was created with this address.")
    )
    text = (
        f"Welcome to Nomikos, {username}\n\n"
        "Nomikos transcribes historical manuscripts in Greek, Armenian, Coptic and Syriac. "
        "Correct the results in the editor and retrain the models on your corrections.\n\n"
        f"Open Nomikos: {app_url}\n\n"
        "Get started:\n"
        "- Create a project\n"
        "- Upload page images\n"
        "- Run segmentation then transcription\n"
        "- Correct the result in the editor\n"
        "- Export to Word or PDF\n\n"
        "Reply to this email if you need help getting started.\n\n"
        "You received this because an account was created with this address."
    )
    return EmailMessage(
        subject="Welcome to Nomikos", to="", html=_shell("Welcome to Nomikos", body_html), text=text
    )


def signin_notice_email(username: str, when_utc: datetime, app_url: str) -> EmailMessage:
    """Notice after a login. The caller sets ``to`` from the account address."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=UTC)
    when = when_utc.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    name = html.escape(username)
    body_html = (
        '<h1 style="margin:0 0 16px 0;font-size:22px;color:#1a1a1a;">'
        "New sign-in to your Nomikos account</h1>"
        f'<p style="margin:0 0 12px 0;">Your account {name} was signed in on {when}.</p>'
        '<p style="margin:0;">If this was you, nothing to do. '
        "If not, change your password now.</p>"
        + _button("Open Nomikos", app_url)
        + _footer("You received this because sign-in notices are on for your account.")
    )
    text = (
        "New sign-in to your Nomikos account\n\n"
        f"Your account {username} was signed in on {when}.\n\n"
        "If this was you, nothing to do. If not, change your password now.\n\n"
        f"Open Nomikos: {app_url}\n\n"
        "You received this because sign-in notices are on for your account."
    )
    return EmailMessage(
        subject="New sign-in to your Nomikos account",
        to="",
        html=_shell("New sign-in to your Nomikos account", body_html),
        text=text,
    )
