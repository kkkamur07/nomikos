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

#: Very light warm grey page background (``--bg`` in ``styles/theme-shell.css``).
_BG = "#faf9f7"
_CARD = "#ffffff"
_INK = "#1a1a1a"
_BODY = "#44403c"
_MUTED = "#78716c"

_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _shell(title: str, body_html: str, preheader: str = "") -> str:
    """Centred 560 px white card on a warm grey background.

    Every cell carries explicit background and text colours so dark-mode
    clients that invert unstyled mail keep this readable; clients that honour
    ``prefers-color-scheme`` get tuned overrides from the style block.
    """
    preview = ""
    if preheader:
        preview = (
            '<div style="display:none;max-height:0;overflow:hidden;opacity:0;'
            f'color:transparent;">{html.escape(preheader)}</div>'
        )
    return (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<meta name="color-scheme" content="light dark">'
        '<meta name="supported-color-schemes" content="light dark">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "@media (prefers-color-scheme: dark){"
        ".email-bg{background-color:#1c1917 !important;}"
        ".email-card{background-color:#292524 !important;}"
        ".email-logo{background-color:#292524 !important;}"
        ".email-pad{background-color:#292524 !important;}"
        ".email-title{color:#fafaf9 !important;}"
        ".email-body{color:#e7e5e4 !important;}"
        ".email-muted{color:#a8a29e !important;}"
        ".email-wordmark{color:#e7e5e4 !important;}"
        ".email-link{color:#e7e5e4 !important;}"
        "}"
        "@media only screen and (max-width:600px){"
        ".email-pad{padding-left:24px !important;padding-right:24px !important;}"
        "}"
        "</style>"
        "</head>"
        f'<body style="margin:0;padding:0;background-color:{_BG};color:{_INK};">'
        + preview
        + f'<table class="email-bg" role="presentation" width="100%" cellpadding="0"'
        f' cellspacing="0" border="0" bgcolor="{_BG}"'
        f' style="background-color:{_BG};color:{_INK};">'
        f'<tr><td align="center" bgcolor="{_BG}"'
        f' style="padding:32px 12px;background-color:{_BG};color:{_INK};">'
        '<table class="email-card" role="presentation" width="560" cellpadding="0"'
        f' cellspacing="0" border="0" bgcolor="{_CARD}"'
        ' style="max-width:560px;width:100%;background-color:' + _CARD + ';border-radius:8px;">'
        '<tr><td class="email-pad"'
        f' style="padding:40px 36px 32px 36px;font-family:{_FONT_STACK};'
        f"color:{_BODY};font-size:15px;line-height:1.6;background-color:{_CARD};"
        f'border-radius:8px;" bgcolor="{_CARD}">' + body_html + "</td></tr>"
        "</table>"
        "</td></tr></table></body></html>"
    )


def _logo(app_url: str) -> str:
    """Centred brand mark above the wordmark. The PNG is hosted, not attached:
    email clients do not render SVG and strip inline images."""
    src = html.escape(f"{app_url}/email/nomikos-logo.png", quote=True)
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0"'
        ' align="center" style="margin:0 auto 16px auto;">'
        "<tr>"
        f'<td align="center" class="email-logo" bgcolor="{_CARD}"'
        f' style="background-color:{_CARD};color:{_INK};">'
        f'<img src="{src}" width="44" height="64" alt="Nomikos"'
        ' style="display:block;border:0;outline:none;">'
        "</td>"
        "</tr>"
        "</table>"
    )


def _wordmark() -> str:
    return (
        '<p class="email-wordmark" style="margin:0 0 20px 0;text-align:center;'
        f"font-family:{_FONT_STACK};font-size:12px;font-weight:700;"
        f"letter-spacing:3px;text-transform:uppercase;color:{ACCENT};"
        '">Nomikos</p>'
    )


def _title(text_html: str) -> str:
    return (
        '<h1 class="email-title" style="margin:0 0 16px 0;text-align:center;'
        f"font-family:{_FONT_STACK};font-size:22px;line-height:1.3;"
        f"font-weight:700;color:{_INK};"
        '">' + text_html + "</h1>"
    )


def _paragraph(text_html: str) -> str:
    return (
        '<p class="email-body" style="margin:0 0 12px 0;'
        f"font-family:{_FONT_STACK};font-size:15px;line-height:1.6;"
        f"color:{_BODY};"
        '">' + text_html + "</p>"
    )


def _button(label: str, url: str) -> str:
    """Bulletproof button: a table cell with bgcolor plus a padded link.

    Gmail and Outlook ignore flex and strip padding on anchors in places,
    so the colour sits on the cell and the link carries matching padding.
    """
    safe_url = html.escape(url, quote=True)
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0"'
        ' align="center" style="margin:24px auto 8px auto;">'
        "<tr>"
        f'<td align="center" bgcolor="{ACCENT}"'
        f' style="border-radius:6px;background-color:{ACCENT};color:#ffffff;">'
        f'<a href="{safe_url}" target="_blank"'
        f' style="display:inline-block;padding:12px 28px;font-family:{_FONT_STACK};'
        "font-size:15px;font-weight:600;color:#ffffff;"
        'text-decoration:none;border-radius:6px;">' + html.escape(label) + "</a>"
        "</td>"
        "</tr>"
        "</table>"
    )


def _footer(lines_html: list[str]) -> str:
    joined = "<br>".join(lines_html)
    return (
        '<p class="email-muted" style="margin:24px 0 0 0;text-align:center;'
        f"font-family:{_FONT_STACK};font-size:12px;line-height:1.6;"
        f"color:{_MUTED};"
        '">' + joined + "</p>"
    )


def welcome_email(username: str, app_url: str, *, reply_to: str | None = None) -> EmailMessage:
    """First mail after registration. The caller sets ``to`` from the account address.

    The reply invitation appears only when ``reply_to`` names a mailbox that
    reads replies; otherwise the mail invites none.
    """
    name = html.escape(username)
    footer_lines = ["You received this because an account was created with this address."]
    if reply_to is not None:
        safe_reply = html.escape(reply_to, quote=True)
        footer_lines.insert(
            0,
            "Questions? Just reply to this email or write to "
            f'<a class="email-link" href="mailto:{safe_reply}" '
            f'style="color:{ACCENT};text-decoration:underline;">'
            f"{html.escape(reply_to)}</a>.",
        )
        reply_text = f"Questions? Just reply to this email or write to {reply_to}.\n\n"
    else:
        reply_text = ""
    body_html = (
        _logo(app_url)
        + _wordmark()
        + _title(f"Welcome to Nomikos, {name}")
        + _paragraph(
            "Nomikos transcribes historical manuscripts in Greek, Armenian, Coptic and Syriac."
        )
        + _paragraph(
            "Create a project, upload page images, then run segmentation and transcription."
        )
        + _paragraph("Correct the result in the editor and export to Word or PDF.")
        + _button("Open Nomikos", app_url)
        + _footer(footer_lines)
    )
    text = (
        f"Welcome to Nomikos, {username}\n\n"
        "Nomikos transcribes historical manuscripts in Greek, Armenian, "
        "Coptic and Syriac.\n\n"
        "Create a project, upload page images, then run segmentation "
        "and transcription.\n\n"
        "Correct the result in the editor and export to Word or PDF.\n\n"
        f"Open Nomikos: {app_url}\n\n"
        + reply_text
        + "You received this because an account was created with this address."
    )
    return EmailMessage(
        subject="Welcome to Nomikos",
        to="",
        html=_shell("Welcome to Nomikos", body_html, "Get started with Nomikos in minutes."),
        text=text,
    )


def signin_notice_email(username: str, when_utc: datetime, app_url: str) -> EmailMessage:
    """Notice after a login. The caller sets ``to`` from the account address."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=UTC)
    when = when_utc.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    name = html.escape(username)
    body_html = (
        _logo(app_url)
        + _wordmark()
        + _title("New sign-in to your Nomikos account")
        + _paragraph(f"Hi {name}, your account was signed in on {when}.")
        + _paragraph("If this was you, nothing to do. If not, change your password now.")
        + _button("Review activity", app_url)
        + _footer(["You received this because sign-in notices are on for your account."])
    )
    text = (
        "New sign-in to your Nomikos account\n\n"
        f"Hi {username}, your account was signed in on {when}.\n\n"
        "If this was you, nothing to do. If not, change your password now.\n\n"
        f"Review activity: {app_url}\n\n"
        "You received this because sign-in notices are on for your account."
    )
    return EmailMessage(
        subject="New sign-in to your Nomikos account",
        to="",
        html=_shell(
            "New sign-in to your Nomikos account",
            body_html,
            "A new sign-in was recorded on your account.",
        ),
        text=text,
    )
