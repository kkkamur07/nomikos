"""The two account emails: app link, logo, wording, and no em dashes."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from backend.core.settings import reset_settings_caches
from backend.core.settings.email import get_email_settings
from backend.notifications.templates import signin_notice_email, welcome_email

APP_URL = "https://app.nomikos.app"
WHEN = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)


def test_welcome_links_the_app_and_avoids_em_dashes() -> None:
    message = welcome_email(APP_URL)

    assert message.subject == "Welcome to Nomikos"
    assert APP_URL in message.html
    assert APP_URL in message.text
    assert "Open Nomikos" in message.html
    for field in (message.subject, message.html, message.text):
        assert "\u2014" not in field


def test_welcome_greets_without_a_username() -> None:
    message = welcome_email(APP_URL)

    assert ">Welcome to Nomikos</h1>" in message.html
    assert message.text.startswith("Welcome to Nomikos\n\n")


def test_welcome_describes_the_first_steps_in_short_sentences() -> None:
    message = welcome_email(APP_URL)

    for sentence in (
        "Nomikos transcribes historical manuscripts in Greek, Armenian, Coptic and Syriac.",
        "Create a project, upload page images, then run segmentation and transcription.",
        "Correct the result in the editor and export to Word or PDF.",
    ):
        assert sentence in message.html
        assert sentence in message.text


def test_signin_notice_names_user_time_and_avoids_em_dashes() -> None:
    message = signin_notice_email("Ada", WHEN, APP_URL)

    assert message.subject == "New sign-in to your Nomikos account"
    assert "Ada" in message.html
    assert "2026-10-11 09:30 UTC" in message.html
    assert "2026-10-11 09:30 UTC" in message.text
    for field in (message.subject, message.html, message.text):
        assert "\u2014" not in field


def test_signin_notice_is_information_only() -> None:
    for message in (
        signin_notice_email("Ada", WHEN, APP_URL),
        signin_notice_email("Ada", WHEN, APP_URL, reply_to="hello@nomikos.app"),
    ):
        for body in (message.html, message.text):
            assert "change your password" not in body
            assert "Review activity" not in body
            assert "sign-in notices are on for your account" not in body
        assert "<a " not in message.html.replace('<a class="email-link" href="mailto:', "")


def test_signin_notice_points_at_reply_to_or_the_project_owner() -> None:
    with_reply = signin_notice_email("Ada", WHEN, APP_URL, reply_to="hello@nomikos.app")
    assert "If this was not you, write to hello@nomikos.app." in with_reply.text
    assert 'href="mailto:hello@nomikos.app"' in with_reply.html
    assert "If this was not you, write to" in with_reply.html

    without = signin_notice_email("Ada", WHEN, APP_URL)
    assert "If this was not you, contact the project owner." in without.text
    assert "If this was not you, contact the project owner." in without.html
    assert "mailto:" not in without.html


def test_templates_escape_user_input() -> None:
    notice = signin_notice_email("<b>Ada</b>", datetime(2026, 10, 11, tzinfo=UTC), APP_URL)
    assert "<b>Ada</b>" not in notice.html
    assert "&lt;b&gt;Ada&lt;/b&gt;" in notice.html

    welcome = welcome_email(APP_URL, reply_to='"><script>x</script>@example.com')
    assert "<script>" not in welcome.html


def test_welcome_omits_the_reply_invite_without_reply_to() -> None:
    message = welcome_email(APP_URL)

    assert "reply to this email" not in message.html
    assert "reply to this email" not in message.text


def test_welcome_invites_a_reply_with_reply_to() -> None:
    message = welcome_email(APP_URL, reply_to="hello@nomikos.app")

    assert "Just reply to this email or write to" in message.html
    assert "hello@nomikos.app" in message.html
    assert "Just reply to this email or write to hello@nomikos.app." in message.text


def test_emails_carry_wordmark_button_link_and_plain_text_twin() -> None:
    welcome = welcome_email(APP_URL, reply_to="hello@nomikos.app")
    notice = signin_notice_email("Ada", WHEN, APP_URL)

    assert ">Nomikos</p>" in welcome.html
    assert ">Nomikos</p>" in notice.html

    assert "Open Nomikos" in welcome.html
    assert f'href="{APP_URL}"' in welcome.html
    assert f"Open Nomikos: {APP_URL}" in welcome.text

    assert 'href="mailto:hello@nomikos.app"' in welcome.html
    assert "hello@nomikos.app" in welcome.text


def test_emails_load_no_outside_resources() -> None:
    welcome = welcome_email(APP_URL, reply_to="hello@nomikos.app")
    notice = signin_notice_email("Ada", WHEN, APP_URL)

    for message in (welcome, notice):
        urls = re.findall(r"https?://[^\s\"'<>]+", message.html)
        assert urls, "expected at least the logo"
        for url in urls:
            assert url.startswith(APP_URL), url
        images = re.findall(r"<img[^>]*>", message.html)
        assert len(images) == 1, images
        assert f'src="{APP_URL}/email/nomikos-logo.png"' in images[0]
        assert 'alt="Nomikos"' in images[0]
        assert "@import" not in message.html
        assert "url(" not in message.html


def test_emails_declare_dark_mode_support() -> None:
    welcome = welcome_email(APP_URL)
    notice = signin_notice_email("Ada", WHEN, APP_URL)

    for message in (welcome, notice):
        assert 'name="color-scheme"' in message.html
        assert "prefers-color-scheme" in message.html


def test_dark_mode_puts_the_logo_on_a_white_rounded_tile() -> None:
    html = signin_notice_email("Ada", WHEN, APP_URL).html
    dark = html.split("@media (prefers-color-scheme: dark){", 1)[1].split("}@media", 1)[0]
    logo_rule = re.search(r"\.email-logo\{([^}]*)\}", dark)
    assert logo_rule is not None
    assert "background-color:#ffffff" in logo_rule.group(1)
    assert "border-radius:" in logo_rule.group(1)
    # Light mode is unchanged: the cell stays white with no extra padding inline.
    assert 'class="email-logo" bgcolor="#ffffff" style="background-color:#ffffff;color:' in html


@pytest.fixture
def default_email_settings(monkeypatch):
    for name in ("APP_PUBLIC_URL", "EMAIL_REPLY_TO", "RESEND_API_KEY", "EMAIL_SIGNIN_NOTICES"):
        monkeypatch.delenv(name, raising=False)
    reset_settings_caches()
    yield get_email_settings()
    reset_settings_caches()


def test_default_settings_point_logo_and_button_at_the_app(default_email_settings) -> None:
    app_url = default_email_settings.app_public_url
    notice = signin_notice_email("Ada", WHEN, app_url)
    assert 'src="https://app.nomikos.app/email/nomikos-logo.png"' in notice.html

    welcome = welcome_email(app_url, reply_to=default_email_settings.reply_to)
    assert 'src="https://app.nomikos.app/email/nomikos-logo.png"' in welcome.html
    assert 'href="https://app.nomikos.app"' in welcome.html
    assert "Open Nomikos: https://app.nomikos.app\n" in welcome.text
