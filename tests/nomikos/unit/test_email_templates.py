"""The two account emails carry the username, the app link, and no em dashes."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from backend.notifications.templates import signin_notice_email, welcome_email

APP_URL = "https://app.nomikos.app"


def test_welcome_names_the_user_links_the_app_and_avoids_em_dashes() -> None:
    message = welcome_email("Ada", APP_URL)

    assert message.subject == "Welcome to Nomikos"
    assert "Ada" in message.html
    assert "Ada" in message.text
    assert APP_URL in message.html
    assert APP_URL in message.text
    assert "Open Nomikos" in message.html
    for field in (message.subject, message.html, message.text):
        assert "\u2014" not in field


def test_welcome_describes_the_first_steps_in_short_sentences() -> None:
    message = welcome_email("Ada", APP_URL)

    for sentence in (
        "Nomikos transcribes historical manuscripts in Greek, Armenian, Coptic and Syriac.",
        "Create a project, upload page images, then run segmentation and transcription.",
        "Correct the result in the editor and export to Word or PDF.",
    ):
        assert sentence in message.html
        assert sentence in message.text


def test_signin_notice_names_user_time_and_avoids_em_dashes() -> None:
    when = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)
    message = signin_notice_email("Ada", when, APP_URL)

    assert message.subject == "New sign-in to your Nomikos account"
    assert "Ada" in message.html
    assert "2026-10-11 09:30 UTC" in message.html
    assert "2026-10-11 09:30 UTC" in message.text
    assert APP_URL in message.html
    assert "change your password now" in message.html
    for field in (message.subject, message.html, message.text):
        assert "\u2014" not in field


def test_templates_escape_user_input() -> None:
    message = welcome_email("<script>alert(1)</script>", APP_URL)

    assert "<script>" not in message.html
    assert "&lt;script&gt;" in message.html

    notice = signin_notice_email("<b>Ada</b>", datetime(2026, 10, 11, tzinfo=UTC), APP_URL)
    assert "<b>Ada</b>" not in notice.html
    assert "&lt;b&gt;Ada&lt;/b&gt;" in notice.html


def test_welcome_omits_the_reply_invite_without_reply_to() -> None:
    message = welcome_email("Ada", APP_URL)

    assert "Reply to this email" not in message.html
    assert "Reply to this email" not in message.text


def test_welcome_invites_a_reply_with_reply_to() -> None:
    message = welcome_email("Ada", APP_URL, reply_to="hello@nomikos.app")

    assert "Just reply to this email or write to" in message.html
    assert "hello@nomikos.app" in message.html
    assert "Just reply to this email or write to hello@nomikos.app." in message.text


def test_emails_carry_wordmark_button_link_and_plain_text_twin() -> None:
    welcome = welcome_email("Ada", APP_URL, reply_to="hello@nomikos.app")
    when = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)
    notice = signin_notice_email("Ada", when, APP_URL)

    assert ">Nomikos</p>" in welcome.html
    assert ">Nomikos</p>" in notice.html

    assert "Open Nomikos" in welcome.html
    assert f'href="{APP_URL}"' in welcome.html
    assert f"Open Nomikos: {APP_URL}" in welcome.text

    assert "Review activity" in notice.html
    assert f'href="{APP_URL}"' in notice.html
    assert f"Review activity: {APP_URL}" in notice.text

    assert 'href="mailto:hello@nomikos.app"' in welcome.html
    assert "hello@nomikos.app" in welcome.text


def test_emails_load_no_outside_resources() -> None:
    welcome = welcome_email("Ada", APP_URL, reply_to="hello@nomikos.app")
    when = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)
    notice = signin_notice_email("Ada", when, APP_URL)

    for message in (welcome, notice):
        urls = re.findall(r"https?://[^\s\"'<>]+", message.html)
        assert urls, "expected at least the button link"
        for url in urls:
            assert url.startswith(APP_URL), url
        images = re.findall(r"<img[^>]*>", message.html)
        assert len(images) == 1, images
        assert f'src="{APP_URL}/email/nomikos-logo.png"' in images[0]
        assert 'alt="Nomikos"' in images[0]
        assert "@import" not in message.html
        assert "url(" not in message.html


def test_emails_declare_dark_mode_support() -> None:
    welcome = welcome_email("Ada", APP_URL)
    when = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)
    notice = signin_notice_email("Ada", when, APP_URL)

    for message in (welcome, notice):
        assert 'name="color-scheme"' in message.html
        assert "prefers-color-scheme" in message.html
