"""The two account emails carry the username, the app link, and no em dashes."""

from __future__ import annotations

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


def test_welcome_lists_the_first_steps() -> None:
    message = welcome_email("Ada", APP_URL)

    for step in (
        "Create a project",
        "Upload page images",
        "Run segmentation then transcription",
        "Correct the result in the editor",
        "Export to Word or PDF",
    ):
        assert step in message.html
        assert step in message.text


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

    assert "Reply to this email if you need help getting started." in message.html
    assert "Reply to this email if you need help getting started." in message.text
