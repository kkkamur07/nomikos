"""Settings parsing: one placeholder rule and split CORS validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.core.settings._env import REPO_ROOT
from backend.core.settings.app import AppSettings, _check_origin, _split_origins
from backend.core.settings.secrets import is_placeholder_secret


def test_placeholder_rule_catches_every_shipped_example() -> None:
    for raw in (
        "change-me",
        "change-me-in-production",
        "replace-me",
        "replace-with-a-secret",
        "  Replace-With-X  ",
        "",
        "   ",
        None,
    ):
        assert is_placeholder_secret(raw), raw


def test_placeholder_rule_passes_real_secrets() -> None:
    assert not is_placeholder_secret("xQ7v2Kd9RmZ4pB6wLt1yHs3nCf8jUa5eG0oV")
    assert not is_placeholder_secret("correct horse battery staple")


def test_split_origins_trims_and_dedupes() -> None:
    assert _split_origins("http://a:3000/, http://a:3000 ,https://b/") == [
        "http://a:3000",
        "http://a:3000",
        "https://b",
    ]
    assert _split_origins("  , ") == []


def test_check_origin_accepts_explicit_http_origins() -> None:
    assert _check_origin("http://localhost:3000") == "http://localhost:3000"
    assert _check_origin("https://app.nomikos.app") == "https://app.nomikos.app"


def test_check_origin_rejects_everything_else() -> None:
    for raw in (
        "ftp://x",
        "http://",
        "http://user@x",
        "http://x/path",
        "http://x?q=1",
        "http://x#frag",
        "http://*",
        "not-a-url",
    ):
        with pytest.raises(ValueError):
            _check_origin(raw)


def test_cors_settings_round_trip() -> None:
    settings = AppSettings(
        CORS_ORIGINS="http://localhost:3000/,http://localhost:3000,https://b/",
        MEDIA_ROOT=str(REPO_ROOT / "backend" / "media"),
    )
    assert settings.cors_origin_list == ["http://localhost:3000", "https://b"]

    with pytest.raises(ValidationError):
        AppSettings(CORS_ORIGINS="   ", MEDIA_ROOT=str(REPO_ROOT / "backend" / "media"))
    with pytest.raises(ValidationError):
        AppSettings(CORS_ORIGINS="http://x/path", MEDIA_ROOT=str(REPO_ROOT / "backend" / "media"))
