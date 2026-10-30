"""Shared placeholder-secret detection for settings validators."""

from __future__ import annotations

PLACEHOLDER_SECRET_VALUES = frozenset(
    {
        "change-me",
        "change-me-in-production",
        "replace-me",
        "replace-with-a-secret",
    }
)


def is_placeholder_secret(value: str | None) -> bool:
    """Whether ``value`` is empty or one of the shipped example secrets."""
    normalized = (value or "").strip().casefold()
    return (
        not normalized
        or normalized in PLACEHOLDER_SECRET_VALUES
        or normalized.startswith("replace-with-")
    )
