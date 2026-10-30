"""Device display helpers: clock, text normalisation, and status derivation."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime

from backend.core.settings.device import DeviceSettings
from backend.ml.domain.devices import DeviceStatus
from backend.ml.infrastructure.device_orm_models import HelperDevice


def _now() -> datetime:
    return datetime.now(UTC)


def _clean_text(value: str, *, limit: int, fallback: str = "unknown") -> str:
    """Normalise a helper-supplied string before it is stored.

    These strings are attacker-controlled: anyone can POST a pairing request.
    They are rendered on the consent screen, so control characters and
    over-long values are stripped here, and the UI renders them as inert text
    under fixed labels.
    """
    normalized = unicodedata.normalize("NFC", value or "")
    stripped = "".join(
        char for char in normalized if char.isprintable() and unicodedata.category(char) != "Cf"
    ).strip()
    return stripped[:limit] or fallback


def _device_status(
    device: HelperDevice, *, settings: DeviceSettings, now: datetime
) -> DeviceStatus:
    if device.revoked_at is not None:
        return DeviceStatus.revoked
    if not device.token_hash:
        return DeviceStatus.pairing
    if device.last_seen_at is None:
        return DeviceStatus.offline
    age = (now - device.last_seen_at).total_seconds()
    if age <= settings.device_online_window_seconds:
        return DeviceStatus.online
    if age <= settings.device_idle_window_seconds:
        return DeviceStatus.idle
    return DeviceStatus.offline
