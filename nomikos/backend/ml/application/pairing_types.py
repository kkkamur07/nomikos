"""Pairing result types shared by the pairing service and its API layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from backend.ml.domain.devices import PairingStatus


@dataclass(frozen=True)
class StartedPairing:
    """Returned once, to the helper that created the pairing."""

    pairing_id: UUID
    device_code: str
    verification_url: str
    confirmation_code: str
    expires_in: int
    interval_seconds: int


@dataclass(frozen=True)
class PairingPollResult:
    """Poll outcome. ``device_token`` is populated exactly once, ever."""

    status: PairingStatus
    interval_seconds: int
    device_id: UUID | None = None
    device_token: str | None = None
    token_expires_at: datetime | None = None
    account_email: str | None = None
