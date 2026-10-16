"""Device pairing API DTOs.

No response model on this surface carries ``token_hash``,
``device_code_hash``, or ``verification_token_hash``. The single field that ever
carries raw secret material is ``PairingTokenResponse.device_token``, populated
exactly once, at redemption.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from backend.ml.domain.devices import DeviceStatus, PairingStatus

#: A helper advertises string tags like ``{"runtime": "torch"}``; the consent
#: screen renders them as inert text. Bounded in count and length so an
#: unauthenticated caller cannot store megabytes of JSON per pairing row.
MAX_CAPABILITY_ENTRIES = 32
MAX_CAPABILITY_KEY_LENGTH = 64
MAX_CAPABILITY_VALUE_LENGTH = 512


def bound_capabilities(value: dict[str, str]) -> dict[str, str]:
    """Reject capability maps that exceed the count or length ceilings."""
    if len(value) > MAX_CAPABILITY_ENTRIES:
        raise ValueError(f"at most {MAX_CAPABILITY_ENTRIES} capabilities")
    for key, item in value.items():
        if len(key) > MAX_CAPABILITY_KEY_LENGTH:
            raise ValueError("capability names must be short")
        if len(item) > MAX_CAPABILITY_VALUE_LENGTH:
            raise ValueError("capability values must be short")
    return value


class PairingStartRequest(BaseModel):
    """Sent by an unpaired helper. Every field is attacker-controlled."""

    device_name: str = Field(min_length=1, max_length=120)
    platform: str = Field(min_length=1, max_length=32)
    helper_version: str = Field(min_length=1, max_length=32)
    capabilities: dict[str, str] = Field(default_factory=dict)

    @field_validator("capabilities")
    @classmethod
    def _bound_capabilities(cls, value: dict[str, str]) -> dict[str, str]:
        return bound_capabilities(value)


class PairingStartResponse(BaseModel):
    pairing_id: UUID
    device_code: str
    verification_url: str
    confirmation_code: str
    """Non-secret. The helper MUST show or log this so the researcher can compare
    it with the code on the consent screen."""
    expires_in: int
    interval_seconds: int


class PairingTokenRequest(BaseModel):
    pairing_id: UUID
    device_code: str = Field(min_length=1, max_length=256)


class PairingTokenResponse(BaseModel):
    """Always returned with HTTP 200 - the status is the payload.

    The platform error envelope replaces ``HTTPException.detail`` with a fixed
    public string, so a machine-readable protocol state cannot survive a non-2xx
    response.
    """

    status: PairingStatus
    interval_seconds: int
    device_id: UUID | None = None
    device_token: str | None = None
    token_expires_at: datetime | None = None
    account_email: str | None = None


class PairingLookupRequest(BaseModel):
    """The fragment token travels in a POST body, never a path or query."""

    verification_token: str = Field(min_length=1, max_length=256)


class PairingConsentRequest(BaseModel):
    verification_token: str = Field(min_length=1, max_length=256)


class PairingRequestResponse(BaseModel):
    """Consent-screen data. Rendered as inert text under fixed labels.

    Carries no network-derived signal. ``same_network`` and ``request_ip`` were
    removed: behind a proxy the platform does not allowlist, the observed address
    is the edge's, so ``same_network`` was unconditionally true and ``request_ip``
    was the same string for every request on the platform. A reassurance that is
    always shown is worse than none, because the screen presents it as evidence.
    """

    pairing_id: UUID
    device_name: str
    platform: str
    helper_version: str
    confirmation_code: str
    """Must be displayed prominently. The researcher compares it against the code
    their own helper shows; a mismatch means this request came from elsewhere."""
    requested_at: datetime
    expires_at: datetime


class DeviceResponse(BaseModel):
    id: UUID
    name: str
    platform: str
    helper_version: str
    status: DeviceStatus
    token_prefix: str
    paired_at: datetime
    paired_from_ip: str | None = None
    last_seen_at: datetime | None = None
    last_seen_ip: str | None = None
    token_expires_at: datetime | None = None
    revoked_at: datetime | None = None


class DeviceSelfResponse(BaseModel):
    """What a paired helper is allowed to learn about itself."""

    device_id: UUID
    name: str
    account_email: str
    token_expires_at: datetime | None = None
    server_time: datetime


class DeviceTokenRenewResponse(BaseModel):
    device_token: str
    token_expires_at: datetime
