"""ORM to DTO mapping for helper devices."""

from __future__ import annotations

from datetime import datetime

from backend.ml.api.device_schemas import DeviceResponse, PairingRequestResponse
from backend.ml.application.pairing_service import DevicePairingService
from backend.ml.infrastructure.device_orm_models import HelperDevice, HelperPairing


def device_response_from_orm(
    device: HelperDevice, *, service: DevicePairingService, now: datetime
) -> DeviceResponse:
    return DeviceResponse(
        id=device.id,
        name=device.name,
        platform=device.platform,
        helper_version=device.helper_version,
        status=service.device_status(device, now=now),
        token_prefix=device.token_prefix,
        paired_at=device.created_at,
        paired_from_ip=device.paired_from_ip,
        last_seen_at=device.last_seen_at,
        last_seen_ip=device.last_seen_ip,
        token_expires_at=device.token_expires_at,
        revoked_at=device.revoked_at,
    )


def pairing_response_from_orm(
    pairing: HelperPairing, *, service: DevicePairingService
) -> PairingRequestResponse:
    """Consent-screen payload.

    No IP-derived field is returned. ``request_ip`` is still recorded on the row
    for support correlation, but it is the address the platform observed, which
    behind an unallowlisted proxy is the edge's - identical for every user. It
    must not be shown to a researcher as though it identified their computer.
    """
    return PairingRequestResponse(
        pairing_id=pairing.id,
        device_name=pairing.requested_name,
        platform=pairing.requested_platform,
        helper_version=pairing.requested_helper_version,
        confirmation_code=service.confirmation_code(pairing.id),
        requested_at=pairing.created_at,
        expires_at=pairing.expires_at,
    )
