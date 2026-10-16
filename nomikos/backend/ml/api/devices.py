"""Browser-side device management: consent, listing, and revocation.

Every route here is guarded by ``get_current_user`` (Bearer JWT). No CSRF header
is required: CSRF is enforced only inside ``BrowserSessionService._require_csrf``
for the cookie-authenticated ``/auth/refresh`` and ``/auth/logout`` routes. A
Bearer JWT is not ambient credential material, so a cross-site page cannot forge
one of these calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ml.api.device_dependencies import require_device_pairing_enabled
from backend.ml.api.device_responses import device_response_from_orm, pairing_response_from_orm
from backend.ml.api.device_schemas import (
    DeviceResponse,
    PairingConsentRequest,
    PairingLookupRequest,
    PairingRequestResponse,
)
from backend.ml.application.device_management import DeviceManagementService
from backend.ml.application.pairing_service import DevicePairingService
from backend.users.api.dependencies import get_current_user
from backend.users.api.rate_limit import client_ip_for_request
from backend.users.infrastructure.orm_models import User
from infrastructure.db import get_db

router = APIRouter(tags=["devices"], dependencies=[Depends(require_device_pairing_enabled)])
_pairings = DevicePairingService()
_devices = DeviceManagementService()


@router.post("/devices/pairings/lookup", response_model=PairingRequestResponse)
async def lookup_device_pairing(
    body: PairingLookupRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PairingRequestResponse:
    """Resolve the fragment token into the consent screen's contents.

    Unknown, expired, consumed, and denied all return one indistinguishable 404.

    There is deliberately no companion "list my live pairing requests" route.
    The one that existed filtered solely on the observed client address - a
    pairing has no owner before consent, so there was no user to filter on - and
    behind a proxy the platform does not allowlist that address is the same for
    everybody, which made it a list of *every* user's pending pairing requests,
    ``pairing_id`` included. Recovering a closed consent tab now means starting a
    new pairing from the helper, which costs one click.
    """
    pairing = await _pairings.lookup_pairing(db, verification_token=body.verification_token)
    return pairing_response_from_orm(pairing, service=_pairings)


@router.post("/devices/pairings/{pairing_id}/approve", response_model=DeviceResponse)
async def approve_device_pairing(
    pairing_id: UUID,
    body: PairingConsentRequest,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeviceResponse:
    """Grant this computer permission to run the caller's jobs.

    The click is the entire anti-phishing control, so this must never be reached
    from page load. ``verification_token`` is re-verified server-side: possession
    of the fragment, not knowledge of a pairing id, is what authorises the grant.

    Possession of the fragment is *all* it proves. Nothing binds this approval to
    the computer that asked for it, so a consent link forwarded to a researcher
    and clicked mints a token on that researcher's account for someone else's
    device. The confirmation code on the consent screen is what gives the
    researcher a chance to notice; it is a mitigation, not a barrier. See ADR
    0001, "Pairing phishing".
    """
    now = datetime.now(UTC)
    device = await _pairings.approve_pairing(
        db,
        user=current_user,
        pairing_id=pairing_id,
        verification_token=body.verification_token,
        request_ip=client_ip_for_request(request),
        now=now,
    )
    return device_response_from_orm(device, service=_pairings, now=now)


@router.post("/devices/pairings/{pairing_id}/deny", status_code=status.HTTP_204_NO_CONTENT)
async def deny_device_pairing(
    pairing_id: UUID,
    body: PairingConsentRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _pairings.deny_pairing(
        db,
        user=current_user,
        pairing_id=pairing_id,
        verification_token=body.verification_token,
    )


@router.get("/devices", response_model=list[DeviceResponse])
async def list_devices(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    include_revoked: bool = False,
) -> list[DeviceResponse]:
    now = datetime.now(UTC)
    devices = await _devices.list_devices(db, user=current_user, include_revoked=include_revoked)
    return [device_response_from_orm(device, service=_pairings, now=now) for device in devices]


@router.delete("/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_device(
    device_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove a computer's access. Works from a phone; the helper is not consulted.

    Takes effect on that device's next request - every device call re-reads the
    row, so there is no cache expiry to wait out.
    """
    await _devices.revoke_device(db, user=current_user, device_id=device_id)
