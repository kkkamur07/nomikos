"""Unauthenticated helper bootstrap: start a pairing, poll for the token.

These two routes are the only device-pairing surface an unpaired helper can
reach. Everything else requires either a logged-in browser or an already-issued
device token.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import ConflictError
from backend.ml.api.device_dependencies import require_device_pairing_enabled
from backend.ml.api.device_schemas import (
    PairingStartRequest,
    PairingStartResponse,
    PairingTokenRequest,
    PairingTokenResponse,
)
from backend.ml.application.pairing_service import DevicePairingService
from backend.users.api.rate_limit import client_ip_for_request, throttle_device_pairing_starts
from infrastructure.db import get_db

router = APIRouter(tags=["devices"], dependencies=[Depends(require_device_pairing_enabled)])
_service = DevicePairingService()

_USER_AGENT_LIMIT = 255


@router.post(
    "/device/v1/pairings",
    response_model=PairingStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_device_pairing(
    body: PairingStartRequest,
    request: Request,
    _rate_limit: Annotated[None, Depends(throttle_device_pairing_starts)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PairingStartResponse:
    """Create a pairing request and hand the helper its two one-time secrets.

    Under its own throttle, not the shared auth one. The body carries no account
    identity, so under ``throttle_auth_attempts`` this route had no per-caller
    dimension at all and every honest pairing was charged to the coarse
    ``unattributable:<path>`` bucket - one bucket shared by every researcher, so
    filling it locked all of them out of `nomikos pair`.
    ``throttle_device_pairing_starts`` charges a per-client bucket where the
    address identifies one client and charges nothing where it does not, leaving
    the platform-wide live-pairing ceiling below as the bound on table growth.
    """
    request_ip = client_ip_for_request(request)
    try:
        started = await _service.start_pairing(
            db,
            device_name=body.device_name,
            platform=body.platform,
            helper_version=body.helper_version,
            capabilities=body.capabilities,
            request_ip=request_ip,
            user_agent=(request.headers.get("user-agent") or "")[:_USER_AGENT_LIMIT] or None,
        )
    except ConflictError as exc:
        # The only conflict this method raises is the platform-wide live-pairing
        # ceiling, which is a backstop against table growth, not an abuse control.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": str(_service.settings.device_pairing_ttl_seconds)},
        ) from exc
    return PairingStartResponse(
        pairing_id=started.pairing_id,
        device_code=started.device_code,
        verification_url=started.verification_url,
        confirmation_code=started.confirmation_code,
        expires_in=started.expires_in,
        interval_seconds=started.interval_seconds,
    )


@router.post("/device/v1/pairings/token", response_model=PairingTokenResponse)
async def collect_device_token(
    body: PairingTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PairingTokenResponse:
    """Poll for the browser's decision; return the device token exactly once.

    Deliberately **not** under ``throttle_auth_attempts``: that limiter is
    10 requests / 60s / (ip + path), and a compliant 5 second poll is 12 per
    minute, so a well-behaved helper would throttle itself at poll 11. Cadence
    is enforced on the pairing row instead (``slow_down`` with a doubling
    interval), and wrong ``device_code`` presentations burn the row after
    ``DEVICE_PAIRING_MAX_ATTEMPTS``.

    Always HTTP 200 - see :class:`PairingTokenResponse`.
    """
    result = await _service.collect_token(
        db, pairing_id=body.pairing_id, device_code=body.device_code
    )
    return PairingTokenResponse(
        status=result.status,
        interval_seconds=result.interval_seconds,
        device_id=result.device_id,
        device_token=result.device_token,
        token_expires_at=result.token_expires_at,
        account_email=result.account_email,
    )
