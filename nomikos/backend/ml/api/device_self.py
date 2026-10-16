"""Device-authenticated self-service routes (``X-Nomikos-Device-Token``).

Kept deliberately minimal. The claim / heartbeat / complete / release surface
lives with the job lifecycle and drops in on top of the same
``get_current_device`` dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ml.api.device_dependencies import require_device_pairing_enabled
from backend.ml.api.device_schemas import DeviceSelfResponse, DeviceTokenRenewResponse
from backend.ml.application.device_auth import AuthenticatedDevice
from backend.ml.application.device_management import DeviceManagementService
from backend.users.api.dependencies import get_current_device
from backend.users.api.rate_limit import client_ip_for_request
from infrastructure.db import get_db

router = APIRouter(tags=["devices"], dependencies=[Depends(require_device_pairing_enabled)])
_service = DeviceManagementService()


@router.get("/device/v1/self", response_model=DeviceSelfResponse)
async def read_device_self(
    request: Request,
    current_device: Annotated[AuthenticatedDevice, Depends(get_current_device)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeviceSelfResponse:
    """Confirm the credential and record liveness.

    ``server_time`` is returned so a laptop with a wrong clock still behaves,
    and ``token_expires_at`` is what tells the helper when to renew.
    """
    now = datetime.now(UTC)
    device = await _service.touch_device(
        db,
        device=current_device.device,
        request_ip=client_ip_for_request(request),
        now=now,
    )
    return DeviceSelfResponse(
        device_id=device.id,
        name=device.name,
        account_email=current_device.user.email,
        token_expires_at=device.token_expires_at,
        server_time=now,
    )


@router.post("/device/v1/token/renew", response_model=DeviceTokenRenewResponse)
async def renew_device_token(
    current_device: Annotated[AuthenticatedDevice, Depends(get_current_device)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeviceTokenRenewResponse:
    """Replace this device's token, keeping the previous one valid for an overlap.

    A helper with no UI cannot be told that a lost rotation response has bricked
    it, so the predecessor stays valid for ``DEVICE_TOKEN_RENEW_OVERLAP_HOURS``.
    """
    issued = await _service.renew_token(db, device=current_device.device)
    return DeviceTokenRenewResponse(
        device_token=issued.device_token,
        token_expires_at=issued.token_expires_at,
    )
