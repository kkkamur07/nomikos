"""Paired-device lifecycle: listing, revocation, liveness, and token renewal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.core.settings.device import DeviceSettings, get_device_settings
from backend.ml.application.device_status import _clean_text, _now
from backend.ml.application.opaque_tokens import (
    device_token_prefix,
    format_device_token,
    hash_secret,
    new_secret,
)
from backend.ml.infrastructure.device_orm_models import (
    MAX_HELPER_VERSION_LENGTH,
    MAX_IP_LENGTH,
    HelperDevice,
)
from backend.ml.infrastructure.device_repository import HelperDeviceRepository
from backend.users.infrastructure.orm_models import User

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IssuedDeviceToken:
    device: HelperDevice
    device_token: str
    token_expires_at: datetime


def _mint_token(
    device: HelperDevice, settings: DeviceSettings, *, now: datetime
) -> IssuedDeviceToken:
    secret = new_secret()
    expires_at = now + timedelta(days=settings.device_token_ttl_days)
    device.token_hash = hash_secret(secret, settings.hmac_key())
    device.token_prefix = device_token_prefix(device.id)
    device.token_issued_at = now
    device.token_expires_at = expires_at
    device.previous_token_hash = None
    device.previous_token_expires_at = None
    return IssuedDeviceToken(
        device=device,
        device_token=format_device_token(device.id, secret),
        token_expires_at=expires_at,
    )


class DeviceManagementService:
    def __init__(
        self,
        repository: HelperDeviceRepository | None = None,
        settings: DeviceSettings | None = None,
    ) -> None:
        self._repo = repository or HelperDeviceRepository()
        self._settings = settings or get_device_settings()

    @property
    def settings(self) -> DeviceSettings:
        return self._settings

    async def list_devices(
        self, session: AsyncSession, *, user: User, include_revoked: bool = False
    ) -> list[HelperDevice]:
        return await self._repo.list_devices(session, user.id, include_revoked=include_revoked)

    async def revoke_device(
        self,
        session: AsyncSession,
        *,
        user: User,
        device_id: UUID,
        now: datetime | None = None,
    ) -> HelperDevice:
        """Kill a credential. Effective on the device's next request.

        There is no cache and no JWT to wait out: every device request re-reads
        this row, so revocation lands within one poll cycle.
        """
        now = now or _now()
        device = await self._repo.get_device_for_update(session, device_id)
        if device is None:
            raise NotFoundError("Device not found")
        if device.user_id != user.id:
            raise AccessDeniedError("Device belongs to another account")
        if device.revoked_at is None:
            device.revoked_at = now
            device.token_hash = ""
            device.previous_token_hash = None
            device.previous_token_expires_at = None
            logger.info(
                "device_revoked device_id=%s user_id=%s token_prefix=%s",
                device.id,
                device.user_id,
                device.token_prefix,
            )
        await session.commit()
        return device

    async def touch_device(
        self,
        session: AsyncSession,
        *,
        device: HelperDevice,
        request_ip: str | None = None,
        helper_version: str | None = None,
        capabilities: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> HelperDevice:
        """Record liveness. ``last_seen_at`` is the enqueue-time routing gate."""
        now = now or _now()
        device.last_seen_at = now
        if request_ip:
            device.last_seen_ip = request_ip[:MAX_IP_LENGTH]
        if helper_version:
            device.helper_version = _clean_text(helper_version, limit=MAX_HELPER_VERSION_LENGTH)
        if capabilities is not None:
            device.capabilities = capabilities
        await session.commit()
        return device

    async def renew_token(
        self,
        session: AsyncSession,
        *,
        device: HelperDevice,
        now: datetime | None = None,
    ) -> IssuedDeviceToken:
        """Issue a replacement token, keeping the old one valid for an overlap.

        Not per-request rotation: a browser that loses a rotation response logs
        in again, but a UI-less helper would be bricked with no terminal to
        recover from. The overlap makes a lost renewal response harmless.
        """
        now = now or _now()
        locked = await self._repo.get_device_for_update(session, device.id)
        if locked is None or locked.revoked_at is not None:
            raise NotFoundError("Device not found")
        previous_hash = locked.token_hash
        old_previous_hash = locked.previous_token_hash
        old_previous_expires_at = locked.previous_token_expires_at
        issued = _mint_token(locked, self._settings, now=now)
        if previous_hash:
            if (
                old_previous_hash
                and old_previous_expires_at is not None
                and old_previous_expires_at > now
            ):
                # A predecessor is still inside its overlap window, so a lost
                # renewal response may have left the helper holding it. Keep it
                # rather than replacing it with the token just rotated out,
                # which would brick that helper with no terminal to recover.
                locked.previous_token_hash = old_previous_hash
                locked.previous_token_expires_at = old_previous_expires_at
            else:
                locked.previous_token_hash = previous_hash
                locked.previous_token_expires_at = now + timedelta(
                    hours=self._settings.device_token_renew_overlap_hours
                )
        await session.commit()
        logger.info(
            "device_token_renewed device_id=%s user_id=%s token_prefix=%s expires_at=%s",
            locked.id,
            locked.user_id,
            locked.token_prefix,
            issued.token_expires_at.isoformat(),
        )
        return issued
