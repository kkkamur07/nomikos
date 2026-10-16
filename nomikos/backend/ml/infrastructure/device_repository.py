"""Persistence for helper devices and pairing requests.

The repository is the only module that talks to Postgres, which is what lets the
pairing state machine be unit-tested against an in-memory double without a
database. Transaction control (``commit``) stays in the service so an approve or
a token mint is one atomic step.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.jobs.infrastructure.job_sql import rows_affected
from backend.ml.domain.execution import ExecutionTarget
from backend.ml.infrastructure.device_orm_models import HelperDevice, HelperPairing


class HelperDeviceRepository:
    async def get_device(self, session: AsyncSession, device_id: UUID) -> HelperDevice | None:
        """Primary-key fetch with the owning user eagerly loaded.

        The token carries its own device id, so authentication never scans
        ``token_hash``. The user is joined in because every device request has
        to resolve the authorization scope anyway.
        """
        result = await session.execute(
            select(HelperDevice)
            .options(selectinload(HelperDevice.user))
            .where(HelperDevice.id == device_id)
        )
        return result.scalar_one_or_none()

    async def get_device_for_update(
        self, session: AsyncSession, device_id: UUID
    ) -> HelperDevice | None:
        result = await session.execute(
            select(HelperDevice).where(HelperDevice.id == device_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def list_devices(
        self, session: AsyncSession, user_id: UUID, *, include_revoked: bool = False
    ) -> list[HelperDevice]:
        query = select(HelperDevice).where(HelperDevice.user_id == user_id)
        if not include_revoked:
            query = query.where(HelperDevice.revoked_at.is_(None))
        result = await session.execute(query.order_by(HelperDevice.created_at, HelperDevice.id))
        return list(result.scalars().all())

    async def count_live_devices(self, session: AsyncSession, user_id: UUID) -> int:
        return (
            await session.scalar(
                select(func.count())
                .select_from(HelperDevice)
                .where(HelperDevice.user_id == user_id, HelperDevice.revoked_at.is_(None))
            )
            or 0
        )

    def add_device(self, session: AsyncSession, device: HelperDevice) -> HelperDevice:
        session.add(device)
        return device

    async def hosts_with_recent_devices(
        self, session: AsyncSession, *, user_id: UUID, seen_after: datetime
    ) -> frozenset[ExecutionTarget]:
        """Which **inference host**s currently have **capacity**.

        One query, because a researcher's laptop and a hosted worker are the same
        kind of thing: both are rows here, both report liveness through
        ``last_seen_at``, and capacity is "was one of them seen recently".

        The scoping differs and that is the whole asymmetry. A ``local`` device
        is one researcher's own computer, so it only answers for its owner; a
        ``cloud`` worker is platform infrastructure and answers for everyone.

        Reads ``ix_helper_devices_last_seen_at`` - the index the device layer
        already carries - rather than introducing a second liveness mechanism.
        """
        result = await session.execute(
            select(HelperDevice.inference_host)
            .where(
                HelperDevice.revoked_at.is_(None),
                HelperDevice.last_seen_at.is_not(None),
                HelperDevice.last_seen_at >= seen_after,
                or_(
                    HelperDevice.inference_host == ExecutionTarget.cloud,
                    HelperDevice.user_id == user_id,
                ),
            )
            .distinct()
        )
        return frozenset(result.scalars().all())

    async def get_pairing_for_update(
        self, session: AsyncSession, pairing_id: UUID
    ) -> HelperPairing | None:
        """Serialise every state transition of one pairing row.

        Approve, deny, and token collection all mutate the same row, and two of
        them mint or consume a credential. Without the row lock a concurrent
        approve/collect pair could mint twice.
        """
        result = await session.execute(
            select(HelperPairing).where(HelperPairing.id == pairing_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_pairing_by_verification_hash(
        self, session: AsyncSession, verification_token_hash: str
    ) -> HelperPairing | None:
        """The one lookup keyed on a digest - the browser holds no pairing id."""
        result = await session.execute(
            select(HelperPairing).where(
                HelperPairing.verification_token_hash == verification_token_hash
            )
        )
        return result.scalar_one_or_none()

    async def count_live_pairings(self, session: AsyncSession, now: datetime) -> int:
        """Platform-wide count of pairing requests that can still be approved.

        Deliberately not scoped by ``request_ip``. Behind a proxy the platform
        does not trust, every request carries the edge's address, so an IP-keyed
        count is one shared counter wearing a per-client label - which turns a
        per-client cap into a platform-wide outage. See ADR 0001.
        """
        return (
            await session.scalar(
                select(func.count())
                .select_from(HelperPairing)
                .where(
                    HelperPairing.expires_at > now,
                    HelperPairing.denied_at.is_(None),
                    HelperPairing.consumed_at.is_(None),
                )
            )
            or 0
        )

    async def delete_finished_pairings(self, session: AsyncSession, cutoff: datetime) -> int:
        """Drop pairing rows that can never be acted on again.

        ``helper_pairings`` is written by an unauthenticated endpoint, so without
        this the table only grows. The foreign key to ``helper_devices`` points
        *from* this table, so deleting a consumed pairing never touches the paired
        device it created.
        """
        result = await session.execute(
            delete(HelperPairing).where(
                or_(
                    HelperPairing.expires_at < cutoff,
                    and_(
                        or_(
                            HelperPairing.consumed_at.is_not(None),
                            HelperPairing.denied_at.is_not(None),
                        ),
                        HelperPairing.created_at < cutoff,
                    ),
                )
            )
        )
        return rows_affected(result)

    def add_pairing(self, session: AsyncSession, pairing: HelperPairing) -> HelperPairing:
        session.add(pairing)
        return pairing
