"""Durable media-deletion intents, drained by ``media_gc``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.infrastructure.orm_models import MediaDeletionIntent


class MediaIntentRepository:
    async def enqueue_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        await self._add_media_deletion_intent(session, image_key)
        await session.commit()

    async def _add_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        """Stage an intent unless one for this key is already queued.

        ``image_key`` is unique on the table, so a blind insert after a rejected
        finalize already queued the same key would fail the whole transaction.
        """
        existing = await session.execute(
            select(MediaDeletionIntent.id).where(MediaDeletionIntent.image_key == image_key)
        )
        if existing.scalar_one_or_none() is None:
            session.add(MediaDeletionIntent(image_key=image_key))
