"""Durable media-deletion intents, drained by ``media_gc``."""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.infrastructure.orm_models import MediaDeletionIntent


class MediaIntentRepository:
    async def enqueue_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        await self._add_media_deletion_intent(session, image_key)
        await session.commit()

    async def _add_media_deletion_intent(self, session: AsyncSession, image_key: str) -> None:
        """Stage an intent unless one for this key is already queued.

        ``image_key`` is unique on the table, and a rejected finalize may already have
        queued the same key. One ``INSERT ... ON CONFLICT DO NOTHING`` rather than a
        check then an insert, so two transactions queuing the same key concurrently
        cannot both pass the check and fail the loser's whole transaction.
        """
        await session.execute(
            insert(MediaDeletionIntent)
            .values(image_key=image_key)
            .on_conflict_do_nothing(index_elements=[MediaDeletionIntent.image_key])
        )
