"""Durable object-store deletion retries for document media."""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

import infrastructure.models  # noqa: F401 - register MediaDeletionIntent mapper
from backend.document.infrastructure.media_store import (
    MediaStore,
    get_media_store,
    persisted_thumbnail_keys,
)
from backend.document.infrastructure.media_store.keys import minted_pending_key
from backend.document.infrastructure.orm_models import DocumentPart, MediaDeletionIntent
from infrastructure.db import sync_system_session

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncio import Event

# How long a begun-but-never-finalized direct upload may sit before it is reaped.
# Supabase's signed upload URLs live for a fixed ~2 hours (the requested expiry is
# not honored), so past this bound the browser can no longer complete the upload.
ABANDONED_PART_UPLOAD_MAX_AGE = timedelta(hours=3)


def delete_media_object(store: MediaStore, image_key: str) -> None:
    """Delete a page image and the thumbnails persisted next to it.

    The original's deletion is the intent and raises like before, so a failure is
    retried. The thumbnails are swept afterwards on a best-effort basis: they are
    only ever read through their original's part, so one left behind is a few
    stray kilobytes, not a leak of anything reachable.
    """
    store.delete(image_key)
    for derived_key in persisted_thumbnail_keys(image_key):
        try:
            store.delete(derived_key)
        except Exception:
            logger.warning("persisted thumbnail not deleted key=%s", derived_key, exc_info=True)


def process_media_deletion_intents(*, batch_size: int = 50) -> int:
    """Delete a bounded batch of committed media intents."""
    completed = 0
    with sync_system_session() as session:
        intents = list(
            session.execute(
                select(MediaDeletionIntent)
                .where(MediaDeletionIntent.completed_at.is_(None))
                .order_by(MediaDeletionIntent.created_at, MediaDeletionIntent.id)
                .with_for_update(skip_locked=True)
                .limit(batch_size)
            ).scalars()
        )
        store = get_media_store()
        for intent in intents:
            try:
                delete_media_object(store, intent.image_key)
            except Exception as exc:
                intent.attempts += 1
                intent.last_error = f"{type(exc).__name__}: {str(exc)[:900]}"
                logger.warning(
                    "media deletion deferred intent_id=%s attempt=%s exception=%s",
                    intent.id,
                    intent.attempts,
                    type(exc).__name__,
                )
                continue
            intent.attempts += 1
            intent.last_error = None
            intent.completed_at = datetime.now(UTC)
            completed += 1
        session.commit()
    return completed


def sweep_abandoned_part_uploads(*, batch_size: int = 50) -> int:
    """Reap direct uploads whose browser never came back to finalize.

    ``begin_upload`` commits the part row before the browser PUTs the bytes, so a
    closed tab leaves a ``pending`` row - visible in the document as a broken page -
    and possibly an uploaded blob referenced by nothing. Rows older than the presign
    window can never be finalized: delete the row and queue the minted key (kept on
    the row as ``pending:<key>``) for the durable delete pass above.
    """
    cutoff = datetime.now(UTC) - ABANDONED_PART_UPLOAD_MAX_AGE
    reaped = 0
    with sync_system_session() as session:
        parts = list(
            session.execute(
                select(DocumentPart)
                .where(
                    DocumentPart.image_key.like("pending%"),
                    DocumentPart.created_at < cutoff,
                )
                .with_for_update(skip_locked=True)
                .limit(batch_size)
            ).scalars()
        )
        for part in parts:
            minted_key = minted_pending_key(part.image_key)
            if minted_key:
                already_queued = session.execute(
                    select(MediaDeletionIntent.id).where(
                        MediaDeletionIntent.image_key == minted_key
                    )
                ).scalar_one_or_none()
                if already_queued is None:
                    session.add(MediaDeletionIntent(image_key=minted_key))
            session.delete(part)
            reaped += 1
        session.commit()
    if reaped:
        logger.info("Reaped %d abandoned part upload(s)", reaped)
    return reaped


async def media_gc_loop(stop_event: Event, *, interval_seconds: float = 60.0) -> None:
    """Periodically retry durable media deletes without blocking request handlers."""
    import asyncio

    while not stop_event.is_set():
        try:
            await asyncio.to_thread(process_media_deletion_intents)
        except Exception:
            logger.exception("media deletion GC pass failed")
        try:
            await asyncio.to_thread(sweep_abandoned_part_uploads)
        except Exception:
            logger.exception("abandoned part upload sweep failed")
        # Timing out is the normal path: it means no shutdown was requested and
        # the next sweep is due.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
