"""DocumentPart ordering and deletion."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.document.infrastructure.media_intent_repository import MediaIntentRepository
from backend.document.infrastructure.media_store.keys import is_pending_key, minted_pending_key
from backend.document.infrastructure.orm_models import Document, DocumentPart


def temporary_reorder_offset(current_orders: Sequence[int], target_count: int) -> int:
    """Offset that parks every row below the ``[0, target_count)`` range being written.

    ``uq_document_parts_document_order`` is checked per statement, so the shift has to
    clear both the orders still in the table and the orders about to be assigned.
    Anchoring at ``min(minimum_order, 0)`` matters: when the surviving parts no longer
    start at 0 (the first pages were deleted), anchoring at ``minimum_order`` leaves the
    shifted rows inside the target range and the reorder collides with itself.
    """
    minimum_order = min(current_orders)
    maximum_order = max(current_orders)
    return min(minimum_order, 0) - maximum_order - target_count - 1


class PartRepository(MediaIntentRepository):
    async def part_page_number(self, session: AsyncSession, part: DocumentPart) -> int:
        """1-based position of ``part`` among its document's parts, in display order.

        Computed from the rows rather than read off ``part.order`` because orders are
        allowed to have gaps: ``next_part_order`` hands out max+1, so deleting a middle
        part leaves a hole, and the UI numbers pages by position, not by order value.
        """
        result = await session.execute(
            select(func.count())
            .select_from(DocumentPart)
            .where(
                DocumentPart.document_id == part.document_id,
                DocumentPart.order < part.order,
            )
        )
        return int(result.scalar_one()) + 1

    async def next_part_order(self, session: AsyncSession, document_id: UUID) -> int:
        await session.execute(
            select(Document.id).where(Document.id == document_id).with_for_update()
        )
        result = await session.execute(
            select(DocumentPart.order)
            .where(DocumentPart.document_id == document_id)
            .order_by(DocumentPart.order.desc())
            .limit(1)
        )
        current = result.scalar_one_or_none()
        if current is None:
            return 0
        return current + 1

    async def reorder_parts(
        self, session: AsyncSession, document: Document, ordered_part_ids: list[UUID]
    ) -> list[DocumentPart]:
        # ``populate_existing`` is what makes ``with_for_update`` mean anything here.
        # The caller reached this through ``require_document``, which eager-loads
        # ``Document.parts``, so every row is already in the session's identity map -
        # and SQLAlchemy returns the mapped instance untouched rather than overwriting
        # loaded attributes with the freshly locked values. The offset below is computed
        # from ``part.order``, so without this it could be computed from orders a
        # concurrent reorder had already superseded, land on the range that transaction
        # wrote, and violate uq_document_parts_document_order - a 500 on a plain drag.
        # ``order_by(id)`` is for the locking, not the reading: everything below
        # works off a dict and set comparisons, so the order rows arrive in does
        # not matter, but the order they are *locked* in does. ``lock_parts``
        # takes the same one, so two transactions that overlap on this document
        # queue on the same row rather than each holding what the other needs.
        result = await session.execute(
            select(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .order_by(DocumentPart.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        current_parts = list(result.scalars().all())
        parts_by_id = {part.id: part for part in current_parts}
        if len(ordered_part_ids) != len(parts_by_id):
            return []
        if len(set(ordered_part_ids)) != len(ordered_part_ids):
            return []
        if set(ordered_part_ids) != set(parts_by_id):
            return []
        temporary_offset = temporary_reorder_offset(
            [part.order for part in current_parts], len(ordered_part_ids)
        )
        await session.execute(
            update(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .values(order=DocumentPart.order + temporary_offset)
        )
        # One bulk write instead of a statement per part. Safe after the offset
        # above cleared the [0, n) range: every row is remapped to a distinct final
        # order, and set-equality was checked, so no row is left with a null order.
        final_order = case(
            {part_id: index for index, part_id in enumerate(ordered_part_ids)},
            value=DocumentPart.id,
        )
        await session.execute(
            update(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .values(order=final_order)
        )
        await session.commit()
        result = await session.execute(
            select(DocumentPart)
            .where(DocumentPart.document_id == document.id)
            .order_by(DocumentPart.order)
        )
        return list(result.scalars().all())

    async def delete_part(self, session: AsyncSession, part: DocumentPart) -> None:
        # A pending row's key is the sentinel, not an object key: enqueueing it
        # verbatim would poison the GC with a key the store refuses forever. The
        # sentinel carries the minted key; a bare "pending" row has no blob at all.
        image_key = part.image_key
        if is_pending_key(image_key):
            image_key = minted_pending_key(image_key) or ""
        if image_key:
            await self._add_media_deletion_intent(session, image_key)
        await session.delete(part)
        await session.commit()
