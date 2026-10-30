"""Part reordering must never collide with uq_document_parts_document_order.

The reorder parks every row on a temporary order before writing the final 0..n-1 range.
When the surviving parts no longer start at order 0 — the state left behind by deleting
the first pages of a document — the temporary range has to clear the target range too.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.dml import Update
from sqlalchemy.sql.elements import Case
from sqlalchemy.sql.selectable import Select

from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import Document, DocumentPart
from backend.document.infrastructure.part_repository import temporary_reorder_offset


class _Result:
    def __init__(self, rows: list[DocumentPart]) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[DocumentPart]:
        return list(self._rows)


class _UniqueOrderSession:
    """Fake session applying the repository's UPDATEs under the unique order constraint."""

    def __init__(self, parts: list[DocumentPart]) -> None:
        self.parts = parts
        self.commits = 0
        self.selects: list[Select] = []

    def _assign(self, part: DocumentPart, order: int) -> None:
        clash = next(
            (other for other in self.parts if other is not part and other.order == order), None
        )
        if clash is not None:
            raise IntegrityError(
                "UPDATE document_parts",
                {},
                Exception(
                    "duplicate key value violates unique constraint "
                    '"uq_document_parts_document_order"'
                ),
            )
        part.order = order

    async def execute(self, statement):
        if isinstance(statement, Select):
            self.selects.append(statement)
        if isinstance(statement, Update):
            order_value = next(
                value for column, value in statement._values.items() if column.name == "order"
            )
            if isinstance(order_value, Case):
                # Final bulk write: CASE document_parts.id WHEN <id> THEN <order>.
                # Applied row by row like Postgres would, which is collision-free only
                # because the offset below already vacated the target range.
                target = {crit.value: result.value for crit, result in order_value.whens}
                for part in self.parts:
                    self._assign(part, target[part.id])
            else:
                # Temporary offset: order = order + :offset, applied to every row.
                offset = statement.compile(dialect=postgresql.dialect()).params["order_1"]
                for part in sorted(self.parts, key=lambda part: part.order):
                    self._assign(part, part.order + offset)
            return _Result([])
        return _Result(sorted(self.parts, key=lambda part: part.order))

    async def commit(self) -> None:
        self.commits += 1


def _parts(orders: list[int], document_id: uuid.UUID) -> list[DocumentPart]:
    return [
        DocumentPart(id=uuid.uuid4(), document_id=document_id, order=order, image_key=f"p{order}")
        for order in orders
    ]


# --- Temporary offset invariant ---


@pytest.mark.parametrize(
    "orders",
    [
        [0, 1, 2],
        [1, 2],
        # Orders left by deleting the first five pages and uploading two more.
        [5, 6, 7],
        [9],
        [4, 40, 400],
        [-3, 0, 2],
    ],
)
def test_temporary_offset_clears_both_the_old_and_the_target_range(orders: list[int]) -> None:
    offset = temporary_reorder_offset(orders, len(orders))
    shifted = {order + offset for order in orders}

    assert len(shifted) == len(orders)
    assert shifted.isdisjoint(set(orders)), "temporary orders collide with rows not yet moved"
    assert shifted.isdisjoint(set(range(len(orders)))), (
        "temporary orders collide with the target range"
    )


# --- End-to-end reorder against the unique constraint ---


@pytest.mark.asyncio
@pytest.mark.parametrize("orders", [[0, 1, 2], [5, 6, 7], [10, 11]])
async def test_reorder_parts_rewrites_orders_without_violating_uniqueness(
    orders: list[int],
) -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    parts = _parts(orders, document.id)
    session = _UniqueOrderSession(parts)
    reversed_ids = [part.id for part in reversed(parts)]

    result = await DocumentRepository().reorder_parts(session, document, reversed_ids)

    assert [part.id for part in result] == reversed_ids
    assert [part.order for part in result] == list(range(len(orders)))
    assert session.commits == 1


@pytest.mark.asyncio
async def test_reorder_parts_rejects_a_mismatched_id_set() -> None:
    document = Document(id=uuid.uuid4(), name="codex")
    parts = _parts([0, 1], document.id)
    session = _UniqueOrderSession(parts)

    assert await DocumentRepository().reorder_parts(session, document, [parts[0].id]) == []
    assert (
        await DocumentRepository().reorder_parts(session, document, [parts[0].id, parts[0].id])
        == []
    )
    assert session.commits == 0


@pytest.mark.asyncio
async def test_the_locking_read_refreshes_the_rows_it_locked() -> None:
    """The row lock decided nothing while the orders came from the identity map.

    `reorder_parts` is reached through `require_document`, which eager-loads
    `Document.parts`, so by the time the `SELECT ... FOR UPDATE` runs every row is
    already a mapped instance. SQLAlchemy returns those instances as they stand and
    does not overwrite loaded attributes with what it just read, so `part.order` could
    still hold a value a concurrent reorder had committed away - and the temporary
    offset computed from it could land on the range that transaction wrote, violating
    uq_document_parts_document_order.

    Asserted on the statement rather than on two racing sessions: the divergence only
    exists inside a real identity map, and a fake that modelled one would be asserting
    this test's own idea of SQLAlchemy rather than SQLAlchemy.
    """
    document = Document(id=uuid.uuid4(), name="codex")
    parts = _parts([0, 1, 2], document.id)
    session = _UniqueOrderSession(parts)

    await DocumentRepository().reorder_parts(session, document, [part.id for part in parts])

    locking_read = session.selects[0]
    assert locking_read._for_update_arg is not None, "the read must still take the row lock"
    assert locking_read.get_execution_options().get("populate_existing") is True
