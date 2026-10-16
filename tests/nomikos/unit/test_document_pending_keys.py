"""Pending direct-upload sentinels: which blob, if any, a pending part row owns.

``delete_part`` and the abandoned-upload sweep both turn a pending row into a media
deletion intent, and must agree: ``pending:<key>`` owns ``<key>``, a bare ``pending``
row owns nothing, and an ordinary key owns itself.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.expression import Executable

from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store.keys import is_pending_key, minted_pending_key
from backend.document.infrastructure.orm_models import DocumentPart


@pytest.mark.parametrize(
    ("image_key", "pending", "minted"),
    [
        ("pending:parts/abc.webp", True, "parts/abc.webp"),
        ("pending", True, None),
        ("pending:", True, None),
        ("parts/abc.webp", False, None),
    ],
)
def test_pending_key_parsing(image_key: str, pending: bool, minted: str | None) -> None:
    assert is_pending_key(image_key) is pending
    assert minted_pending_key(image_key) == minted


class _Session:
    def __init__(self) -> None:
        self.statements: list[Executable] = []
        self.deleted: list[object] = []
        self.commits = 0

    async def execute(self, statement: Executable) -> None:
        self.statements.append(statement)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def commit(self) -> None:
        self.commits += 1

    def queued_keys(self) -> list[str]:
        return [statement.compile().params["image_key"] for statement in self.statements]


def _part(image_key: str) -> DocumentPart:
    return DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key=image_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_key", "intent_key"),
    [
        ("parts/abc.webp", "parts/abc.webp"),
        ("pending:parts/abc.webp", "parts/abc.webp"),
        ("pending", None),
    ],
)
async def test_delete_part_queues_the_blob_the_row_owns(image_key: str, intent_key: str | None):
    session = _Session()
    part = _part(image_key)

    await DocumentRepository().delete_part(session, part)

    assert session.queued_keys() == ([intent_key] if intent_key else [])
    assert session.deleted == [part]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_queuing_an_already_queued_key_is_a_no_op_insert() -> None:
    session = _Session()

    await DocumentRepository().enqueue_media_deletion_intent(session, "parts/abc.webp")

    [statement] = session.statements
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert sql.startswith("INSERT INTO media_deletion_intents")
    assert "ON CONFLICT (image_key) DO NOTHING" in sql
    assert session.commits == 1
