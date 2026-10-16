"""Pending direct-upload sentinels: which blob, if any, a pending part row owns.

``delete_part`` and the abandoned-upload sweep both turn a pending row into a media
deletion intent, and must agree: ``pending:<key>`` owns ``<key>``, a bare ``pending``
row owns nothing, and an ordinary key owns itself.
"""

from __future__ import annotations

import uuid

import pytest

from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.media_store.keys import is_pending_key, minted_pending_key
from backend.document.infrastructure.orm_models import DocumentPart, MediaDeletionIntent


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


class _Result:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _Session:
    def __init__(self, queued: bool = False) -> None:
        self._queued = queued
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    async def execute(self, _statement: object) -> _Result:
        return _Result(uuid.uuid4() if self._queued else None)

    def add(self, item: object) -> None:
        self.added.append(item)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def commit(self) -> None:
        self.commits += 1


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

    intents = [item.image_key for item in session.added if isinstance(item, MediaDeletionIntent)]
    assert intents == ([intent_key] if intent_key else [])
    assert session.deleted == [part]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_delete_part_does_not_queue_a_key_twice() -> None:
    session = _Session(queued=True)

    await DocumentRepository().delete_part(session, _part("parts/abc.webp"))

    assert session.added == []
