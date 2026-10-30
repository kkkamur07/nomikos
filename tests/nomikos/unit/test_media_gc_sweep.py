"""sweep_abandoned_part_uploads against a fake system session.

A direct upload the browser never finalized leaves a ``pending`` part row. Past the
presign window the row is deleted and the minted key, if the row carries one, is queued
once for the durable delete pass.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.expression import Executable

from backend.document.infrastructure import media_gc
from backend.document.infrastructure.orm_models import DocumentPart, MediaDeletionIntent


class _Rows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> list[object]:
        return list(self._rows)

    def scalar_one_or_none(self) -> object:
        return self._rows[0] if self._rows else None


class _Session:
    def __init__(self, parts: list[DocumentPart], queued: set[str] | None = None) -> None:
        self._parts = parts
        self._queued = queued or set()
        self.statements: list[Executable] = []
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    def execute(self, statement: Executable) -> _Rows:
        self.statements.append(statement)
        if len(self.statements) == 1:
            return _Rows(list(self._parts))
        params = set(statement.compile().params.values())
        return _Rows([uuid.uuid4()] if params & self._queued else [])

    def add(self, item: object) -> None:
        self.added.append(item)

    def delete(self, item: object) -> None:
        self.deleted.append(item)

    def commit(self) -> None:
        self.commits += 1


def _install(monkeypatch: pytest.MonkeyPatch, session: _Session) -> None:
    @contextmanager
    def fake_session():
        yield session

    monkeypatch.setattr(media_gc, "sync_system_session", fake_session)


def _part(image_key: str) -> DocumentPart:
    return DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key=image_key)


def test_reaps_pending_rows_and_queues_only_minted_keys(monkeypatch) -> None:
    minted = _part("pending:parts/a.webp")
    bare = _part("pending")
    session = _Session([minted, bare])
    _install(monkeypatch, session)

    reaped = media_gc.sweep_abandoned_part_uploads(batch_size=7)

    assert reaped == 2
    assert session.deleted == [minted, bare]
    assert [
        intent.image_key for intent in session.added if isinstance(intent, MediaDeletionIntent)
    ] == ["parts/a.webp"]
    assert session.commits == 1
    query = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "LIKE" in query
    assert "FOR UPDATE SKIP LOCKED" in query
    assert session.statements[0].compile().params["param_1"] == 7


def test_an_already_queued_key_is_not_queued_twice(monkeypatch) -> None:
    session = _Session([_part("pending:parts/a.webp")], queued={"parts/a.webp"})
    _install(monkeypatch, session)

    assert media_gc.sweep_abandoned_part_uploads() == 1
    assert session.added == []


def test_nothing_abandoned_reaps_nothing(monkeypatch) -> None:
    session = _Session([])
    _install(monkeypatch, session)

    assert media_gc.sweep_abandoned_part_uploads() == 0
    assert session.commits == 1
