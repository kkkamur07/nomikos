"""A line text save names the text it was edited from; a different stored text is a 409.

Two tabs on one page each save a row. Without this check the second save
silently replaces the first. The client sends the text its row was seeded with
as ``previous_text``; the server compares it with what it holds and refuses a
mismatch, so the second tab is told instead of overwriting. Leaving the field
out keeps the old last-write-wins behaviour for callers that do not send it.
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.exceptions import ConflictError
from backend.document.infrastructure.orm_models import LineTranscription
from tests.nomikos.unit.test_transcription_write_locking import _service, _Session


def _session_holding(stored: LineTranscription | None):
    class _TextSession(_Session):
        def __init__(self) -> None:
            super().__init__()
            self.added: list[object] = []

        async def execute(self, _stmt):
            class _Result:
                @staticmethod
                def scalar_one_or_none():
                    return stored

            return _Result()

        def add(self, item: object) -> None:
            self.added.append(item)

    return _TextSession()


def _patched_service():
    service, _repository, part, line = _service()
    layer = service._ground_truth._layer

    async def _transcription_or_404(*_args, **_kwargs):
        return layer

    async def _line_in_document_or_404(*_args, **_kwargs):
        return line

    service._transcription_or_404 = _transcription_or_404  # type: ignore[method-assign]
    service._line_in_document_or_404 = _line_in_document_or_404  # type: ignore[method-assign]
    return service, part, line, layer


def _stored(line, layer, text: str) -> LineTranscription:
    return LineTranscription(
        id=uuid.uuid4(), line_id=line.id, transcription_id=layer.id, text=text, confidence=None
    )


@pytest.mark.asyncio
async def test_a_save_from_stale_text_is_a_conflict_and_writes_nothing() -> None:
    service, part, line, layer = _patched_service()
    stored = _stored(line, layer, "FROM_A")
    session = _session_holding(stored)

    with pytest.raises(ConflictError):
        await service.patch_ground_truth_line_text(
            session,
            object(),
            uuid.uuid4(),
            part.document_id,
            layer.id,
            line.id,
            text="FROM_B",
            previous_text="",
        )

    assert stored.text == "FROM_A"
    assert session.commits == 0


@pytest.mark.asyncio
async def test_a_first_save_from_an_empty_row_conflicts_when_text_appeared() -> None:
    """No stored row means the empty text; a client that saw text there is stale."""
    service, part, line, layer = _patched_service()
    session = _session_holding(None)

    with pytest.raises(ConflictError):
        await service.patch_ground_truth_line_text(
            session,
            object(),
            uuid.uuid4(),
            part.document_id,
            layer.id,
            line.id,
            text="new",
            previous_text="old",
        )

    assert session.added == []


@pytest.mark.asyncio
async def test_a_save_from_the_stored_text_is_written() -> None:
    service, part, line, layer = _patched_service()
    stored = _stored(line, layer, "FROM_A")
    session = _session_holding(stored)

    await service.patch_ground_truth_line_text(
        session,
        object(),
        uuid.uuid4(),
        part.document_id,
        layer.id,
        line.id,
        text="FROM_A and more",
        previous_text="FROM_A",
    )

    assert stored.text == "FROM_A and more"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_a_save_without_previous_text_overwrites_as_before() -> None:
    service, part, line, layer = _patched_service()
    stored = _stored(line, layer, "FROM_A")
    session = _session_holding(stored)

    await service.patch_ground_truth_line_text(
        session,
        object(),
        uuid.uuid4(),
        part.document_id,
        layer.id,
        line.id,
        text="FROM_B",
    )

    assert stored.text == "FROM_B"
    assert session.commits == 1
