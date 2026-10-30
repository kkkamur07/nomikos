"""TranscriptionService page import and copy-to-ground-truth against in-memory fakes.

``import_page_transcription`` replaces a page's imported text and un-writes the ground
truth its old pairings produced. ``copy_to_ground_truth`` promotes a model layer's text
into the ground-truth layer, clearing confidence, after locking every affected part.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application import transcription_service as transcription_module
from backend.document.application.document_access import PartContext
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    PageTranscriptionLine,
    Transcription,
    TranscriptionKind,
)


class _Scalars:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return list(self._rows)


class _Result:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> _Scalars:
        return _Scalars(self._rows)


class _Session:
    def __init__(self, results: list[list] | None = None) -> None:
        self._results = list(results or [])
        self.statements: list[object] = []
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.flushes = 0
        self.commits = 0

    async def execute(self, statement: object) -> _Result:
        self.statements.append(statement)
        return _Result(self._results.pop(0))

    def add(self, item: object) -> None:
        self.added.append(item)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1


class _Access:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._context = PartContext(project=object(), document=document, part=part)

    async def require_part(self, *_args: object, **_kwargs: object) -> PartContext:
        return self._context

    async def require_document(self, *_args: object, **_kwargs: object) -> PartContext:
        return self._context


class _GroundTruth:
    def __init__(self, layer: Transcription) -> None:
        self.layer = layer
        self.writes: list[tuple[Line, Transcription, str | None]] = []

    async def layer_for(self, _session: object, _document: Document) -> Transcription:
        return self.layer

    async def write(
        self, _session: object, line: Line, layer: Transcription, text: str | None
    ) -> None:
        self.writes.append((line, layer, text))


class _Repository:
    def __init__(self, part: DocumentPart, ground_truth: Transcription | None) -> None:
        self._part = part
        self._ground_truth = ground_truth
        self.text_lines: list[PageTranscriptionLine] = []
        self.lines: dict[UUID, Line] = {}
        self.transcriptions: dict[UUID, Transcription] = {}
        self.locks: list[UUID] = []

    async def lock_part(self, _session: object, part_id: UUID) -> None:
        self.locks.append(part_id)

    async def lock_parts(self, _session: object, part_ids: list[UUID]) -> None:
        self.locks.extend(part_ids)

    async def list_page_transcription_lines(
        self, _session: object, _part_id: UUID
    ) -> list[PageTranscriptionLine]:
        return list(self.text_lines)

    async def get_ground_truth_transcription(
        self, _session: object, _document_id: UUID
    ) -> Transcription | None:
        return self._ground_truth

    async def get_line_in_part(self, _session: object, _part_id: UUID, line_id: UUID):
        return self.lines.get(line_id)

    async def get_transcription_in_document(
        self, _session: object, _document_id: UUID, transcription_id: UUID
    ) -> Transcription | None:
        return self.transcriptions.get(transcription_id)

    async def count_part_lines(self, _session: object, _part_id: UUID) -> int:
        return len(self.lines)

    async def count_paired_ground_truth_lines(self, _session: object, _part_id: UUID) -> int:
        return sum(1 for text_line in self.text_lines if text_line.paired_line_id)


class _World:
    def __init__(self, *, with_ground_truth: bool = True, results: list[list] | None = None):
        self.document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
        self.part = DocumentPart(
            id=uuid.uuid4(), document_id=self.document.id, order=0, image_key="page.webp"
        )
        self.layer = self._layer(TranscriptionKind.ground_truth)
        self.repository = _Repository(self.part, self.layer if with_ground_truth else None)
        self.ground_truth = _GroundTruth(self.layer)
        self.session = _Session(results)
        self.service = TranscriptionService(
            documents=self.repository,
            access=_Access(self.document, self.part),
            ground_truth=self.ground_truth,
        )

    def _layer(self, kind: TranscriptionKind) -> Transcription:
        return Transcription(id=uuid.uuid4(), document_id=self.document.id, name=kind, kind=kind)

    def model_layer(self, kind: TranscriptionKind = TranscriptionKind.model) -> Transcription:
        layer = self._layer(kind)
        self.repository.transcriptions[layer.id] = layer
        return layer

    def line(self) -> Line:
        line = Line(id=uuid.uuid4(), part_id=self.part.id, order=0, baseline={}, points=[])
        self.repository.lines[line.id] = line
        return line

    def text_line(self, order: int, paired_line_id: UUID | None = None) -> PageTranscriptionLine:
        text_line = PageTranscriptionLine(
            id=uuid.uuid4(),
            part_id=self.part.id,
            order=order,
            text=f"line {order}",
            paired_line_id=paired_line_id,
        )
        self.repository.text_lines.append(text_line)
        return text_line

    async def import_text(self, text: str) -> tuple[list[PageTranscriptionLine], dict[str, int]]:
        return await self.service.import_page_transcription(
            self.session,
            object(),
            self.document.project_id,
            self.document.id,
            self.part.id,
            text=text,
        )

    async def copy(self, source: Transcription, line_ids: list[UUID] | None = None) -> list[UUID]:
        return await self.service.copy_to_ground_truth(
            self.session,
            object(),
            self.document.project_id,
            self.document.id,
            source.id,
            line_ids=line_ids,
        )


def _new_text_lines(session: _Session) -> list[PageTranscriptionLine]:
    return [item for item in session.added if isinstance(item, PageTranscriptionLine)]


@pytest.mark.asyncio
async def test_import_splits_on_lines_and_drops_blank_ones() -> None:
    world = _World()

    await world.import_text("  alpha \n\n\tbeta\n   \ngamma")

    assert [(row.order, row.text) for row in _new_text_lines(world.session)] == [
        (0, "alpha"),
        (1, "beta"),
        (2, "gamma"),
    ]
    assert all(row.part_id == world.part.id for row in _new_text_lines(world.session))
    assert world.repository.locks == [world.part.id]
    assert world.session.commits == 1


@pytest.mark.asyncio
async def test_import_rejects_more_lines_than_the_cap_before_writing(monkeypatch) -> None:
    monkeypatch.setattr(transcription_module, "MAX_PAGE_TRANSCRIPTION_LINES", 2)
    world = _World()
    world.text_line(0)

    with pytest.raises(ValidationError, match="cannot exceed 2"):
        await world.import_text("a\nb\nc")

    assert world.session.deleted == []
    assert world.session.commits == 0


@pytest.mark.asyncio
async def test_reimport_replaces_old_text_and_unwrites_paired_ground_truth() -> None:
    world = _World()
    paired = world.line()
    old_paired = world.text_line(0, paired_line_id=paired.id)
    old_unpaired = world.text_line(1)

    await world.import_text("fresh")

    assert world.ground_truth.writes == [(paired, world.layer, None)]
    assert world.session.deleted == [old_paired, old_unpaired]
    # The old rows are flushed away before the new ones reuse their orders.
    assert world.session.flushes == 1
    assert [row.text for row in _new_text_lines(world.session)] == ["fresh"]


@pytest.mark.asyncio
async def test_reimport_without_a_ground_truth_layer_has_nothing_to_unwrite() -> None:
    world = _World(with_ground_truth=False)
    paired = world.line()
    world.text_line(0, paired_line_id=paired.id)

    await world.import_text("fresh")

    assert world.ground_truth.writes == []
    assert world.session.commits == 1


@pytest.mark.asyncio
async def test_reimport_with_a_pairing_to_a_vanished_line_is_a_404() -> None:
    world = _World()
    world.text_line(0, paired_line_id=uuid.uuid4())

    with pytest.raises(NotFoundError, match="Line not found"):
        await world.import_text("fresh")

    assert world.session.commits == 0


@pytest.mark.asyncio
async def test_import_returns_the_page_pairing_progress() -> None:
    world = _World()
    world.line()
    world.line()

    _text_lines, progress = await world.import_text("alpha")

    assert progress == {"paired_lines": 0, "total_lines": 2, "percent": 0}


def _row(line_id: UUID, layer: Transcription, text: str, confidence: float | None = 0.9):
    return LineTranscription(
        id=uuid.uuid4(),
        line_id=line_id,
        transcription_id=layer.id,
        text=text,
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_copy_requires_an_existing_layer() -> None:
    world = _World()

    with pytest.raises(NotFoundError, match="Transcription layer not found"):
        await world.copy(world._layer(TranscriptionKind.model))


@pytest.mark.asyncio
async def test_copy_refuses_a_ground_truth_source() -> None:
    world = _World()
    source = world.model_layer(TranscriptionKind.ground_truth)

    with pytest.raises(ConflictError, match="requires a model transcription layer"):
        await world.copy(source)

    assert world.session.statements == []


@pytest.mark.asyncio
async def test_copy_locks_parts_then_inserts_new_and_overwrites_existing_rows() -> None:
    other_part_id = uuid.uuid4()
    world = _World()
    source = world.model_layer()
    fresh_line, approved_line = uuid.uuid4(), uuid.uuid4()
    existing = _row(approved_line, world.layer, "old human text", confidence=0.4)
    world.session = _Session(
        [
            [world.part.id, other_part_id],
            [_row(fresh_line, source, "model one"), _row(approved_line, source, "model two")],
            [existing],
        ]
    )

    copied = await world.copy(source)

    assert copied == [fresh_line, approved_line]
    assert world.repository.locks == [world.part.id, other_part_id]
    [inserted] = world.session.added
    assert isinstance(inserted, LineTranscription)
    assert (inserted.line_id, inserted.transcription_id, inserted.text, inserted.confidence) == (
        fresh_line,
        world.layer.id,
        "model one",
        None,
    )
    assert (existing.text, existing.confidence) == ("model two", None)
    assert world.session.commits == 1


@pytest.mark.asyncio
async def test_copy_with_no_source_rows_skips_the_existing_rows_query() -> None:
    world = _World()
    source = world.model_layer()
    world.session = _Session([[], []])

    copied = await world.copy(source, line_ids=[uuid.uuid4()])

    assert copied == []
    assert len(world.session.statements) == 2
    assert world.session.added == []
    assert world.session.commits == 1


@pytest.mark.asyncio
async def test_copy_narrows_both_queries_to_the_requested_lines() -> None:
    world = _World()
    source = world.model_layer()
    wanted = uuid.uuid4()
    world.session = _Session([[world.part.id], [_row(wanted, source, "text")], []])

    await world.copy(source, line_ids=[wanted])

    affected, rows = (str(statement) for statement in world.session.statements[:2])
    assert "line_transcriptions.line_id IN" in affected
    assert "line_transcriptions.line_id IN" in rows
