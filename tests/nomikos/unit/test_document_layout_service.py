"""LayoutService bulk paths against in-memory fakes: the line replace and the layout reset.

``replace_part_lines`` validates the whole payload (size, id uniqueness, server-generated
ids, block membership) before it deletes or writes anything, then diffs the payload
against the page: unlisted lines go, listed ones are updated in place, unknown ones are
inserted. ``reset_part_layout`` hands geometry back to the model by clearing
``manual_geometry``.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    Line,
    LineGeometryKind,
    LineSource,
    Transcription,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User

_POINTS = [[10.0, 10.0], [40.0, 10.0], [40.0, 25.0], [10.0, 25.0]]
_OTHER_POINTS = [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]


class _Session:
    def __init__(self, repository: _Repository) -> None:
        self._repository = repository
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    def add(self, item: object) -> None:
        self.added.append(item)
        if isinstance(item, Line):
            self._repository.lines.append(item)

    async def commit(self) -> None:
        self.commits += 1

    async def delete(self, item: object) -> None:
        self.deleted.append(item)
        if isinstance(item, Line):
            self._repository.lines.remove(item)


class _Repository:
    def __init__(self, part: DocumentPart, lines: list[Line], blocks: list[Block]) -> None:
        self._part = part
        self.lines = list(lines)
        self.blocks = list(blocks)
        self.locks: list[UUID] = []
        self.block_queries = 0

    async def lock_part(self, _session: object, part_id: UUID) -> None:
        self.locks.append(part_id)

    async def list_part_lines(self, _session: object, _part_id: UUID) -> list[Line]:
        return list(self.lines)

    async def list_part_blocks(self, _session: object, part_id: UUID) -> list[Block]:
        self.block_queries += 1
        return [block for block in self.blocks if block.part_id == part_id]


class _Access:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._context = PartContext(project=object(), document=document, part=part)
        self.calls = 0

    async def require_part(self, *_args: object, **_kwargs: object) -> PartContext:
        self.calls += 1
        return self._context


class _GroundTruth:
    def __init__(self, layer: Transcription) -> None:
        self._layer = layer
        self.writes: list[tuple[Line, Transcription, str | None]] = []

    async def layer_for(self, _session: object, _document: Document) -> Transcription:
        return self._layer

    async def write(
        self, _session: object, line: Line, layer: Transcription, text: str | None
    ) -> None:
        self.writes.append((line, layer, text))


class _World:
    def __init__(self, lines: list[Line] | None = None, blocks: list[Block] | None = None):
        self.document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
        self.part = DocumentPart(
            id=uuid.uuid4(), document_id=self.document.id, order=0, image_key="page.webp"
        )
        self.layer = Transcription(
            id=uuid.uuid4(),
            document_id=self.document.id,
            name="Ground truth",
            kind=TranscriptionKind.ground_truth,
        )
        self.repository = _Repository(self.part, lines or [], blocks or [])
        self.access = _Access(self.document, self.part)
        self.ground_truth = _GroundTruth(self.layer)
        self.session = _Session(self.repository)
        self.service = LayoutService(
            documents=self.repository,
            access=self.access,
            ground_truth=self.ground_truth,
        )

    def line(self, order: int, **fields: object) -> Line:
        line = Line(
            id=uuid.uuid4(),
            part_id=self.part.id,
            order=order,
            kind=LineGeometryKind.polygon,
            points=_POINTS,
            baseline={"points": _POINTS},
            mask={"points": _POINTS},
            source=LineSource.kraken,
            manual_geometry=False,
            **fields,
        )
        self.repository.lines.append(line)
        return line

    async def replace(self, lines: list[dict], *, allow_new_ids: bool = False) -> list[Line]:
        return await self.service.replace_part_lines(
            self.session,
            # A real user row: the service stamps ``updated_by`` with its id.
            User(id=uuid.uuid4(), email="editor@example.test"),
            self.document.project_id,
            self.document.id,
            self.part.id,
            lines=lines,
            allow_new_ids=allow_new_ids,
        )

    async def reset(self, line_ids: list[UUID] | None = None) -> tuple[list[Block], list[Line]]:
        return await self.service.reset_part_layout(
            self.session,
            object(),
            self.document.project_id,
            self.document.id,
            self.part.id,
            line_ids=line_ids,
        )


def _payload(order: int, line_id: UUID | None = None, **fields: object) -> dict:
    data: dict = {"order": order, "points": _POINTS, **fields}
    if line_id is not None:
        data["id"] = line_id
    return data


@pytest.mark.asyncio
async def test_replace_rejects_an_oversized_payload_before_authorizing() -> None:
    world = _World()
    too_many = [_payload(order) for order in range(10_001)]

    with pytest.raises(ValidationError, match="more than 10000"):
        await world.replace(too_many)

    assert world.access.calls == 0
    assert world.repository.locks == []


@pytest.mark.asyncio
async def test_replace_rejects_duplicate_ids_without_writing() -> None:
    world = _World()
    kept = world.line(0)

    with pytest.raises(ValidationError, match="unique"):
        await world.replace([_payload(0, kept.id), _payload(1, kept.id)])

    assert world.session.deleted == []
    assert world.session.commits == 0
    assert world.repository.locks == [world.part.id]


@pytest.mark.asyncio
async def test_replace_rejects_client_chosen_ids_for_new_lines() -> None:
    world = _World()
    world.line(0)

    with pytest.raises(ValidationError, match="server-generated"):
        await world.replace([_payload(0, uuid.uuid4())])

    assert world.session.deleted == []
    assert world.session.added == []


@pytest.mark.asyncio
async def test_replace_keeps_a_client_id_for_a_new_line_when_allowed() -> None:
    world = _World()
    restored_id = uuid.uuid4()

    lines = await world.replace([_payload(0, restored_id)], allow_new_ids=True)

    assert [line.id for line in lines] == [restored_id]
    assert world.session.added == lines


@pytest.mark.asyncio
async def test_replace_diffs_the_page_deleting_updating_and_inserting() -> None:
    world = _World()
    kept = world.line(0, source_metadata={"engine": "kraken"}, kraken_ceiling=[[1.0, 2.0]])
    dropped = world.line(1)

    lines = await world.replace(
        [
            _payload(5, kept.id, points=_OTHER_POINTS),
            _payload(6, approved_text="καὶ"),
        ]
    )

    assert world.session.deleted == [dropped]
    assert world.session.commits == 1
    updated, inserted = lines
    # The existing row is updated in place, not replaced. The server makes the
    # client's orders dense, so 5 and 6 become 0 and 1.
    assert updated is kept
    assert (updated.order, updated.points) == (0, _OTHER_POINTS)
    # Omitted optional fields keep what the row had.
    assert updated.source is LineSource.kraken
    assert updated.manual_geometry is False
    assert updated.source_metadata == {"engine": "kraken"}
    assert updated.kraken_ceiling == [[1.0, 2.0]]
    # A new row takes the schema defaults and a fresh geometry from its points.
    assert inserted.part_id == world.part.id
    assert inserted.order == 1
    assert inserted.kind is LineGeometryKind.polygon
    assert inserted.source is LineSource.manual
    assert inserted.manual_geometry is True
    assert inserted.baseline is not None
    assert [(line, text) for line, _layer, text in world.ground_truth.writes] == [
        (kept, None),
        (inserted, "καὶ"),
    ]
    assert {layer.id for _line, layer, _text in world.ground_truth.writes} == {world.layer.id}


@pytest.mark.asyncio
async def test_replace_applies_sent_optional_fields_including_explicit_nulls() -> None:
    world = _World()
    block = Block(id=uuid.uuid4(), part_id=world.part.id, order=0, box={})
    world.repository.blocks.append(block)
    line = world.line(0, block_id=block.id, source_metadata={"engine": "kraken"})

    [updated] = await world.replace(
        [
            _payload(
                0,
                line.id,
                block_id=None,
                source_metadata=None,
                kraken_ceiling=[[3.0, 4.0]],
                source=LineSource.model,
                kind=LineGeometryKind.rectangle,
            )
        ]
    )

    assert updated.block_id is None
    assert updated.source_metadata is None
    assert updated.kraken_ceiling == [[3.0, 4.0]]
    assert updated.source is LineSource.model
    assert updated.kind is LineGeometryKind.rectangle
    assert updated.manual_geometry is False


@pytest.mark.asyncio
async def test_replace_accepts_a_plain_string_source_from_a_snapshot() -> None:
    world = _World()

    [line] = await world.replace([_payload(0, source="manual")])

    assert line.manual_geometry is True


@pytest.mark.asyncio
async def test_replace_checks_every_block_with_one_query() -> None:
    world = _World()
    block = Block(id=uuid.uuid4(), part_id=world.part.id, order=0, box={})
    world.repository.blocks.append(block)

    lines = await world.replace([_payload(order, block_id=block.id) for order in range(25)])

    # One query checks every block, one more ranks the regions for reading order.
    assert world.repository.block_queries == 2
    assert {line.block_id for line in lines} == {block.id}


@pytest.mark.asyncio
async def test_replace_skips_the_block_check_when_no_line_names_a_block() -> None:
    world = _World()

    await world.replace([_payload(0, block_id=None)])

    # Only the reading-order ranking reads the blocks; there is nothing to check.
    assert world.repository.block_queries == 1


@pytest.mark.asyncio
async def test_replace_rejects_a_block_from_another_part_before_deleting() -> None:
    world = _World()
    foreign = Block(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, box={})
    world.repository.blocks.append(foreign)
    doomed = world.line(0)

    with pytest.raises(NotFoundError, match="Block not found"):
        await world.replace([_payload(0, block_id=foreign.id)])

    assert doomed in world.repository.lines
    assert world.session.deleted == []
    assert world.session.commits == 0


@pytest.mark.asyncio
async def test_reset_without_ids_clears_every_line_and_block() -> None:
    world = _World()
    block = Block(id=uuid.uuid4(), part_id=world.part.id, order=0, box={}, manual_geometry=True)
    world.repository.blocks.append(block)
    first = world.line(0, block_id=block.id)
    second = world.line(1)
    first.manual_geometry = second.manual_geometry = True

    blocks, lines = await world.reset()

    assert world.repository.locks == [world.part.id]
    assert world.session.commits == 1
    assert blocks == [block]
    assert block.manual_geometry is False
    assert [line.manual_geometry for line in lines] == [False, False]


@pytest.mark.asyncio
async def test_reset_with_ids_clears_only_those_lines_and_no_block() -> None:
    world = _World()
    block = Block(id=uuid.uuid4(), part_id=world.part.id, order=0, box={}, manual_geometry=True)
    world.repository.blocks.append(block)
    chosen = world.line(0)
    untouched = world.line(1)
    chosen.manual_geometry = untouched.manual_geometry = True

    await world.reset([chosen.id])

    assert chosen.manual_geometry is False
    assert untouched.manual_geometry is True
    assert block.manual_geometry is True


@pytest.mark.asyncio
async def test_reset_with_an_unknown_id_is_a_404_without_commit() -> None:
    world = _World()
    line = world.line(0)
    line.manual_geometry = True

    with pytest.raises(NotFoundError, match="Line not found"):
        await world.reset([line.id, uuid.uuid4()])

    assert line.manual_geometry is True
    assert world.session.commits == 0
