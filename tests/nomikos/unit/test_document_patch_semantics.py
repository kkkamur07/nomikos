"""PATCH bodies must distinguish an omitted field from an explicit null.

The line/block patch handlers dropped every ``None`` value, so a client could never clear
``lines.block_id`` (a nullable column) — the update was silently ignored and the response
still carried the old block. Fields backing NOT NULL columns reject an explicit null the
same way ``DocumentUpdateRequest`` already did.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.document.api.schemas import BlockPatchRequest, LinePatchRequest
from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.infrastructure.orm_models import Block, Document, DocumentPart, Line
from backend.users.infrastructure.orm_models import User


def _user() -> User:
    """A real user row: the service stamps ``updated_by`` with its id."""
    return User(id=uuid.uuid4(), email="editor@example.test")


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _item: object) -> None:
        pass

    async def delete(self, _item: object) -> None:
        pass


class _StubAccess:
    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._document = document
        self._part = part

    async def require_part(self, *_args, **_kwargs) -> PartContext:
        return PartContext(project=object(), document=self._document, part=self._part)


class _StubRepository:
    """Only the two row lookups ``patch_part_line`` performs."""

    def __init__(self, line: Line | None, block: Block | None) -> None:
        self._line = line
        self._block = block
        self.locks = 0

    async def lock_part(self, _session, _part_id) -> None:
        self.locks += 1

    async def get_line_in_part(self, _session, _part_id, _line_id):
        return self._line

    async def get_block_in_part(self, _session, _part_id, _block_id):
        return self._block


def _service_with(monkeypatch, *, part: DocumentPart, line: Line | None, block: Block | None):
    document = Document(id=part.document_id, name="codex")
    service = LayoutService(
        documents=_StubRepository(line, block),
        access=_StubAccess(document, part),
    )
    return service, document


# --- Request schema: an explicit null is refused for NOT NULL columns ---


@pytest.mark.parametrize("field", ["order", "baseline", "points"])
def test_line_patch_rejects_null_for_not_null_columns(field: str) -> None:
    with pytest.raises(PydanticValidationError):
        LinePatchRequest.model_validate({field: None})


@pytest.mark.parametrize("field", ["order", "box"])
def test_block_patch_rejects_null_for_not_null_columns(field: str) -> None:
    with pytest.raises(PydanticValidationError):
        BlockPatchRequest.model_validate({field: None})


# --- Handler: an explicit null actually clears the column ---


@pytest.mark.asyncio
async def test_patch_line_clears_block_id_when_explicitly_null(monkeypatch) -> None:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=uuid.uuid4(), baseline={}, order=0)
    service, _ = _service_with(monkeypatch, part=part, line=line, block=None)
    body = LinePatchRequest.model_validate({"block_id": None})

    patched = await service.patch_part_line(
        _Session(),
        _user(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert patched.block_id is None


@pytest.mark.asyncio
async def test_patch_line_leaves_block_id_alone_when_omitted(monkeypatch) -> None:
    block_id = uuid.uuid4()
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=block_id, baseline={}, order=0)
    service, _ = _service_with(monkeypatch, part=part, line=line, block=None)
    body = LinePatchRequest.model_validate({"order": 4})

    patched = await service.patch_part_line(
        _Session(),
        _user(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert patched.block_id == block_id
    assert patched.order == 4


# --- Concurrency: a single-line write holds the part too ---


@pytest.mark.asyncio
async def test_patch_line_locks_the_part(monkeypatch) -> None:
    """One line is still part of a list segment health renumbers wholesale."""
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=None, baseline={}, order=0)
    repository = _StubRepository(line, None)
    service = LayoutService(
        documents=repository,
        access=_StubAccess(Document(id=part.document_id, name="codex"), part),
    )
    body = LinePatchRequest.model_validate({"order": 3})

    await service.patch_part_line(
        _Session(),
        _user(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        line.id,
        **body.model_dump(exclude_unset=True),
    )

    assert repository.locks == 1


@pytest.mark.asyncio
async def test_delete_line_locks_the_part(monkeypatch) -> None:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    line = Line(id=uuid.uuid4(), part_id=part.id, block_id=None, baseline={}, order=0)
    repository = _StubRepository(line, None)
    service = LayoutService(
        documents=repository,
        access=_StubAccess(Document(id=part.document_id, name="codex"), part),
    )

    await service.delete_part_line(
        _Session(), _user(), uuid.uuid4(), part.document_id, part.id, line.id
    )

    assert repository.locks == 1
