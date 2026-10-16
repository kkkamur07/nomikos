"""Per-part publishing and whole-document export selection against in-memory fakes.

Both read the parts ``require_document`` already loaded. Publishing is owner-only and
validates every id against the document before flipping anything; the export selection
numbers pages by position and narrows to reviewed pages on request.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from uuid import UUID

import pytest

from backend.core.exceptions import AccessDeniedError, NotFoundError
from backend.document.application.document_access import DocumentContext
from backend.document.application.document_export_selection import DocumentExportSelector
from backend.document.application.part_service import DocumentPartService
from backend.document.infrastructure.orm_models import Document, DocumentPart


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _Access:
    def __init__(self, project: object, document: Document) -> None:
        self._context = DocumentContext(project=project, document=document)

    async def require_document(self, *_args: object, **_kwargs: object) -> DocumentContext:
        return self._context


def _document(orders: list[int], reviewed: list[bool] | None = None) -> Document:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    flags = reviewed or [False] * len(orders)
    document.parts = [
        DocumentPart(
            id=uuid.uuid4(),
            document_id=document.id,
            order=order,
            image_key=f"parts/{order}.webp",
            reviewed=flag,
            published=False,
        )
        for order, flag in zip(orders, flags, strict=True)
    ]
    return document


_OWNER = SimpleNamespace(id=uuid.uuid4())


async def _publish(
    document: Document, updates: list[tuple[UUID, bool]], *, user: object = _OWNER
) -> tuple[list[DocumentPart], _Session]:
    project = SimpleNamespace(owner_id=_OWNER.id)
    service = DocumentPartService(access=_Access(project, document))
    session = _Session()
    parts = await service.update_parts_published(
        session, user, document.project_id, document.id, updates=updates
    )
    return parts, session


@pytest.mark.asyncio
async def test_owner_publishes_a_batch_and_gets_parts_in_order() -> None:
    document = _document([2, 0, 1])
    by_order = {part.order: part for part in document.parts}

    parts, session = await _publish(document, [(by_order[0].id, True), (by_order[2].id, True)])

    assert [part.order for part in parts] == [0, 1, 2]
    assert [part.published for part in parts] == [True, False, True]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_a_repeated_id_takes_its_last_value() -> None:
    document = _document([0])
    [part] = document.parts

    await _publish(document, [(part.id, True), (part.id, False)])

    assert part.published is False


@pytest.mark.asyncio
async def test_a_collaborator_cannot_publish() -> None:
    document = _document([0])

    with pytest.raises(AccessDeniedError, match="project owner"):
        await _publish(
            document, [(document.parts[0].id, True)], user=SimpleNamespace(id=uuid.uuid4())
        )

    assert document.parts[0].published is False


@pytest.mark.asyncio
async def test_a_foreign_part_id_flips_nothing() -> None:
    document = _document([0])

    with pytest.raises(NotFoundError, match="Part not found"):
        await _publish(document, [(document.parts[0].id, True), (uuid.uuid4(), True)])

    assert document.parts[0].published is False


async def _select(document: Document, *, reviewed_only: bool = False):
    selector = DocumentExportSelector(access=_Access(object(), document))
    return await selector.select(
        object(), object(), document.project_id, document.id, reviewed_only=reviewed_only
    )


@pytest.mark.asyncio
async def test_selection_numbers_pages_by_position_not_order_value() -> None:
    document = _document([10, 3, 7])

    selection = await _select(document)

    assert [part.order for part in selection.parts] == [3, 7, 10]
    assert [selection.page_numbers[part.id] for part in selection.parts] == [1, 2, 3]
    assert selection.document is document


@pytest.mark.asyncio
async def test_reviewed_only_keeps_the_full_document_numbering() -> None:
    document = _document([0, 1, 2], reviewed=[False, True, False])

    selection = await _select(document, reviewed_only=True)

    [only] = selection.parts
    assert only.order == 1
    assert selection.page_numbers[only.id] == 2
    assert len(selection.page_numbers) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("orders", "reviewed"), [([], []), ([0, 1], [False, False])])
async def test_nothing_to_export_is_a_404(orders: list[int], reviewed: list[bool]) -> None:
    with pytest.raises(NotFoundError, match="No pages to export"):
        await _select(_document(orders, reviewed), reviewed_only=True)
