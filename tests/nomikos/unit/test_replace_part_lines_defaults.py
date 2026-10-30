"""PUT /lines accepts payloads that omit the optional ``kind``/``source`` fields.

``model_dump(exclude_unset=True)`` strips fields the client never sent, so the service
cannot subscript the dict for them. What it does instead depends on whether the line
already exists: for a new line the schema default is right, but for an existing one an
absent field means "leave it alone", the same reading the service already gives
``block_id``, ``source_metadata`` and ``kraken_ceiling``.
"""

from __future__ import annotations

import uuid

import pytest

from backend.document.api.schemas import LinesReplaceRequest
from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.infrastructure.orm_models import (
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


def _user() -> User:
    """A real user row: the service stamps ``updated_by`` with its id."""
    return User(id=uuid.uuid4(), email="editor@example.test")


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    def add(self, item: object) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1

    async def delete(self, item: object) -> None:
        self.deleted.append(item)


class _Repository:
    def __init__(
        self,
        part: DocumentPart,
        ground_truth: Transcription,
        existing: list[Line] | None = None,
    ) -> None:
        self._part = part
        self._ground_truth = ground_truth
        self.persisted: list[object] = list(existing or [])
        self.locks = 0

    async def lock_part(self, _session, _part_id) -> None:
        self.locks += 1

    async def get_part(self, _session, part_id):
        return self._part if part_id == self._part.id else None

    async def get_ground_truth_transcription(self, _session, _document_id):
        return self._ground_truth

    async def list_part_lines(self, _session, _part_id):
        # Mirrors the repository contract: the rows the service just wrote back.
        return list(self.persisted)

    async def list_part_blocks(self, _session, _part_id):
        return []


class _StubAccess:
    """The bulk replace is already authorized; what is under test is the payload defaults."""

    def __init__(self, document: Document, part: DocumentPart) -> None:
        self._document = document
        self._part = part

    async def require_part(self, *_args, **_kwargs) -> PartContext:
        return PartContext(project=object(), document=self._document, part=self._part)


def _service(
    monkeypatch, existing: list[Line] | None = None
) -> tuple[LayoutService, _Session, Document, DocumentPart, _Repository]:
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="page.webp")
    ground_truth = Transcription(
        id=uuid.uuid4(),
        document_id=document.id,
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )
    repository = _Repository(part, ground_truth, existing)
    service = LayoutService(documents=repository, access=_StubAccess(document, part))
    session = _Session()

    original_add = session.add

    def add(item: object) -> None:
        original_add(item)
        repository.persisted.append(item)

    session.add = add  # type: ignore[method-assign]
    return service, session, document, part, repository


def _payload(**overrides: object) -> list[dict]:
    body = LinesReplaceRequest.model_validate(
        {"lines": [{"order": 0, "points": _POINTS, **overrides}]}
    )
    return [line.model_dump(exclude_unset=True) for line in body.lines]


@pytest.mark.asyncio
async def test_replace_part_lines_defaults_kind_and_source_when_omitted(monkeypatch) -> None:
    service, session, document, part, _repository = _service(monkeypatch)
    payload = _payload()
    # The regression is only reachable when the client omits both optional fields.
    assert "kind" not in payload[0]
    assert "source" not in payload[0]

    lines = await service.replace_part_lines(
        session,
        user=_user(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=payload,
    )

    assert len(lines) == 1
    assert lines[0].kind is LineGeometryKind.polygon
    assert lines[0].source is LineSource.manual
    assert lines[0].manual_geometry is True
    assert lines[0].points == _POINTS
    assert session.commits == 1


@pytest.mark.asyncio
async def test_replace_part_lines_still_honours_explicit_kind_and_source(monkeypatch) -> None:
    service, session, document, part, _repository = _service(monkeypatch)

    lines = await service.replace_part_lines(
        session,
        user=_user(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=_payload(kind="polygon", source="kraken"),
    )

    assert lines[0].source is LineSource.kraken
    assert lines[0].manual_geometry is False


# --- An omitted field on an existing line means "leave it alone" ---
# The editor saves the whole page on every change, and the frontend does not send
# ``source``. Reading the schema default there rewrote every line's provenance.


_KRAKEN_CEILING = [[10.0, 8.0], [40.0, 8.0], [40.0, 9.0], [10.0, 9.0]]


def _kraken_line(part_id: uuid.UUID) -> Line:
    return Line(
        id=uuid.uuid4(),
        part_id=part_id,
        order=0,
        kind=LineGeometryKind.polygon,
        points=_POINTS,
        baseline={"points": _POINTS},
        mask={"points": _POINTS},
        source=LineSource.kraken,
        source_metadata={"model": "kraken-segment-default"},
        kraken_ceiling=_KRAKEN_CEILING,
        manual_geometry=False,
    )


@pytest.mark.asyncio
async def test_omitting_source_leaves_a_kraken_line_kraken(monkeypatch) -> None:
    """Regression: a redraw of one line used to relabel every other line on the page.

    ``source`` fell to the schema default while ``kraken_ceiling`` was preserved, so
    the row ended up claiming a human drew a shape a model had measured. Two comments
    in the service assert the opposite of what it did.
    """
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="page.webp")
    prior = _kraken_line(part.id)
    service, session, document, part, _repository = _service(monkeypatch, existing=[prior])
    prior.part_id = part.id
    payload = _payload(id=str(prior.id))
    assert "source" not in payload[0]
    assert "kind" not in payload[0]

    lines = await service.replace_part_lines(
        session,
        user=_user(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=payload,
    )

    assert lines[0].source is LineSource.kraken
    assert lines[0].manual_geometry is False
    assert lines[0].kraken_ceiling == _KRAKEN_CEILING
    assert lines[0].source_metadata == {"model": "kraken-segment-default"}


@pytest.mark.asyncio
async def test_an_explicit_source_still_overrides_an_existing_line(monkeypatch) -> None:
    """Preserving on absence must not make the field unwritable."""
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="codex")
    part = DocumentPart(id=uuid.uuid4(), document_id=document.id, order=0, image_key="page.webp")
    prior = _kraken_line(part.id)
    service, session, document, part, _repository = _service(monkeypatch, existing=[prior])
    prior.part_id = part.id

    lines = await service.replace_part_lines(
        session,
        user=_user(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=_payload(id=str(prior.id), source="manual"),
    )

    assert lines[0].source is LineSource.manual
    assert lines[0].manual_geometry is True


# --- Concurrency: a bulk replace holds the part while it rewrites the list ---


@pytest.mark.asyncio
async def test_replace_part_lines_locks_the_part_before_it_reads(monkeypatch) -> None:
    """Segment health and re-segmentation both take this row before rewriting a page.

    A bulk replace that skipped it would interleave with either of them and the
    later commit would win on rows it never read. Asserted here rather than in a
    concurrency test because the lock is unobservable from a single caller: the
    only evidence it is still being taken is that it is asked for.
    """
    service, session, document, part, repository = _service(monkeypatch)

    await service.replace_part_lines(
        session,
        user=_user(),
        project_id=document.project_id,
        document_id=document.id,
        part_id=part.id,
        lines=_payload(),
    )

    assert repository.locks == 1
