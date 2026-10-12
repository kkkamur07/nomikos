"""Per-character confidences flow from model output to the API.

The inference adapters report a score per character, but until the merge
service stores them, the ORM keeps them, the human-edit paths clear them, and
the response builder serves them, the editor paints every character with the
line score. Each test below names the drop it guards: remove the behaviour and
the test goes red.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from backend.document.api.line_responses import line_transcription_response
from backend.document.application.ground_truth import GroundTruthText
from backend.document.application.transcribe_merge_service import (
    TranscribeMergeService,
    character_confidences_for_storage,
)
from backend.document.application.transcription_service import TranscriptionService
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)
from nomikos_inference.contracts.transcribe import (
    CharacterConfidence,
    TranscribeRunResponse,
)


def _output(text: str, *scores: float) -> TranscribeRunResponse:
    return TranscribeRunResponse(
        text=text,
        confidence=0.8,
        character_confidences=[
            CharacterConfidence(char=char, confidence=score)
            for char, score in zip(text, scores, strict=True)
        ],
    )


def _model_layer() -> Transcription:
    return Transcription(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        name="Model transcription test",
        kind=TranscriptionKind.model,
    )


def _stored_row(
    text: str,
    scores: list[float] | None,
    *,
    layer: Transcription,
) -> LineTranscription:
    row = LineTranscription(
        id=uuid.uuid4(),
        line_id=uuid.uuid4(),
        transcription_id=layer.id,
        text=text,
        confidence=0.8,
        character_confidences=scores,
    )
    row.transcription = layer
    return row


# --- Storage helper ---


def test_storage_keeps_scores_rounded_to_three_decimals() -> None:
    stored = character_confidences_for_storage(_output("ab", 0.91234, 0.5))

    assert stored == [0.912, 0.5]


def test_storage_returns_null_when_scores_do_not_describe_the_text() -> None:
    output = _output("ab", 0.9, 0.8)
    output.text = "abc"

    assert character_confidences_for_storage(output) is None


# --- Merge service ---


class _SyncSession:
    def __init__(self, part: DocumentPart) -> None:
        self._part = part
        self.added: list[object] = []
        self.commits = 0

    def execute(self, _statement, *_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(scalar_one_or_none=lambda: self._part)

    def add(self, item: object) -> None:
        self.added.append(item)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.commits += 1


def _part_and_line() -> tuple[DocumentPart, Line]:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="p")
    line = Line(id=uuid.uuid4(), part_id=part.id, order=0, baseline={})
    return part, line


def test_merge_stores_the_floats_with_the_model_transcription() -> None:
    """Without this the editor never sees the real per-character values."""
    part, line = _part_and_line()
    session = _SyncSession(part)
    service = TranscribeMergeService()

    service.apply_sync(
        session,  # type: ignore[arg-type]
        document_id=part.document_id,
        part_id=part.id,
        job_id=uuid.uuid4(),
        lines_with_output=[(line, _output("ab", 0.91234, 0.5))],
    )

    rows = [item for item in session.added if isinstance(item, LineTranscription)]
    assert len(rows) == 1
    assert rows[0].text == "ab"
    assert rows[0].character_confidences == [0.912, 0.5]
    assert session.commits == 1


def test_merge_stores_null_and_still_succeeds_on_length_mismatch() -> None:
    """A worker's misaligned scores must never fail the job or poison the row."""
    part, line = _part_and_line()
    session = _SyncSession(part)
    output = _output("ab", 0.9, 0.8)
    output.text = "abc"

    summary = TranscribeMergeService().apply_sync(
        session,  # type: ignore[arg-type]
        document_id=part.document_id,
        part_id=part.id,
        job_id=uuid.uuid4(),
        lines_with_output=[(line, output)],
    )

    rows = [item for item in session.added if isinstance(item, LineTranscription)]
    assert len(rows) == 1
    assert rows[0].character_confidences is None
    assert summary["lines"][0]["text"] == "abc"
    assert session.commits == 1


# --- Human edits clear the scores ---


def _line_with(*rows: LineTranscription) -> Line:
    line = Line(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, baseline={})
    line.transcriptions = list(rows)
    return line


class _BareSession:
    async def commit(self) -> None:
        pass

    async def refresh(self, _item: object) -> None:
        pass


@pytest.mark.asyncio
async def test_ground_truth_rewrite_clears_character_confidences() -> None:
    """A human rewrite leaves no model scores attached to the new text."""
    layer = Transcription(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )
    row = LineTranscription(
        transcription_id=layer.id,
        text="old",
        confidence=0.42,
        character_confidences=[0.4, 0.2, 0.9],
    )
    line = _line_with(row)

    await GroundTruthText().write(
        _BareSession(),  # type: ignore[arg-type]
        line,
        layer,
        "new",
    )

    assert (row.text, row.confidence, row.character_confidences) == ("new", None, None)


# --- Response builder ---


def test_api_pairs_scores_with_combining_marks_by_code_point() -> None:
    """Syriac seyame and Greek accents are code points of their own in storage."""
    layer = _model_layer()
    text = "ܡ\u0308ά"
    row = _stored_row(text, [0.95, 0.2, 0.88], layer=layer)

    response = line_transcription_response(row)

    assert response.character_confidences is not None
    assert [entry.char for entry in response.character_confidences] == list(text)
    assert [entry.confidence for entry in response.character_confidences] == [0.95, 0.2, 0.88]


def test_api_returns_null_for_old_rows_without_scores() -> None:
    layer = _model_layer()
    row = _stored_row("abc", None, layer=layer)

    assert line_transcription_response(row).character_confidences is None


def test_api_returns_null_when_scores_no_longer_match_the_text() -> None:
    layer = _model_layer()
    row = _stored_row("abc", [0.9, 0.8], layer=layer)

    assert line_transcription_response(row).character_confidences is None


# --- Copy to ground truth and single-line human edits clear the scores ---


class _StubAccess:
    def __init__(self, document: Document) -> None:
        self._document = document

    async def require_document(self, *_args, **_kwargs):
        return SimpleNamespace(project=object(), document=self._document, part=None)


class _StubGroundTruth:
    def __init__(self, layer: Transcription) -> None:
        self._layer = layer

    async def layer_for(self, _session, _document) -> Transcription:
        return self._layer


class _StubRepository:
    def __init__(self, *, source_layer: Transcription | None = None) -> None:
        self._source_layer = source_layer

    async def lock_part(self, _session, _part_id) -> None:
        pass

    async def lock_parts(self, _session, _part_ids) -> None:
        pass

    async def get_transcription_in_document(self, _session, _document_id, _transcription_id):
        return self._source_layer


class _ServiceSession:
    """Queued SELECT results plus a record of added rows."""

    def __init__(self, queued: list[list]) -> None:
        self._queued = list(queued)
        self.added: list[object] = []
        self.commits = 0

    async def execute(self, _statement, *_args, **_kwargs):
        items = self._queued.pop(0)
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: items),
            scalar_one_or_none=lambda: items[0] if items else None,
        )

    def add(self, item: object) -> None:
        self.added.append(item)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, _item: object) -> None:
        pass


def _ground_truth_layer(document_id: uuid.UUID) -> Transcription:
    return Transcription(
        id=uuid.uuid4(),
        document_id=document_id,
        name="Ground truth",
        kind=TranscriptionKind.ground_truth,
    )


@pytest.mark.asyncio
async def test_copy_to_ground_truth_writes_text_without_scores() -> None:
    """Approving a model line must not carry its scores onto the ground truth."""
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="d")
    source = Transcription(
        id=uuid.uuid4(),
        document_id=document.id,
        name="Model",
        kind=TranscriptionKind.model,
    )
    ground_truth = _ground_truth_layer(document.id)
    line_id = uuid.uuid4()
    source_row = SimpleNamespace(line_id=line_id, text="ab")
    target = LineTranscription(
        line_id=line_id,
        transcription_id=ground_truth.id,
        text="old",
        confidence=0.5,
        character_confidences=[0.5, 0.4, 0.3],
    )
    repository = _StubRepository(source_layer=source)
    service = TranscriptionService(
        documents=repository,  # type: ignore[arg-type]
        access=_StubAccess(document),  # type: ignore[arg-type]
        ground_truth=_StubGroundTruth(ground_truth),  # type: ignore[arg-type]
    )
    session = _ServiceSession([[uuid.uuid4()], [source_row], [target]])

    await service.copy_to_ground_truth(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),
        document.project_id,
        document.id,
        source.id,
    )

    assert (target.text, target.confidence, target.character_confidences) == ("ab", None, None)


@pytest.mark.asyncio
async def test_patch_ground_truth_line_text_clears_scores() -> None:
    """Typing over a line must detach the scores that described the old text."""
    document = Document(id=uuid.uuid4(), project_id=uuid.uuid4(), name="d")
    ground_truth = _ground_truth_layer(document.id)
    line = Line(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, baseline={})
    row = LineTranscription(
        line_id=line.id,
        transcription_id=ground_truth.id,
        text="ab",
        confidence=0.5,
        character_confidences=[0.5, 0.4],
    )

    class _Repo(_StubRepository):
        async def get_transcription_in_document(self, _s, _d, _t):
            return ground_truth

        async def get_line_in_document(self, _s, _d, _lid):
            return line

    service = TranscriptionService(
        documents=_Repo(),  # type: ignore[arg-type]
        access=_StubAccess(document),  # type: ignore[arg-type]
        ground_truth=_StubGroundTruth(ground_truth),  # type: ignore[arg-type]
    )
    session = _ServiceSession([[row]])

    updated = await service.patch_ground_truth_line_text(
        session,  # type: ignore[arg-type]
        SimpleNamespace(),
        document.project_id,
        document.id,
        ground_truth.id,
        line.id,
        text="ac",
    )

    assert (updated.text, updated.confidence, updated.character_confidences) == (
        "ac",
        None,
        None,
    )
