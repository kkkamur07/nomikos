"""SegmentMergeService.apply_sync against a fake session.

A re-segment must replace only machine geometry nobody touched. These tests pin the
protected-line rules (manual geometry, approved text, pairing), the block rule, the
covered-redraw skip, the part lock, and the commit switch.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from backend.core.exceptions import NotFoundError
from backend.document.application.segment_merge_service import SegmentMergeService
from backend.document.infrastructure.orm_models import (
    Block,
    DocumentPart,
    Line,
    LineGeometryKind,
    LineSource,
)
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class _Result:
    def __init__(self, part: DocumentPart | None) -> None:
        self._part = part

    def scalar_one_or_none(self) -> DocumentPart | None:
        return self._part


class _Session:
    """Answers the part load, then the ground-truth query, then the pairing query."""

    def __init__(
        self,
        part: DocumentPart | None,
        transcribed: list[UUID] | None = None,
        paired: list[UUID | None] | None = None,
    ) -> None:
        self._part = part
        self._scalars = [list(transcribed or []), list(paired or [])]
        self.statements: list[object] = []
        self.scalar_queries = 0
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.flushes = 0
        self.commits = 0

    def execute(self, statement: object) -> _Result:
        self.statements.append(statement)
        return _Result(self._part)

    def scalars(self, statement: object) -> list:
        self.statements.append(statement)
        self.scalar_queries += 1
        return self._scalars.pop(0)

    def add(self, item: object) -> None:
        if isinstance(item, Block) and item.id is None:
            item.id = uuid.uuid4()
        self.added.append(item)

    def delete(self, item: object) -> None:
        self.deleted.append(item)

    def flush(self) -> None:
        self.flushes += 1

    def commit(self) -> None:
        self.commits += 1


def _part(lines: list[Line], blocks: list[Block] | None = None) -> DocumentPart:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="p.webp")
    part.lines = lines
    part.blocks = blocks or []
    return part


def _line(points: list[list[float]], *, manual: bool = False) -> Line:
    return Line(
        id=uuid.uuid4(),
        order=0,
        kind=LineGeometryKind.polygon,
        points=points,
        baseline={},
        source=LineSource.manual if manual else LineSource.kraken,
        manual_geometry=manual,
    )


def _canonical_line(
    external_id: str, points: list[list[float]], block: str | None = None, order: int = 0
) -> CanonicalLine:
    return CanonicalLine(
        external_id=external_id,
        order=order,
        block_external_id=block,
        baseline={"points": points[:2]},
        mask={"points": points},
        points=points,
        kraken_ceiling=[[1.0, 1.0]],
        source_metadata={"engine": "blla"},
    )


def _added_lines(session: _Session) -> list[Line]:
    return [item for item in session.added if isinstance(item, Line)]


def test_unknown_part_is_a_404() -> None:
    with pytest.raises(NotFoundError, match="Part not found"):
        SegmentMergeService().apply_sync(
            _Session(None),
            part_id=uuid.uuid4(),
            canonical_segment=CanonicalSegmentResult(),
            job_id=uuid.uuid4(),
        )


def test_the_part_is_loaded_under_a_row_lock() -> None:
    session = _Session(_part([]))

    SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
    )

    load = session.statements[0]
    assert getattr(load, "_for_update_arg", None) is not None


def test_an_empty_page_skips_the_protection_queries() -> None:
    session = _Session(_part([]))

    summary = SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
    )

    assert session.scalar_queries == 0
    assert summary.pruned_lines == 0
    assert session.commits == 1


def test_manual_approved_and_paired_lines_survive_and_the_rest_are_pruned() -> None:
    manual = _line(_box(0, 0, 100, 20), manual=True)
    approved = _line(_box(0, 40, 100, 60))
    paired = _line(_box(0, 80, 100, 100))
    untouched = _line(_box(0, 120, 100, 140))
    session = _Session(
        _part([manual, approved, paired, untouched]),
        transcribed=[approved.id],
        paired=[paired.id],
    )

    summary = SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
    )

    assert session.deleted == [untouched]
    assert summary.preserved_manual_lines == 1
    assert summary.preserved_transcribed_lines == 2
    assert summary.pruned_lines == 1


def test_a_manual_line_with_approved_text_counts_once_as_manual() -> None:
    line = _line(_box(0, 0, 100, 20), manual=True)
    session = _Session(_part([line]), transcribed=[line.id], paired=[line.id])

    summary = SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
    )

    assert (summary.preserved_manual_lines, summary.preserved_transcribed_lines) == (1, 0)


def test_only_machine_blocks_are_deleted() -> None:
    drawn = Block(id=uuid.uuid4(), order=0, box={}, manual_geometry=True)
    machine = Block(id=uuid.uuid4(), order=1, box={}, manual_geometry=False)
    session = _Session(_part([], [drawn, machine]))

    SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
    )

    assert session.deleted == [machine]


def test_fresh_lines_are_added_as_kraken_geometry_in_their_block() -> None:
    part = _part([])
    session = _Session(part)
    job_id = uuid.uuid4()
    segment = CanonicalSegmentResult(
        blocks=[CanonicalBlock(external_id="b1", order=0, box={"x": 1})],
        lines=[
            _canonical_line("l1", _box(0, 0, 100, 20), block="b1", order=0),
            _canonical_line("l2", _box(0, 40, 100, 60), block="missing", order=1),
            _canonical_line("l3", _box(0, 80, 100, 100), order=2),
        ],
    )

    summary = SegmentMergeService().apply_sync(
        session, part_id=part.id, canonical_segment=segment, job_id=job_id
    )

    [block] = [item for item in session.added if isinstance(item, Block)]
    assert (block.part_id, block.order, block.box, block.manual_geometry) == (
        part.id,
        0,
        {"x": 1},
        False,
    )
    first, second, third = _added_lines(session)
    assert first.block_id == block.id
    assert second.block_id is None
    assert third.block_id is None
    assert [line.order for line in (first, second, third)] == [0, 1, 2]
    assert first.part_id == part.id
    assert first.source is LineSource.kraken
    assert first.manual_geometry is False
    assert first.kraken_ceiling == [[1.0, 1.0]]
    assert first.mask == {"points": _box(0, 0, 100, 20)}
    assert first.source_metadata == {"engine": "blla", "external_id": "l1", "job_id": str(job_id)}
    assert summary.blocks_count == 1
    assert summary.lines_count == 3
    assert summary.added_lines == 3
    assert summary.skipped_covered_lines == 0


def test_a_redraw_of_a_kept_line_is_skipped_but_counted() -> None:
    kept = _line(_box(100, 100, 900, 160), manual=True)
    session = _Session(_part([kept]))
    segment = CanonicalSegmentResult(
        lines=[
            _canonical_line("same-ink", _box(96, 104, 905, 158)),
            _canonical_line("new-ink", _box(100, 300, 900, 360)),
        ]
    )

    summary = SegmentMergeService().apply_sync(
        session, part_id=uuid.uuid4(), canonical_segment=segment, job_id=uuid.uuid4()
    )

    assert [line.source_metadata["external_id"] for line in _added_lines(session)] == ["new-ink"]
    assert summary.added_lines == 1
    assert summary.skipped_covered_lines == 1
    assert summary.lines_count == 2


def test_a_pruned_line_does_not_block_its_redraw() -> None:
    pruned = _line(_box(100, 100, 900, 160))
    session = _Session(_part([pruned]))
    segment = CanonicalSegmentResult(lines=[_canonical_line("again", _box(100, 100, 900, 160))])

    summary = SegmentMergeService().apply_sync(
        session, part_id=uuid.uuid4(), canonical_segment=segment, job_id=uuid.uuid4()
    )

    assert summary.added_lines == 1
    assert summary.skipped_covered_lines == 0


def test_commit_false_leaves_the_transaction_to_the_caller() -> None:
    session = _Session(_part([_line(_box(0, 0, 10, 10))]))

    SegmentMergeService().apply_sync(
        session,
        part_id=uuid.uuid4(),
        canonical_segment=CanonicalSegmentResult(),
        job_id=uuid.uuid4(),
        commit=False,
    )

    assert session.commits == 0
    # The run record, then deletes and new blocks, are flushed so the inserts that
    # follow see them.
    assert session.flushes == 3
