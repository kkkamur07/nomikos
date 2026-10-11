"""Server-owned reading order: where a new line goes, and the page staying numbered 0..n-1."""

from __future__ import annotations

import uuid

import pytest

from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.application.reading_order import (
    LineShape,
    insertion_index,
    merge_reading_order,
    renumber,
)
from backend.document.infrastructure.orm_models import (
    Document,
    DocumentPart,
    Line,
    LineGeometryKind,
)


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line(name: str, x0: float, y0: float, x1: float, y1: float) -> LineShape:
    """A kraken-style line: a polygon and a baseline along its bottom edge."""
    return LineShape(
        id=name,
        points=_box(x0, y0, x1, y1),
        baseline={"points": [[x0, y1 - 5], [x1, y1 - 5]]},
    )


def _column(prefix: str, x0: float, x1: float, count: int) -> list[LineShape]:
    return [_line(f"{prefix}{i}", x0, 100 + 50 * i, x1, 140 + 50 * i) for i in range(count)]


def _drawn(x0: float, y0: float, x1: float, y1: float) -> LineShape:
    """A hand-drawn rectangle: the editor sends the polygon as its own baseline."""
    points = _box(x0, y0, x1, y1)
    return LineShape(id="new", points=points, baseline={"points": points})


# --- insertion_index ---


def test_a_line_drawn_between_two_lines_goes_between_them():
    page = _column("a", 100, 900, 6)  # bottoms at 140, 190, 240, ...
    # Between a3 (y 250-290) and a4 (y 300-340).
    assert insertion_index(page, _drawn(110, 292, 880, 302)) == 4


def test_a_line_drawn_above_the_first_goes_first():
    page = _column("a", 100, 900, 4)
    assert insertion_index(page, _drawn(100, 20, 900, 60)) == 0


def test_a_line_drawn_below_the_last_goes_last():
    page = _column("a", 100, 900, 4)
    assert insertion_index(page, _drawn(100, 400, 900, 440)) == 4


def test_two_columns_side_by_side_insert_within_the_right_column():
    left = _column("l", 100, 480, 5)
    right = _column("r", 520, 900, 5)
    page = left + right  # reading order: the whole left column, then the right
    # Top of the right column: above r0, and every left line overlaps nothing.
    assert insertion_index(page, _drawn(530, 40, 890, 80)) == 5
    # Between r1 and r2 of the right column, not after l1 in the left one.
    assert insertion_index(page, _drawn(530, 192, 890, 202)) == 7
    # Under the last line of the left column: before the right column starts.
    assert insertion_index(page, _drawn(110, 360, 470, 400)) == 5


def test_a_spread_keeps_the_folios_apart():
    # Two folios side by side with a gutter; the verso is shorter than the recto.
    verso = _column("v", 50, 650, 3)
    recto = _column("r", 750, 1400, 8)
    page = verso + recto
    # Low on the verso, level with recto lines 5 and 6: still the end of the verso.
    assert insertion_index(page, _drawn(60, 360, 640, 400)) == 3
    # Between recto lines 5 and 6.
    assert insertion_index(page, _drawn(760, 392, 1390, 402)) == 3 + 6


def test_a_line_overlapping_no_column_goes_last():
    page = _column("a", 100, 500, 3)
    assert insertion_index(page, _drawn(800, 120, 900, 140)) == 3


def test_a_short_numeral_overlapping_a_line_counts_as_its_column():
    page = _column("a", 100, 900, 3)
    # A 40px numeral at the right end, between a0 and a1.
    assert insertion_index(page, _drawn(850, 143, 890, 147)) == 1


def test_a_line_without_baseline_uses_the_polygon_centre():
    page = [
        LineShape(id="a", points=_box(100, 100, 900, 140), baseline=None),
        LineShape(id="b", points=_box(100, 200, 900, 240), baseline=None),
    ]
    assert insertion_index(page, _drawn(100, 150, 900, 190)) == 1


# --- renumber / merge_reading_order ---


def test_renumber_is_dense_and_keeps_the_sequence():
    lines = [LineShape(id="x", order=7), LineShape(id="y", order=2), LineShape(id="z", order=7)]
    assert renumber(lines) == {"x": 0, "y": 1, "z": 2}


def test_a_kept_manual_line_lands_between_the_model_lines_around_it():
    model = _column("m", 100, 900, 4)
    kept = [_drawn(110, 192, 880, 202)]  # between m1 and m2
    sequence = merge_reading_order(kept, model)
    assert [line.id for line in sequence] == ["m0", "m1", "new", "m2", "m3"]


def test_kept_lines_in_one_gap_keep_their_relative_order():
    model = _column("m", 100, 900, 3)
    upper = LineShape(id="k1", points=_box(100, 141, 900, 145), baseline=None)
    lower = LineShape(id="k2", points=_box(100, 146, 900, 150), baseline=None)
    sequence = merge_reading_order([upper, lower], model)
    assert [line.id for line in sequence] == ["m0", "k1", "k2", "m1", "m2"]


def test_only_kept_lines_keep_their_sequence():
    kept = _column("k", 100, 900, 3)
    assert [line.id for line in merge_reading_order(kept, [])] == ["k0", "k1", "k2"]


# --- LayoutService: create and delete keep the page dense ---


class _Repository:
    def __init__(self, lines: list[Line]) -> None:
        self.lines = lines

    async def lock_part(self, _session, _part_id) -> None:
        pass

    async def list_part_lines(self, _session, _part_id) -> list[Line]:
        return sorted(self.lines, key=lambda line: line.order)

    async def get_line_in_part(self, _session, _part_id, line_id):
        return next((line for line in self.lines if line.id == line_id), None)


class _Session:
    def __init__(self, repository: _Repository) -> None:
        self._repository = repository
        self.added: list[Line] = []
        self.deleted: list[Line] = []

    def add(self, item: Line) -> None:
        self.added.append(item)
        self._repository.lines.append(item)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def commit(self) -> None:
        pass


class _Access:
    def __init__(self, part: DocumentPart) -> None:
        self._part = part

    async def require_part(self, *_args, **_kwargs) -> PartContext:
        document = Document(id=self._part.document_id, name="codex")
        return PartContext(project=object(), document=document, part=self._part)


def _page(count: int) -> tuple[DocumentPart, list[Line]]:
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    lines = []
    for i in range(count):
        shape = _line(f"l{i}", 100, 100 + 50 * i, 900, 140 + 50 * i)
        lines.append(
            Line(
                id=uuid.uuid4(),
                part_id=part.id,
                order=i,
                points=shape.points,
                baseline=shape.baseline,
            )
        )
    return part, lines


async def _create(part, repository, points, order=None) -> Line:
    session = _Session(repository)
    service = LayoutService(documents=repository, access=_Access(part))
    await service.create_part_line(
        session,
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        order=order,
        kind=LineGeometryKind.polygon,
        points=points,
    )
    (created,) = session.added
    return created


@pytest.mark.asyncio
async def test_create_without_order_numbers_the_line_by_position_and_shifts_the_rest():
    part, lines = _page(6)
    repository = _Repository(list(lines))
    created = await _create(part, repository, _box(110, 292, 880, 302))

    assert created.order == 4
    assert [line.order for line in lines] == [0, 1, 2, 3, 5, 6]


@pytest.mark.asyncio
async def test_create_with_an_explicit_order_inserts_there_and_stays_dense():
    part, lines = _page(4)
    lines[2].order, lines[3].order = 5, 9  # a page left gapped by the old delete
    repository = _Repository(list(lines))
    created = await _create(part, repository, _box(0, 0, 10, 10), order=5)

    assert created.order == 2
    assert [line.order for line in lines] == [0, 1, 3, 4]


@pytest.mark.asyncio
async def test_delete_closes_the_gap():
    part, lines = _page(5)
    repository = _Repository(list(lines))
    service = LayoutService(documents=repository, access=_Access(part))
    session = _Session(repository)

    await service.delete_part_line(
        session, object(), uuid.uuid4(), part.document_id, part.id, lines[1].id
    )

    assert session.deleted == [lines[1]]
    assert [line.order for line in lines if line is not lines[1]] == [0, 1, 2, 3]
