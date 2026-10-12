"""Server-owned reading order: where a new line goes, and the page staying numbered 0..n-1."""

from __future__ import annotations

import uuid

import pytest

from backend.document.application.document_access import PartContext
from backend.document.application.layout_service import LayoutService
from backend.document.application.reading_order import (
    LineShape,
    block_ranks,
    insertion_index,
    merge_reading_order,
    region_sequence,
    renumber,
)
from backend.document.infrastructure.orm_models import (
    Block,
    Document,
    DocumentPart,
    Line,
    LineGeometryKind,
    LineSource,
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


def test_a_line_overlapping_no_column_goes_after_the_line_on_its_row():
    page = _column("a", 100, 500, 3)
    # Level with a0 (y 100-140), well off to the side: after a0, not last.
    assert insertion_index(page, _drawn(800, 120, 900, 140)) == 1


def test_a_line_sharing_neither_a_column_nor_a_row_goes_last():
    page = _column("a", 100, 500, 3)
    assert insertion_index(page, _drawn(800, 400, 900, 420)) == 3


def test_a_note_in_the_margin_beside_a_line_goes_right_after_it():
    # The bug: a box drawn in the left margin next to line 2 of a 51-line page was
    # numbered 52 (it overlaps no column) or 25 (a ragged line far down the page
    # reaches into the margin far enough to count as its column).
    page = _column("a", 100, 900, 6)
    page[4] = _line("a4", 40, 300, 900, 340)
    # Left of a1 (y 150-190), overlapping nothing near it.
    assert insertion_index(page, _drawn(20, 160, 90, 180)) == 2
    # Narrower still: it shares a column only with a4, 150px further down.
    assert insertion_index(page, _drawn(45, 160, 95, 180)) == 2
    # Right of a1.
    assert insertion_index(page, _drawn(910, 160, 980, 180)) == 2
    # Far out in the margin, 60px off: still the row of a1.
    assert insertion_index(page, _drawn(0, 160, 30, 180)) == 2


def test_a_margin_note_beside_a_spread_attaches_to_its_own_folio():
    verso = _column("v", 50, 650, 4)
    recto = _column("r", 750, 1400, 4)
    page = verso + recto
    # Just right of verso line 1, in the gutter but far from the recto.
    assert insertion_index(page, _drawn(655, 160, 690, 180)) == 2
    # Just left of recto line 1.
    assert insertion_index(page, _drawn(715, 160, 745, 180)) == 4 + 2


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
    def __init__(self, lines: list[Line], blocks: list[Block] | None = None) -> None:
        self.lines = lines
        self.blocks = blocks or []

    async def list_part_blocks(self, _session, _part_id) -> list[Block]:
        return sorted(self.blocks, key=lambda block: block.order)

    async def get_block_in_part(self, _session, _part_id, block_id):
        return next((block for block in self.blocks if block.id == block_id), None)

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

    def add(self, item: Line | Block) -> None:
        self.added.append(item)
        if isinstance(item, Block):
            self._repository.blocks.append(item)
        else:
            self._repository.lines.append(item)

    async def refresh(self, _item: object) -> None:
        pass

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


# --- Regions first (ADR 0011) ---


def _in(block: str | None, lines: list[LineShape]) -> list[LineShape]:
    for line in lines:
        line.block_id = block
    return lines


def test_renumber_puts_each_region_together_and_unblocked_lines_last():
    # A segmenter that interleaved the folios row by row.
    verso = _in("v", _column("v", 50, 650, 3))
    recto = _in("r", _column("r", 750, 1400, 3))
    loose = _in(None, [_line("x", 1500, 50, 1600, 80)])
    interleaved = [loose[0], *[line for pair in zip(verso, recto, strict=True) for line in pair]]
    orders = renumber(interleaved, {"v": 0, "r": 1})
    sequence = sorted(orders, key=orders.get)
    assert sequence == ["v0", "v1", "v2", "r0", "r1", "r2", "x"]


def test_region_order_not_page_position_decides_which_folio_comes_first():
    verso = _in("v", _column("v", 50, 650, 2))
    recto = _in("r", _column("r", 750, 1400, 2))
    sequence = region_sequence(verso + recto, {"r": 0, "v": 1})
    assert [line.id for line in sequence] == ["r0", "r1", "v0", "v1"]


def test_no_regions_means_no_change():
    lines = _column("a", 100, 900, 3)
    assert renumber(lines, {}) == renumber(lines) == {"a0": 0, "a1": 1, "a2": 2}


def test_insertion_in_a_region_compares_only_that_regions_lines():
    # Two regions over the same x-range (a page with a heading region on top): a line
    # of region "b" drawn high up must not land among region "a"'s lines.
    a = _in("a", _column("a", 100, 900, 3))
    b = _in("b", [_line("b0", 100, 400, 900, 440), _line("b1", 100, 450, 900, 490)])
    new = _drawn(110, 120, 880, 130)
    new.block_id = "b"
    assert insertion_index(a + b, new, {"a": 0, "b": 1}) == 3


def test_a_region_line_overlapping_nothing_goes_to_the_end_of_its_region():
    a = _in("a", _column("a", 100, 400, 2))
    b = _in("b", _column("b", 600, 900, 2))
    loose = _in(None, [_line("x", 1000, 50, 1100, 80)])
    new = _drawn(450, 10, 550, 20)
    new.block_id = "a"
    assert insertion_index(a + b + loose, new, {"a": 0, "b": 1}) == 2


def test_merge_keeps_kept_lines_inside_their_region():
    model = _in("v", _column("v", 50, 650, 3)) + _in("r", _column("r", 750, 1400, 3))
    kept = _drawn(760, 192, 1390, 202)  # between r1 and r2
    kept.block_id = "r"
    sequence = merge_reading_order([kept], model, {"v": 0, "r": 1})
    assert [line.id for line in sequence] == ["v0", "v1", "v2", "r0", "r1", "new", "r2"]


def test_block_ranks_follow_the_given_sequence():
    blocks = [Block(id="p", order=0), Block(id="q", order=0), Block(id="s", order=3)]
    assert block_ranks(blocks) == {"p": 0, "q": 1, "s": 2}


def _spread() -> tuple[DocumentPart, list[Line]]:
    """Verso lines at x 50-650, recto at 750-1400, numbered row by row (interleaved)."""
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    lines = []
    for i in range(3):
        for prefix, x0, x1 in (("v", 50, 650), ("r", 750, 1400)):
            shape = _line(f"{prefix}{i}", x0, 100 + 50 * i, x1, 140 + 50 * i)
            lines.append(
                Line(
                    id=uuid.uuid4(),
                    part_id=part.id,
                    order=len(lines),
                    points=shape.points,
                    baseline=shape.baseline,
                )
            )
    return part, lines


async def _create_block(part, repository, box, *, order, assign=True, label=None) -> Block:
    session = _Session(repository)
    service = LayoutService(documents=repository, access=_Access(part))
    return await service.create_part_block(
        session,
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        order=order,
        box={"points": box},
        label=label,
        assign_contained=assign,
    )


@pytest.mark.asyncio
async def test_drawing_a_region_assigns_the_lines_inside_it_and_numbers_them_first():
    part, lines = _spread()
    repository = _Repository(list(lines))
    recto = await _create_block(part, repository, _box(700, 0, 1450, 400), order=0, label="5r")

    assert recto.label == "5r"
    recto_lines = [line for line in lines if line.points[0][0] == 750]
    verso_lines = [line for line in lines if line.points[0][0] == 50]
    assert all(line.block_id == recto.id for line in recto_lines)
    assert all(line.block_id is None for line in verso_lines)
    assert [line.order for line in recto_lines] == [0, 1, 2]
    assert [line.order for line in verso_lines] == [3, 4, 5]


@pytest.mark.asyncio
async def test_a_region_does_not_take_lines_that_already_have_one():
    part, lines = _spread()
    repository = _Repository(list(lines))
    first = await _create_block(part, repository, _box(0, 0, 1450, 400), order=0)
    second = await _create_block(part, repository, _box(700, 0, 1450, 400), order=1)

    assert all(line.block_id == first.id for line in lines)
    assert second.id not in {line.block_id for line in lines}


@pytest.mark.asyncio
async def test_assign_contained_false_leaves_lines_alone():
    part, lines = _spread()
    repository = _Repository(list(lines))
    await _create_block(part, repository, _box(0, 0, 1450, 400), order=0, assign=False)

    assert all(line.block_id is None for line in lines)
    assert [line.order for line in lines] == list(range(6))


@pytest.mark.asyncio
async def test_deleting_a_region_unblocks_its_lines_and_renumbers():
    part, lines = _spread()
    repository = _Repository(list(lines))
    verso = await _create_block(part, repository, _box(0, 0, 700, 400), order=0)
    recto = await _create_block(part, repository, _box(700, 0, 1450, 400), order=1)
    assert [line.order for line in lines] == [0, 3, 1, 4, 2, 5]

    service = LayoutService(documents=repository, access=_Access(part))
    session = _Session(repository)
    await service.delete_part_block(
        session, object(), uuid.uuid4(), part.document_id, part.id, verso.id
    )
    repository.blocks.remove(verso)

    assert session.deleted == [verso]
    assert all(line.block_id in (None, recto.id) for line in lines)
    # The recto comes first now; the former verso lines follow, unblocked, in sequence.
    assert [line.order for line in lines] == [3, 0, 4, 1, 5, 2]


@pytest.mark.asyncio
async def test_moving_a_line_to_another_region_numbers_it_within_that_region():
    part, lines = _spread()
    repository = _Repository(list(lines))
    verso = await _create_block(part, repository, _box(0, 0, 700, 400), order=0)
    await _create_block(part, repository, _box(700, 0, 1450, 400), order=1)
    moved = lines[3]  # r1, the middle recto line

    service = LayoutService(documents=repository, access=_Access(part))
    await service.patch_part_line(
        _Session(repository),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        moved.id,
        block_id=verso.id,
    )

    # It overlaps no verso column but sits level with v1, so it follows v1 inside the
    # verso region, ahead of the recto that it left.
    assert moved.block_id == verso.id
    assert moved.order == 2
    assert sorted(line.order for line in lines) == list(range(6))


@pytest.mark.asyncio
async def test_a_line_drawn_inside_a_region_is_numbered_within_it():
    part, lines = _spread()
    repository = _Repository(list(lines))
    verso = await _create_block(part, repository, _box(0, 0, 700, 400), order=0)
    await _create_block(part, repository, _box(700, 0, 1450, 400), order=1)

    session = _Session(repository)
    service = LayoutService(documents=repository, access=_Access(part))
    await service.create_part_line(
        session,
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        kind=LineGeometryKind.polygon,
        points=_box(60, 300, 640, 340),
        block_id=verso.id,
    )
    (created,) = [item for item in session.added if isinstance(item, Line)]
    assert created.order == 3
    assert sorted(line.order for line in repository.lines) == list(range(7))


# --- Critique round (F1, F4, F7, F8) ---


def _centred(name: str, x0: float, x1: float, centre: float, height: float = 30) -> LineShape:
    """A line of ``height`` centred on ``centre``, baseline a third below the centre."""
    top, bottom = centre - height / 2, centre + height / 2
    return LineShape(
        id=name,
        points=_box(x0, top, x1, bottom),
        baseline={"points": [[x0, centre + height / 3], [x1, centre + height / 3]]},
    )


def _rows(prefix: str, x0: float, x1: float, count: int) -> list[LineShape]:
    return [_centred(f"{prefix}{i}", x0, x1, 100 + 50 * i) for i in range(count)]


def test_a_main_line_below_a_tall_margin_note_stays_in_its_column():
    # A gloss 180px tall, 10px right of the column, numbered after line 3 and spanning
    # lines 3 to 6. A main-column line drawn between lines 5 and 6 belongs before line
    # 6, not right after the gloss (ahead of line 4).
    page = _rows("a", 100, 500, 10)
    gloss = _centred("gloss", 510, 560, 330, height=180)
    page = [*page[:4], gloss, *page[4:]]
    index = insertion_index(page, _centred("new", 100, 500, 375))
    assert page[index].id == "a6"


def test_a_line_in_a_tight_left_column_does_not_jump_to_the_right_column():
    # Two columns with a 15px gutter. A new left line between a5 and a6 goes before a6,
    # not after b6 in the right column.
    page = _rows("a", 100, 300, 10) + _rows("b", 315, 515, 10)
    index = insertion_index(page, _centred("new", 100, 300, 375))
    assert page[index].id == "a6"


def _region_page(count: int = 4) -> tuple[DocumentPart, list[Line], Block]:
    """A segmented page: every line already in one region (not full page)."""
    part, lines = _page(count)
    block = Block(id=uuid.uuid4(), part_id=part.id, order=0, box={"points": _box(50, 50, 950, 400)})
    for line in lines:
        line.block_id = block.id
    return part, lines, block


@pytest.mark.asyncio
async def test_a_line_drawn_without_a_region_inherits_the_region_it_sits_in():
    part, lines, block = _region_page(4)
    repository = _Repository(list(lines), [block])
    # Between l1 (150-190) and l2 (200-240); the editor sends no block_id.
    created = await _create(part, repository, _box(110, 192, 880, 198))

    assert created.block_id == block.id
    assert created.order == 2
    assert [line.order for line in lines] == [0, 1, 3, 4]


@pytest.mark.asyncio
async def test_a_line_drawn_outside_every_region_stays_unblocked():
    part, lines, block = _region_page(2)
    repository = _Repository(list(lines), [block])
    created = await _create(part, repository, _box(100, 600, 900, 640))

    assert created.block_id is None
    assert created.order == 2


class _ScriptSession(_Session):
    """A session that answers the page-script query with ``texts``."""

    def __init__(self, repository: _Repository, texts: list[str]) -> None:
        super().__init__(repository)
        self._texts = texts

    async def scalars(self, _statement):
        return list(self._texts)


async def _draw_region(part, repository, box, texts=()) -> Block:
    service = LayoutService(documents=repository, access=_Access(part))
    return await service.create_part_block(
        _ScriptSession(repository, list(texts)),
        object(),
        uuid.uuid4(),
        part.document_id,
        part.id,
        order=None,
        box={"points": box},
    )


@pytest.mark.asyncio
async def test_regions_without_an_order_rank_left_to_right_whatever_order_they_are_drawn():
    part, lines = _spread()
    repository = _Repository(list(lines))
    recto = await _draw_region(part, repository, _box(700, 0, 1450, 400))
    verso = await _draw_region(part, repository, _box(0, 0, 700, 400))

    assert (verso.order, recto.order) == (0, 1)
    verso_lines = [line for line in lines if line.points[0][0] == 50]
    assert [line.order for line in verso_lines] == [0, 1, 2]


@pytest.mark.asyncio
async def test_regions_without_an_order_rank_right_to_left_on_a_syriac_page():
    part, lines = _spread()
    repository = _Repository(list(lines))
    syriac = ["ܐܒܓܕ"]
    verso = await _draw_region(part, repository, _box(0, 0, 700, 400), syriac)
    recto = await _draw_region(part, repository, _box(700, 0, 1450, 400), syriac)

    assert (recto.order, verso.order) == (0, 1)


@pytest.mark.asyncio
async def test_stacked_regions_without_an_order_rank_top_to_bottom():
    part, lines = _page(6)
    repository = _Repository(list(lines))
    lower = await _draw_region(part, repository, _box(50, 245, 950, 400))
    upper = await _draw_region(part, repository, _box(50, 50, 950, 245))

    assert (upper.order, lower.order) == (0, 1)
    assert [line.order for line in lines] == list(range(6))


@pytest.mark.asyncio
async def test_an_explicit_region_order_is_kept():
    part, lines = _spread()
    repository = _Repository(list(lines))
    recto = await _create_block(part, repository, _box(700, 0, 1450, 400), order=0)
    verso = await _create_block(part, repository, _box(0, 0, 700, 400), order=1)

    assert (recto.order, verso.order) == (0, 1)


async def _patch(part, repository, line, **updates) -> None:
    service = LayoutService(documents=repository, access=_Access(part))
    await service.patch_part_line(
        _Session(repository), object(), uuid.uuid4(), part.document_id, part.id, line.id, **updates
    )


@pytest.mark.asyncio
async def test_filing_a_model_line_into_a_region_keeps_it_a_model_line():
    part, lines, block = _region_page(3)
    for line in lines:
        line.block_id = None
        line.source = LineSource.kraken
        line.manual_geometry = False
    repository = _Repository(list(lines), [block])

    await _patch(part, repository, lines[1], block_id=block.id)

    assert lines[1].block_id == block.id
    assert lines[1].manual_geometry is False
    assert lines[1].source == LineSource.kraken


@pytest.mark.asyncio
async def test_moving_a_line_still_marks_it_hand_drawn():
    part, lines, block = _region_page(2)
    lines[0].source = LineSource.kraken
    lines[0].manual_geometry = False
    repository = _Repository(list(lines), [block])

    await _patch(part, repository, lines[0], points=_box(100, 90, 900, 140))

    assert lines[0].manual_geometry is True
    assert lines[0].source == LineSource.manual
