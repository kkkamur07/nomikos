"""Merging the lines a human picked, rather than the pair a detector offered.

Two things are asserted here and nowhere else. The geometry: one outline that
covers every piece, one baseline read left to right, and the flag that stops a
re-segment from undoing the whole thing. And the identity rule, which is the
dangerous half: which row keeps its id, and therefore which transcriptions and
which pairing survive. All four cases of that rule are here, because the one
that is easy to get wrong (several lines carrying work, no choice named) is the
one where getting it wrong deletes somebody's ground truth.
"""

from __future__ import annotations

import uuid

import pytest

from backend.annotation.application.line_merge_service import (
    MAX_MERGE_LINES,
    LineMergeService,
    LineMergeTextConflict,
)
from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.infrastructure.orm_models import (
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)

# -- fakes ---------------------------------------------------------------


class _Session:
    """Deletes really remove the row, so the list the service answers with is real."""

    def __init__(self, repository: _Repository | None = None) -> None:
        self.deleted: list[object] = []
        self.commits = 0
        self.repository = repository

    async def delete(self, item: object) -> None:
        self.deleted.append(item)
        if self.repository is not None:
            self.repository.lines = [line for line in self.repository.lines if line is not item]

    async def commit(self) -> None:
        self.commits += 1


class _Part:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.document_id = uuid.uuid4()


class _Repository:
    def __init__(self, part: _Part, lines: list[Line]) -> None:
        self._part = part
        self.lines = lines
        self.paired: set[uuid.UUID] = set()
        self.locks = 0

    async def list_part_lines(self, _session, _part_id) -> list[Line]:
        return sorted(self.lines, key=lambda line: line.order)

    async def get_part(self, _session, _part_id):
        return self._part

    async def paired_line_ids(self, _session, _part_id) -> set[uuid.UUID]:
        return self.paired

    async def lock_part(self, _session, _part_id) -> None:
        self.locks += 1


class _DocumentService:
    def __init__(self, part: _Part) -> None:
        self._part = part

    async def get_document(self, _session, _user, _project_id, _document_id):
        class _Doc:
            id = self._part.document_id

        return _Doc()


class _RefusingDocumentService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_document(self, _session, _user, _project_id, _document_id):
        raise self._error


# -- builders ------------------------------------------------------------


def _line(
    x0: float,
    x1: float,
    y: float,
    *,
    order: int,
    height: float = 100.0,
    text: str = "",
    kind: TranscriptionKind = TranscriptionKind.ground_truth,
    baseline: bool = True,
) -> Line:
    top, bottom = y - height / 2, y + height / 2
    line = Line(
        id=uuid.uuid4(),
        part_id=uuid.uuid4(),
        order=order,
        baseline={"points": [[x0, bottom], [x1, bottom]]} if baseline else {},
        points=[[x0, top], [x1, top], [x1, bottom], [x0, bottom]],
        manual_geometry=False,
    )
    line.transcriptions = []
    if text:
        layer = Transcription(id=uuid.uuid4(), document_id=uuid.uuid4(), name="Layer", kind=kind)
        entry = LineTranscription(id=uuid.uuid4(), line_id=line.id, text=text)
        entry.transcription = layer
        line.transcriptions = [entry]
    return line


def _service(lines: list[Line]):
    part = _Part()
    repo = _Repository(part, lines)
    service = LineMergeService(documents=repo, document_service=_DocumentService(part))
    return service, repo, part, _Session(repo)


async def _merge(service, part, session, line_ids, keep_text_from=None):
    return await service.merge_lines(
        session,
        object(),
        uuid.uuid4(),
        uuid.uuid4(),
        part.id,
        line_ids,
        keep_text_from,
    )


# -- geometry ------------------------------------------------------------


class TestGeometry:
    @pytest.mark.asyncio
    async def test_the_merged_outline_covers_every_piece_and_the_gaps_between_them(self) -> None:
        pieces = [
            _line(300.0, 600.0, 500.0, order=0),
            _line(700.0, 900.0, 500.0, order=1),
            _line(1000.0, 1100.0, 500.0, order=2),
        ]
        service, _repo, part, session = _service(list(pieces))

        await _merge(service, part, session, [piece.id for piece in pieces])

        merged = pieces[0]
        xs = [point[0] for point in merged.points]
        ys = [point[1] for point in merged.points]
        assert min(xs) == 300.0
        assert max(xs) == 1100.0
        assert min(ys) == 450.0
        assert max(ys) == 550.0

    @pytest.mark.asyncio
    async def test_the_mask_is_rewritten_with_the_outline_rather_than_left_behind(self) -> None:
        """A stale mask is a bounding box around the piece the row used to be."""
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        first.mask = {"points": list(first.points)}
        service, _repo, part, session = _service([first, second])

        await _merge(service, part, session, [first.id, second.id])

        assert first.mask == {"points": first.points}

    @pytest.mark.asyncio
    async def test_the_merged_line_is_marked_manual_so_a_resegment_cannot_undo_it(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        service, _repo, part, session = _service([first, second])

        await _merge(service, part, session, [first.id, second.id])

        assert first.manual_geometry is True

    @pytest.mark.asyncio
    async def test_the_baselines_are_joined_left_to_right(self) -> None:
        # Named right to left, so a merge that simply concatenated would show it.
        right = _line(700.0, 900.0, 500.0, order=1)
        left = _line(300.0, 600.0, 500.0, order=0)
        service, _repo, part, session = _service([left, right])

        await _merge(service, part, session, [right.id, left.id])

        xs = [point[0] for point in left.baseline["points"]]
        assert xs == sorted(xs)
        assert xs[0] == 300.0
        assert xs[-1] == 900.0

    @pytest.mark.asyncio
    async def test_a_piece_with_no_baseline_contributes_nothing_to_the_merged_one(self) -> None:
        line = _line(300.0, 600.0, 500.0, order=0)
        blot = _line(700.0, 730.0, 500.0, order=1, baseline=False)
        service, _repo, part, session = _service([line, blot])

        await _merge(service, part, session, [line.id, blot.id])

        assert [point[0] for point in line.baseline["points"]] == [300.0, 600.0]

    @pytest.mark.asyncio
    async def test_a_selection_where_nothing_has_a_baseline_leaves_the_merged_one_empty(
        self,
    ) -> None:
        first = _line(300.0, 600.0, 500.0, order=0, baseline=False)
        second = _line(700.0, 900.0, 500.0, order=1, baseline=False)
        service, _repo, part, session = _service([first, second])

        await _merge(service, part, session, [first.id, second.id])

        assert first.baseline == {"points": []}


# -- identity and text ---------------------------------------------------


class TestSurvivor:
    @pytest.mark.asyncio
    async def test_with_nothing_written_on_any_of_them_the_first_in_reading_order_survives(
        self,
    ) -> None:
        pieces = [_line(300.0 + n * 400, 600.0 + n * 400, 500.0, order=n) for n in range(3)]
        service, _repo, part, session = _service(list(pieces))

        lines = await _merge(service, part, session, [pieces[2].id, pieces[0].id, pieces[1].id])

        assert [line.id for line in lines] == [pieces[0].id]
        assert session.deleted == [pieces[1], pieces[2]]

    @pytest.mark.asyncio
    async def test_the_one_line_carrying_ground_truth_survives_even_from_the_middle(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        keeper = _line(700.0, 900.0, 500.0, order=1, text="alpha")
        last = _line(1000.0, 1100.0, 500.0, order=2)
        service, _repo, part, session = _service([first, keeper, last])

        lines = await _merge(service, part, session, [first.id, keeper.id, last.id])

        assert [line.id for line in lines] == [keeper.id]
        assert session.deleted == [first, last]

    @pytest.mark.asyncio
    async def test_a_pairing_with_no_ground_truth_layer_still_makes_a_line_the_survivor(
        self,
    ) -> None:
        """The pairing is a decision recorded outside the transcription layers."""
        first = _line(300.0, 600.0, 500.0, order=0)
        paired = _line(700.0, 900.0, 500.0, order=1)
        service, repo, part, session = _service([first, paired])
        repo.paired = {paired.id}

        lines = await _merge(service, part, session, [first.id, paired.id])

        assert [line.id for line in lines] == [paired.id]

    @pytest.mark.asyncio
    async def test_machine_text_is_not_human_work_and_does_not_decide_the_survivor(self) -> None:
        """Every line has a prediction after one OCR run; that cannot be a conflict."""
        first = _line(300.0, 600.0, 500.0, order=0, text="alpha", kind=TranscriptionKind.model)
        second = _line(700.0, 900.0, 500.0, order=1, text="beta", kind=TranscriptionKind.model)
        service, _repo, part, session = _service([first, second])

        lines = await _merge(service, part, session, [first.id, second.id])

        assert [line.id for line in lines] == [first.id]

    @pytest.mark.asyncio
    async def test_several_lines_carrying_work_are_refused_with_the_ids_on_the_error(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0, text="alpha")
        middle = _line(700.0, 900.0, 500.0, order=1)
        last = _line(1000.0, 1100.0, 500.0, order=2, text="beta")
        service, _repo, part, session = _service([first, middle, last])

        with pytest.raises(LineMergeTextConflict) as raised:
            await _merge(service, part, session, [first.id, middle.id, last.id])

        assert raised.value.line_ids == [first.id, last.id]
        assert session.deleted == []
        assert session.commits == 0

    @pytest.mark.asyncio
    async def test_naming_one_of_them_lets_the_merge_through_and_deletes_the_rest(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0, text="alpha")
        last = _line(1000.0, 1100.0, 500.0, order=1, text="beta")
        service, _repo, part, session = _service([first, last])

        lines = await _merge(service, part, session, [first.id, last.id], last.id)

        assert [line.id for line in lines] == [last.id]
        assert session.deleted == [first]

    @pytest.mark.asyncio
    async def test_keeping_the_text_of_a_line_that_has_none_is_refused(self) -> None:
        """Honouring it would delete every line that does carry work."""
        blank = _line(300.0, 600.0, 500.0, order=0)
        written = _line(700.0, 900.0, 500.0, order=1, text="alpha")
        other = _line(1000.0, 1100.0, 500.0, order=2, text="beta")
        service, _repo, part, session = _service([blank, written, other])

        with pytest.raises(ValidationError):
            await _merge(service, part, session, [blank.id, written.id, other.id], blank.id)


# -- ordering and guards -------------------------------------------------


class TestOrderAndGuards:
    @pytest.mark.asyncio
    async def test_the_merged_line_takes_the_topmost_position_and_the_rest_close_up(self) -> None:
        page = [_line(300.0, 600.0, 400.0 + n * 120, order=n) for n in range(6)]
        keeper = page[3]
        keeper.transcriptions = _line(0, 1, 0, order=0, text="alpha").transcriptions
        service, _repo, part, session = _service(list(page))

        lines = await _merge(service, part, session, [page[1].id, page[3].id])

        assert [line.order for line in lines] == [0, 1, 2, 3, 4]
        assert [line.id for line in lines] == [
            page[0].id,
            keeper.id,
            page[2].id,
            page[4].id,
            page[5].id,
        ]

    @pytest.mark.asyncio
    async def test_the_part_is_locked_before_the_rows_are_read(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        service, repo, part, session = _service([first, second])

        await _merge(service, part, session, [first.id, second.id])

        assert repo.locks == 1

    @pytest.mark.asyncio
    async def test_one_line_is_not_a_merge(self) -> None:
        only = _line(300.0, 600.0, 500.0, order=0)
        service, _repo, part, session = _service([only])

        with pytest.raises(ValidationError):
            await _merge(service, part, session, [only.id])

    @pytest.mark.asyncio
    async def test_the_same_line_named_twice_is_refused_rather_than_merged_with_itself(
        self,
    ) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        service, _repo, part, session = _service([first, second])

        with pytest.raises(ValidationError):
            await _merge(service, part, session, [first.id, first.id])

    @pytest.mark.asyncio
    async def test_a_selection_past_the_cap_is_refused(self) -> None:
        page = [_line(300.0, 600.0, 400.0 + n * 120, order=n) for n in range(MAX_MERGE_LINES + 1)]
        service, _repo, part, session = _service(list(page))

        with pytest.raises(ValidationError):
            await _merge(service, part, session, [line.id for line in page])

    @pytest.mark.asyncio
    async def test_an_id_from_another_page_is_a_not_found_and_nothing_is_written(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        service, _repo, part, session = _service([first, second])

        with pytest.raises(NotFoundError):
            await _merge(service, part, session, [first.id, uuid.uuid4()])

        assert session.deleted == []
        assert session.commits == 0

    @pytest.mark.asyncio
    async def test_keep_text_from_has_to_name_a_line_that_is_being_merged(self) -> None:
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        service, _repo, part, session = _service([first, second])

        with pytest.raises(ValidationError):
            await _merge(service, part, session, [first.id, second.id], uuid.uuid4())

    @pytest.mark.asyncio
    async def test_the_access_check_runs_before_the_lock_is_taken(self) -> None:
        """A caller who cannot read the part must not be able to hold its row."""
        first = _line(300.0, 600.0, 500.0, order=0)
        second = _line(700.0, 900.0, 500.0, order=1)
        part = _Part()
        repo = _Repository(part, [first, second])
        service = LineMergeService(
            documents=repo,
            document_service=_RefusingDocumentService(NotFoundError("Document not found")),
        )

        with pytest.raises(NotFoundError):
            await _merge(service, part, _Session(repo), [first.id, second.id])

        assert repo.locks == 0
