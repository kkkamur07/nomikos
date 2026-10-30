"""The layer between the geometry and the page.

`test_segment_health.py` covers the geometry itself. What is asserted here is
everything that only exists once rows are involved: that a choice is re-derived
from the database rather than trusted from the client, that `order` stays a
position and not a duplicate, and that no path deletes work somebody did.
"""

from __future__ import annotations

import uuid

import pytest

from backend.annotation.application.segment_health_service import SegmentHealthService
from backend.core.exceptions import ValidationError
from backend.document.infrastructure.orm_models import (
    Line,
    LineTranscription,
    Transcription,
    TranscriptionKind,
)

PAGE_WIDTH = 2479.0
PAGE_HEIGHT = 3508.0
LEFT_COLUMN = (300.0, 1100.0)
RIGHT_COLUMN = (1380.0, 2180.0)


# -- fakes ---------------------------------------------------------------


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    def add(self, item: object) -> None:
        self.added.append(item)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def commit(self) -> None:
        self.commits += 1


class _Part:
    def __init__(self, width: int | None, height: int | None) -> None:
        self.id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.width = width
        self.height = height


class _Repository:
    def __init__(self, part: _Part, lines: list[Line]) -> None:
        self._part = part
        self.lines = lines
        self.paired: set[uuid.UUID] = set()
        self.locks = 0

    async def list_part_lines(self, _session, _part_id) -> list[Line]:
        # Rows come back ordered, and a deleted row is gone. The service reads
        # this twice per apply, so it has to reflect the session's deletes.
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
    manual: bool = False,
) -> Line:
    top, bottom = y - height / 2, y + height / 2
    line = Line(
        id=uuid.uuid4(),
        part_id=uuid.uuid4(),
        order=order,
        baseline={"points": [[x0, bottom], [x1, bottom]]},
        points=[[x0, top], [x1, top], [x1, bottom], [x0, bottom]],
        manual_geometry=manual,
    )
    line.transcriptions = []
    if text:
        layer = Transcription(
            id=uuid.uuid4(), document_id=uuid.uuid4(), name="Ground truth", kind=kind
        )
        entry = LineTranscription(id=uuid.uuid4(), line_id=line.id, text=text)
        entry.transcription = layer
        line.transcriptions = [entry]
    return line


def _two_column_page() -> list[Line]:
    lines: list[Line] = []
    order = 0
    for row in range(20):
        y = 400.0 + row * 120.0
        lines.append(_line(*LEFT_COLUMN, y, order=order))
        order += 1
        lines.append(_line(*RIGHT_COLUMN, y, order=order))
        order += 1
    return lines


def _service(lines: list[Line], *, width: int | None = 2479, height: int | None = 3508):
    part = _Part(width, height)
    repo = _Repository(part, lines)
    service = SegmentHealthService(documents=repo, document_service=_DocumentService(part))
    return service, repo, part


async def _report(service, part):
    return await service.report(_Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id)


# -- reading -------------------------------------------------------------


class TestReport:
    @pytest.mark.asyncio
    async def test_a_segment_across_the_gutter_is_reported_as_a_split(self) -> None:
        lines = _two_column_page()
        crosser = _line(LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0, order=99)
        lines.append(crosser)
        service, _repo, part = _service(lines)

        report = await _report(service, part)

        assert [split.line_id for split in report.spanning] == [str(crosser.id)]
        assert report.line_count == len(lines)
        assert report.finding_count >= 1

    @pytest.mark.asyncio
    async def test_a_clean_page_reports_nothing(self) -> None:
        service, _repo, part = _service(_two_column_page())
        report = await _report(service, part)
        assert report.finding_count == 0

    @pytest.mark.asyncio
    async def test_a_part_without_stored_dimensions_is_measured_and_says_so(self) -> None:
        """Nullable width/height must not turn into a page of size zero."""
        service, _repo, part = _service(_two_column_page(), width=None, height=None)

        report = await _report(service, part)

        assert report.measured_page is False
        assert report.page_width == pytest.approx(RIGHT_COLUMN[1])
        assert report.page_height > 0

    @pytest.mark.asyncio
    async def test_a_stored_page_is_not_measured(self) -> None:
        service, _repo, part = _service(_two_column_page())
        report = await _report(service, part)
        assert report.measured_page is True
        assert report.page_width == PAGE_WIDTH

    @pytest.mark.asyncio
    async def test_a_page_with_no_usable_geometry_reports_nothing_rather_than_dividing_by_zero(
        self,
    ) -> None:
        bare = Line(id=uuid.uuid4(), part_id=uuid.uuid4(), order=0, baseline={}, points=[])
        bare.transcriptions = []
        service, _repo, part = _service([bare], width=None, height=None)

        report = await _report(service, part)

        assert report.considered_count == 0
        assert report.finding_count == 0


# -- applying ------------------------------------------------------------


class TestApplySplit:
    @pytest.mark.asyncio
    async def test_the_new_piece_takes_the_next_position_and_pushes_the_rest_down(self) -> None:
        """#114: a partial re-segment left kept and fresh rows sharing an order."""
        lines = _two_column_page()
        crosser = _line(LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0, order=4)
        for line in lines:
            if line.order >= 4:
                line.order += 1
        lines.append(crosser)
        service, repo, part = _service(lines)
        session = _Session()

        await service.apply_split(
            session, object(), uuid.uuid4(), uuid.uuid4(), part.id, crosser.id
        )

        fresh = [item for item in session.added if isinstance(item, Line)]
        assert len(fresh) == 1
        assert fresh[0].order == crosser.order + 1
        orders = [line.order for line in lines] + [fresh[0].order]
        assert len(orders) == len(set(orders)), "two lines claim one position"
        assert crosser.manual_geometry is True
        assert fresh[0].manual_geometry is True

    @pytest.mark.asyncio
    async def test_a_segment_the_report_does_not_offer_is_refused(self) -> None:
        lines = _two_column_page()
        service, _repo, part = _service(lines)

        with pytest.raises(
            ValidationError, match="^This segment is no longer offered a column split$"
        ):
            await service.apply_split(
                _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, lines[0].id
            )


class TestApplyMerge:
    @pytest.mark.asyncio
    async def test_the_primary_keeps_its_id_and_the_fragment_row_goes(self) -> None:
        lines = _two_column_page()
        primary = _line(300.0, 900.0, 2900.0, order=90)
        fragment = _line(960.0, 1060.0, 2900.0, order=91)
        lines += [primary, fragment]
        service, _repo, part = _service(lines)
        session = _Session()
        primary_id = primary.id

        await service.apply_merge(
            session, object(), uuid.uuid4(), uuid.uuid4(), part.id, primary.id, fragment.id
        )

        assert primary.id == primary_id, "the row carrying the text must survive"
        assert session.deleted == [fragment]
        assert primary.manual_geometry is True
        assert primary.points != [
            [300.0, 2850.0],
            [900.0, 2850.0],
            [900.0, 2950.0],
            [300.0, 2950.0],
        ]

    @pytest.mark.asyncio
    async def test_a_fragment_carrying_text_is_never_merged(self) -> None:
        """The finder refuses to offer it, so the apply path refuses too."""
        lines = _two_column_page()
        primary = _line(300.0, 900.0, 2900.0, order=90)
        fragment = _line(960.0, 1060.0, 2900.0, order=91, text="ܡܪܝܐ")
        lines += [primary, fragment]
        service, _repo, part = _service(lines)
        session = _Session()

        with pytest.raises(ValidationError):
            await service.apply_merge(
                session, object(), uuid.uuid4(), uuid.uuid4(), part.id, primary.id, fragment.id
            )
        assert session.deleted == []


class TestApplyDelete:
    @pytest.mark.asyncio
    async def test_an_id_the_report_did_not_flag_is_refused(self) -> None:
        lines = _two_column_page()
        service, _repo, part = _service(lines)
        session = _Session()

        with pytest.raises(ValidationError, match="^This segment is not flagged as a suspect$"):
            await service.apply_delete(
                session, object(), uuid.uuid4(), uuid.uuid4(), part.id, lines[0].id
            )
        assert session.deleted == []

    @pytest.mark.asyncio
    async def test_a_flagged_speck_is_deleted_and_the_page_closes_up(self) -> None:
        lines = _two_column_page()
        speck = _line(2300.0, 2340.0, 1000.0, order=len(lines), height=20.0)
        lines.append(speck)
        below = _line(*LEFT_COLUMN, 3300.0, order=len(lines))
        lines.append(below)
        service, _repo, part = _service(lines)
        session = _Session()

        report = await _report(service, part)
        assert str(speck.id) in {item.line_id for item in report.suspects}

        await service.apply_delete(session, object(), uuid.uuid4(), uuid.uuid4(), part.id, speck.id)

        assert session.deleted == [speck]
        assert below.order == len(lines) - 2, "rows below a deleted one must close the gap"

    @pytest.mark.asyncio
    async def test_a_speck_someone_transcribed_is_not_a_suspect_at_all(self) -> None:
        """Text is what separates unread ink from a smudge, at both layers."""
        lines = _two_column_page()
        speck = _line(2300.0, 2340.0, 1000.0, order=len(lines), height=20.0, text="ܐ")
        lines.append(speck)
        service, _repo, part = _service(lines)

        report = await _report(service, part)

        assert str(speck.id) not in {item.line_id for item in report.suspects}
        with pytest.raises(ValidationError):
            await service.apply_delete(
                _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, speck.id
            )

    @pytest.mark.asyncio
    async def test_an_unapproved_model_prediction_still_counts_as_text(self) -> None:
        """A model prediction nobody has approved is the thing about to be corrected.

        `_is_paired` looks only at the ground-truth layer, so this line is not
        paired. `_has_text` looks at every layer, which is what keeps it off the
        suspect list: deleting it would throw away the draft a reviewer is
        about to correct.
        """
        lines = _two_column_page()
        speck = _line(
            2300.0,
            2340.0,
            1000.0,
            order=len(lines),
            height=20.0,
            text="ܐ",
            kind=TranscriptionKind.model,
        )
        lines.append(speck)
        service, _repo, part = _service(lines)

        report = await _report(service, part)

        assert str(speck.id) not in {item.line_id for item in report.suspects}


class TestPairing:
    """A pairing lives in `page_transcription_lines`, not in the text layers.

    Greptile's P1 on #109: reading only the transcriptions calls a line somebody
    paired but never transcribed "untouched", and the suspect and fragment paths
    then delete it, taking the human's pairing decision with it.
    """

    @pytest.mark.asyncio
    async def test_a_line_paired_without_ground_truth_text_is_never_a_suspect(self) -> None:
        lines = _two_column_page()
        speck = _line(2300.0, 2340.0, 1000.0, order=len(lines), height=20.0)
        lines.append(speck)
        service, repo, part = _service(lines)

        while_unpaired = await _report(service, part)
        assert str(speck.id) in {item.line_id for item in while_unpaired.suspects}

        repo.paired = {speck.id}
        report = await _report(service, part)

        assert str(speck.id) not in {item.line_id for item in report.suspects}

    @pytest.mark.asyncio
    async def test_a_paired_line_cannot_be_deleted_even_carrying_no_text(self) -> None:
        lines = _two_column_page()
        speck = _line(2300.0, 2340.0, 1000.0, order=len(lines), height=20.0)
        lines.append(speck)
        service, repo, part = _service(lines)
        repo.paired = {speck.id}
        session = _Session()

        with pytest.raises(ValidationError):
            await service.apply_delete(
                session, object(), uuid.uuid4(), uuid.uuid4(), part.id, speck.id
            )
        assert session.deleted == []

    @pytest.mark.asyncio
    async def test_a_paired_fragment_is_not_offered_a_merge(self) -> None:
        lines = _two_column_page()
        primary = _line(300.0, 900.0, 2900.0, order=90)
        fragment = _line(960.0, 1060.0, 2900.0, order=91)
        lines += [primary, fragment]
        service, repo, part = _service(lines)

        offered = await _report(service, part)
        assert (str(primary.id), str(fragment.id)) in {
            (item.primary_id, item.fragment_id) for item in offered.fragments
        }

        repo.paired = {fragment.id}
        session = _Session()
        with pytest.raises(ValidationError):
            await service.apply_merge(
                session, object(), uuid.uuid4(), uuid.uuid4(), part.id, primary.id, fragment.id
            )
        assert session.deleted == []


class TestLocking:
    """Recomputing closes the stale-client window; the lock closes the server's own.

    Greptile's second P1 on #109: between the recompute and the commit another
    edit on the same part can land, and the apply then writes over geometry it
    was never derived from.
    """

    @pytest.mark.asyncio
    async def test_every_apply_path_locks_the_part_before_it_reads(self) -> None:
        lines = _two_column_page()
        crosser = _line(LEFT_COLUMN[0], RIGHT_COLUMN[1], 500.0, order=99)
        speck = _line(2300.0, 2340.0, 1000.0, order=100, height=20.0)
        primary = _line(300.0, 900.0, 2900.0, order=101)
        fragment = _line(960.0, 1060.0, 2900.0, order=102)
        lines += [crosser, speck, primary, fragment]
        service, repo, part = _service(lines)
        ids = (uuid.uuid4(), uuid.uuid4())

        await service.apply_split(_Session(), object(), *ids, part.id, crosser.id)
        assert repo.locks == 1
        await service.apply_merge(_Session(), object(), *ids, part.id, primary.id, fragment.id)
        assert repo.locks == 2
        await service.apply_delete(_Session(), object(), *ids, part.id, speck.id)
        assert repo.locks == 3

    @pytest.mark.asyncio
    async def test_reading_the_report_takes_no_lock(self) -> None:
        """The GET is a read. Locking it would queue reviewers behind each other."""
        service, repo, part = _service(_two_column_page())
        await _report(service, part)
        assert repo.locks == 0
