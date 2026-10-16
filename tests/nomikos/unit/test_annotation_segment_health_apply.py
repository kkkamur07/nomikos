"""Apply paths write the mask they derive; history snapshots keep geometry.

`test_segment_health_service.py` covers offers and safety rules. What is
asserted here is the Required-fix behaviour: split, merge and trim rewrite
`mask` alongside `points` and `baseline`, and a history snapshot round-trips
baseline, mask and source_metadata through restore.
"""

from __future__ import annotations

import uuid

import pytest

from backend.annotation.application.history_service import AnnotationHistoryService
from backend.annotation.application.segment_health_service import SegmentHealthService
from backend.core.exceptions import ValidationError
from backend.document.infrastructure.orm_models import (
    Line,
    LineGeometryKind,
    LineSource,
)

PAGE_WIDTH = 2479.0
PAGE_HEIGHT = 3508.0


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
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.width = PAGE_WIDTH
        self.height = PAGE_HEIGHT


class _Repository:
    def __init__(self, part: _Part, lines: list[Line]) -> None:
        self._part = part
        self.lines = lines

    async def list_part_lines(self, _session, _part_id) -> list[Line]:
        return sorted(self.lines, key=lambda line: line.order)

    async def get_part(self, _session, _part_id):
        return self._part

    async def paired_line_ids(self, _session, _part_id) -> set[uuid.UUID]:
        return set()

    async def lock_part(self, _session, _part_id) -> None:
        return None


class _DocumentService:
    def __init__(self, part: _Part) -> None:
        self._part = part

    async def get_document(self, _session, _user, _project_id, _document_id):
        class _Doc:
            id = self._part.document_id

        return _Doc()


def _row(
    x0: float, x1: float, y: float, *, order: int, height: float = 100.0, mask=None
) -> Line:
    top, bottom = y - height / 2, y + height / 2
    line = Line(
        id=uuid.uuid4(),
        part_id=uuid.uuid4(),
        order=order,
        kind=LineGeometryKind.polygon,
        source=LineSource.kraken,
        source_metadata={},
        baseline={"points": [[x0, bottom], [x1, bottom]]},
        mask=mask,
        points=[[x0, top], [x1, top], [x1, bottom], [x0, bottom]],
        manual_geometry=False,
    )
    line.transcriptions = []
    return line


def _service(lines: list[Line]):
    part = _Part()
    repo = _Repository(part, lines)
    service = SegmentHealthService(documents=repo, document_service=_DocumentService(part))
    return service, part


class TestApplyTrimMask:
    @pytest.mark.asyncio
    async def test_trim_rewrites_points_and_mask_and_marks_both_by_hand(self) -> None:
        upper = _row(300.0, 1100.0, 500.0, order=0, mask={"points": [[0.0, 0.0]]})
        lower = _row(300.0, 1100.0, 560.0, order=1, mask={"points": [[0.0, 0.0]]})
        service, part = _service([upper, lower])

        report = await service.report(_Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id)
        (trim,) = [t for t in report.overlaps if t.upper_id == str(upper.id)]
        assert not trim.duplicate

        await service.apply_trim(
            _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, upper.id, lower.id
        )

        assert upper.points == trim.upper_points
        assert lower.points == trim.lower_points
        assert upper.mask == {"points": trim.upper_points}
        assert lower.mask == {"points": trim.lower_points}
        assert upper.manual_geometry is True
        assert lower.manual_geometry is True

    @pytest.mark.asyncio
    async def test_trim_of_one_line_drawn_twice_is_refused(self) -> None:
        first = _row(300.0, 1100.0, 500.0, order=0)
        second = _row(300.0, 1100.0, 500.0, order=1)
        service, part = _service([first, second])

        report = await service.report(_Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id)
        (trim,) = [t for t in report.overlaps if t.upper_id == str(first.id)]
        assert trim.duplicate

        with pytest.raises(ValidationError, match="drawn twice"):
            await service.apply_trim(
                _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, first.id, second.id
            )

    @pytest.mark.asyncio
    async def test_a_pair_the_report_does_not_offer_is_refused(self) -> None:
        upper = _row(300.0, 1100.0, 500.0, order=0)
        lower = _row(300.0, 1100.0, 900.0, order=1)
        service, part = _service([upper, lower])

        with pytest.raises(ValidationError, match="no longer offered"):
            await service.apply_trim(
                _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, upper.id, lower.id
            )


class TestApplyMaskRegression:
    @pytest.mark.asyncio
    async def test_split_tracks_the_mask_with_each_new_piece(self) -> None:
        lines = [
            _row(300.0, 1100.0, 400.0 + row * 120.0, order=2 * row) for row in range(6)
        ] + [
            _row(1380.0, 2180.0, 400.0 + row * 120.0, order=2 * row + 1)
            for row in range(6)
        ]
        crosser = _row(300.0, 2180.0, 500.0, order=99)
        lines.append(crosser)
        service, part = _service(list(lines))

        await service.apply_split(
            _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, crosser.id
        )

        assert crosser.mask == {"points": crosser.points}
        assert crosser.mask["points"] != [
            [300.0, 450.0],
            [2180.0, 450.0],
            [2180.0, 550.0],
            [300.0, 550.0],
        ]

    @pytest.mark.asyncio
    async def test_merge_tracks_the_mask_with_the_merged_outline(self) -> None:
        lines = [
            _row(300.0, 1100.0, 2700.0, order=2),
            _row(300.0, 1100.0, 2900.0, order=0),
            _row(960.0, 1060.0, 2900.0, order=1),
        ]
        service, part = _service(lines)
        primary, fragment = lines[1], lines[2]

        await service.apply_merge(
            _Session(), object(), uuid.uuid4(), uuid.uuid4(), part.id, primary.id, fragment.id
        )

        assert primary.mask == {"points": primary.points}


class TestHistoryGeometryState:
    def test_line_state_carries_baseline_mask_and_source_metadata(self) -> None:
        line = _row(300.0, 1100.0, 500.0, order=3)
        line.source_metadata = {"kraken": True}
        service = AnnotationHistoryService.__new__(AnnotationHistoryService)

        state = service._line_state(line)

        assert state["baseline"] == line.baseline
        assert state["mask"] == line.mask
        assert state["source_metadata"] == {"kraken": True}

    def test_line_state_preserves_a_custom_baseline(self) -> None:
        line = _row(300.0, 1100.0, 500.0, order=3)
        line.baseline = {"points": [[310.0, 540.0], [1090.0, 545.0]]}
        line.mask = {"points": [[300.0, 450.0], [1100.0, 450.0]]}
        service = AnnotationHistoryService.__new__(AnnotationHistoryService)

        state = service._line_state(line)

        assert state["baseline"] == {"points": [[310.0, 540.0], [1090.0, 545.0]]}
        assert state["mask"] == {"points": [[300.0, 450.0], [1100.0, 450.0]]}

    @pytest.mark.asyncio
    async def test_restore_passes_baseline_and_mask_back_to_the_page(self) -> None:
        from backend.document.infrastructure.orm_models import LineSource

        captured: dict[str, object] = {}
        part = _Part()
        line_id = uuid.uuid4()
        snapshot = type(
            "Snapshot",
            (),
            {
                "part_id": part.id,
                "state": {
                    "lines": [
                        {
                            "id": str(line_id),
                            "block_id": None,
                            "order": 0,
                            "kind": "polygon",
                            "points": [[1.0, 1.0]],
                            "baseline": {"points": [[1.0, 2.0], [3.0, 4.0]]},
                            "mask": {"points": [[5.0, 6.0]]},
                            "source": LineSource.kraken.value,
                            "source_metadata": {"kraken": True},
                            "kraken_ceiling": None,
                            "approved_text": None,
                        }
                    ]
                },
            },
        )()
        service = AnnotationHistoryService(
            documents=_Repository(part, []), document_service=_DocumentService(part)
        )

        async def _fake_replace(session, user, project_id, document_id, part_id, *, lines, **kw):
            captured["lines"] = lines
            return []

        service._document_service.replace_part_lines = _fake_replace  # type: ignore[method-assign]

        class _GetSession(_Session):
            async def get(self, _model, _id):
                return snapshot

        await service.restore_snapshot(
            _GetSession(), object(), uuid.uuid4(), uuid.uuid4(), part.id, uuid.uuid4()
        )

        (restored,) = captured["lines"]
        assert restored["baseline"] == {"points": [[1.0, 2.0], [3.0, 4.0]]}
        assert restored["mask"] == {"points": [[5.0, 6.0]]}
