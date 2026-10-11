"""When a freshly segmented line is the same ink as a line the merge kept."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from shapely.geometry import Polygon

from backend.document.application.segment_merge_service import (
    SegmentMergeService,
    _mostly_covered,
)
from backend.document.infrastructure.orm_models import DocumentPart, Line, LineSource
from backend.ml.domain.segment import CanonicalLine, CanonicalSegmentResult


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _kept(*boxes: list[list[float]]) -> list[Polygon]:
    return [Polygon(box) for box in boxes]


def test_a_near_identical_redraw_is_covered():
    kept = _kept(_box(100, 100, 900, 160))
    assert _mostly_covered(_box(96, 104, 905, 158), kept)


def test_a_kept_numeral_inside_a_fresh_text_line_does_not_cover_it():
    # The gutter numeral someone boxed by hand sits inside the sweep of a fresh kraken
    # text line. The numeral claims a sliver of that line's ink, not the line.
    kept = _kept(_box(880, 110, 940, 150))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_neighbours_touching_at_ascenders_do_not_add_up_to_coverage():
    # The line above and the line below each dip a few pixels into this one. Polygons,
    # not bounding boxes, so two shallow overlaps stay two shallow overlaps.
    kept = _kept(_box(100, 40, 900, 106), _box(100, 154, 900, 220))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_generous_neighbours_do_not_add_up_to_swallowing_a_distinct_line():
    # Hand-drawn boxes above and below are drawn tall and each dips 30% into this
    # line, which is neither of them. Summed they would pass the threshold and the
    # line would silently never appear; measured one kept line at a time it stays.
    kept = _kept(_box(100, 40, 900, 118), _box(100, 142, 900, 220))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_a_fresh_line_spanning_two_kept_lines_is_added_as_a_visible_overlap():
    # One fresh box across the gutter over two approved lines. Neither kept line is
    # a redraw of it, so it is added: a duplicate someone can see and delete, rather
    # than a hole nobody is told about.
    kept = _kept(_box(100, 100, 480, 160), _box(520, 100, 900, 160))
    assert not _mostly_covered(_box(100, 100, 900, 160), kept)


def test_half_inside_is_the_same_ink():
    kept = _kept(_box(100, 100, 500, 160))
    assert _mostly_covered(_box(100, 100, 900, 160), kept)
    assert not _mostly_covered(_box(100, 100, 901, 160), kept)


def test_nothing_kept_or_degenerate_geometry_covers_nothing():
    assert not _mostly_covered(_box(0, 0, 10, 10), [])
    assert not _mostly_covered([[0, 0], [10, 0]], _kept(_box(0, 0, 10, 10)))
    assert not _mostly_covered([], _kept(_box(0, 0, 10, 10)))


# --- apply_sync: one reading order for kept and fresh lines ---


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, item: object) -> None:
        self.added.append(item)

    def delete(self, _item: object) -> None:
        pass

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        pass


class _MergeWithPart(SegmentMergeService):
    def __init__(self, part: DocumentPart) -> None:
        self._part = part

    def _load_part(self, _session, _part_id):
        return self._part

    def _human_transcribed_line_ids(self, _session, _part):
        return set()


def test_a_kept_manual_line_is_numbered_between_the_fresh_lines_around_it():
    part = DocumentPart(id=uuid.uuid4(), document_id=uuid.uuid4(), order=0, image_key="k")
    # Drawn by hand between what the segmenter will call lines 1 and 2, and numbered 0
    # on the page it was kept from: it collides with the segmenter's first line.
    kept = Line(
        id=uuid.uuid4(),
        part_id=part.id,
        order=0,
        points=_box(110, 192, 880, 202),
        baseline={"points": _box(110, 192, 880, 202)},
        source=LineSource.manual,
        manual_geometry=True,
        created_at=datetime.now(UTC),
    )
    part.lines = [kept]
    part.blocks = []
    segment = CanonicalSegmentResult(
        lines=[
            CanonicalLine(
                external_id=f"l{i}",
                order=i,
                baseline={"points": [[100, 135 + 50 * i], [900, 135 + 50 * i]]},
                points=_box(100, 100 + 50 * i, 900, 140 + 50 * i),
            )
            for i in range(4)
        ]
    )
    session = _Session()

    _MergeWithPart(part).apply_sync(
        session, part_id=part.id, canonical_segment=segment, job_id=uuid.uuid4()
    )

    fresh = {line.source_metadata["external_id"]: line.order for line in session.added}
    assert kept.order == 2
    assert fresh == {"l0": 0, "l1": 1, "l2": 3, "l3": 4}
