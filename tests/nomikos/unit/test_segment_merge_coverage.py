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
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult


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
        # What a real flush does to a new row: give it its primary key. Without this
        # every block id is None and every line's block_id matches it, so region
        # grouping is never really exercised.
        for item in self.added:
            if getattr(item, "id", 0) is None:
                item.id = uuid.uuid4()

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


# --- Region-aware merge (critique F1, F2, F3) ---


def _fresh_lines(block: str | None, x0: float = 100, x1: float = 900, count: int = 4):
    return [
        CanonicalLine(
            external_id=f"{block or 'x'}{i}",
            order=i,
            block_external_id=block,
            baseline={"points": [[x0, 135 + 50 * i], [x1, 135 + 50 * i]]},
            points=_box(x0, 100 + 50 * i, x1, 140 + 50 * i),
        )
        for i in range(count)
    ]


def _kept_line(part: DocumentPart, points, *, order: int = 0, block_id=None) -> Line:
    return Line(
        id=uuid.uuid4(),
        part_id=part.id,
        block_id=block_id,
        order=order,
        points=points,
        baseline={"points": points},
        source=LineSource.manual,
        manual_geometry=True,
        created_at=datetime.now(UTC),
    )


def _part(width: int | None = 1000, height: int | None = 1000) -> DocumentPart:
    part = DocumentPart(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        order=0,
        image_key="k",
        width=width,
        height=height,
    )
    part.lines = []
    part.blocks = []
    return part


def _merge(part: DocumentPart, segment: CanonicalSegmentResult) -> _Session:
    session = _Session()
    _MergeWithPart(part).apply_sync(
        session, part_id=part.id, canonical_segment=segment, job_id=uuid.uuid4()
    )
    return session


def _added_blocks(session: _Session):
    from backend.document.infrastructure.orm_models import Block

    return [item for item in session.added if isinstance(item, Block)]


def _added_lines(session: _Session) -> list[Line]:
    return [item for item in session.added if isinstance(item, Line)]


def test_a_segmenters_single_full_page_block_is_not_a_region():
    # blla wraps every line in one block whose box is the whole image.
    part = _part(1000, 1000)
    segment = CanonicalSegmentResult(
        blocks=[CanonicalBlock(external_id="b", order=0, box={"points": _box(0, 0, 1000, 1000)})],
        lines=_fresh_lines("b"),
    )
    session = _merge(part, segment)

    assert _added_blocks(session) == []
    assert all(line.block_id is None for line in _added_lines(session))


def test_a_single_block_that_is_not_the_whole_page_is_kept():
    part = _part(1000, 1000)
    segment = CanonicalSegmentResult(
        blocks=[CanonicalBlock(external_id="b", order=0, box={"points": _box(80, 80, 920, 320)})],
        lines=_fresh_lines("b"),
    )
    session = _merge(part, segment)

    (block,) = _added_blocks(session)
    assert all(line.block_id == block.id for line in _added_lines(session))


def test_a_kept_line_whose_block_went_joins_the_fresh_block_around_it():
    # A two-column page re-segmented: the left column's old model block is deleted, the
    # kept line drawn between left lines 1 and 2 must stay between them.
    part = _part(2000, 1000)
    old_block_id = uuid.uuid4()
    from backend.document.infrastructure.orm_models import Block

    part.blocks = [Block(id=old_block_id, part_id=part.id, order=0, box={}, manual_geometry=False)]
    kept = _kept_line(part, _box(110, 192, 880, 202), order=0, block_id=old_block_id)
    part.lines = [kept]
    segment = CanonicalSegmentResult(
        blocks=[
            CanonicalBlock(external_id="L", order=0, box={"points": _box(90, 90, 910, 300)}),
            CanonicalBlock(external_id="R", order=1, box={"points": _box(1090, 90, 1910, 300)}),
        ],
        lines=[*_fresh_lines("L"), *_fresh_lines("R", 1100, 1900)],
    )
    session = _merge(part, segment)

    left = next(block for block in _added_blocks(session) if block.order == 0)
    fresh = {line.source_metadata["external_id"]: line.order for line in _added_lines(session)}
    assert kept.block_id == left.id
    assert kept.order == 2
    assert [fresh[f"L{i}"] for i in range(4)] == [0, 1, 3, 4]
    assert [fresh[f"R{i}"] for i in range(4)] == [5, 6, 7, 8]


def test_a_ppocr_title_across_both_columns_stays_first():
    from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad
    from nomikos_inference.architectures.ppocr_det.response import build_ppocr_det_response

    def quad(x0, y0, x1, y1):
        return DetectedQuad(points=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], score=0.9)

    quads = [quad(100, 50, 1300, 90)]
    quads += [quad(100, 150 + 60 * i, 650, 190 + 60 * i) for i in range(8)]
    quads += [quad(750, 150 + 60 * i, 1300, 190 + 60 * i) for i in range(8)]
    response = build_ppocr_det_response(1400, 800, quads)
    segment = CanonicalSegmentResult.model_validate(response.model_dump(mode="json"))
    session = _merge(_part(1400, 800), segment)

    by_external = {line.source_metadata["external_id"]: line for line in _added_lines(session)}
    title = by_external[response.lines[0].external_id]
    assert title.order == 0
    assert sorted(line.order for line in by_external.values()) == list(range(17))
