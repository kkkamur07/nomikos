"""Segment merge - replace the machine geometry nobody has touched, keep everything a human did.

A segment job redraws a page. What it must never do is throw away work: ``manual_geometry``
records that someone moved a polygon, but that is only one of the ways a human touches a
line. Approving its text writes a ground-truth transcription; pairing it with a line of
imported page text writes a ``page_transcription_lines`` row. Neither sets
``manual_geometry``, so a page where forty lines were corrected and approved without a
single polygon being nudged looks, to a geometry-only guard, exactly like a page nobody
has opened. The guard here is the union of the three signals: a line survives a re-segment
if it has manual geometry, **or** approved text, **or** a pairing.

Keeping a line is half of it. The segmenter does not know the line was kept and draws the
same ink again, so a fully approved page would come back with every approved line sitting
under a near-identical fresh twin. A fresh line whose polygon is mostly inside the kept
lines is therefore not added: what it covers is already claimed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shapely.geometry import Polygon
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.core.exceptions import NotFoundError
from backend.document.application.reading_order import (
    block_ranks,
    merge_reading_order,
    renumber,
)
from backend.document.application.regions import box_polygon, region_containing
from backend.document.infrastructure.orm_models import (
    Block,
    DocumentPart,
    Line,
    LineSource,
    LineTranscription,
    PageTranscriptionLine,
    SegmentationRun,
    Transcription,
    TranscriptionKind,
)
from backend.ml.application.model_identity import EMPTY_MODEL_IDENTITY, ModelIdentity
from backend.ml.domain.segment import CanonicalBlock, CanonicalLine, CanonicalSegmentResult


@dataclass(frozen=True)
class SegmentMergeSummary:
    blocks_count: int
    lines_count: int
    added_lines: int
    pruned_lines: int
    preserved_manual_lines: int
    # Lines whose geometry the machine drew but which a human has since approved text
    # on or paired. Reported separately so a job result still says how many polygons
    # on the page are hand-drawn.
    preserved_transcribed_lines: int = 0
    # Fresh lines the segmenter drew over ink a kept line already covers, and which were
    # therefore not added. ``added_lines`` excludes them; ``lines_count`` does not.
    skipped_covered_lines: int = 0
    # The immutable record of what this run proposed, written in the same transaction.
    # Reported so the job result carries a pointer to it even after the job is deleted
    # and the FK the other way is nulled.
    segmentation_run_id: UUID | None = None


# A fresh line at least this much inside the kept geometry is the same ink drawn again.
# Half, not "almost all": a re-run rarely reproduces a polygon exactly, and a fresh line
# that is half inside a kept one is not a new line either way. A kept numeral sitting
# inside a fresh text line's polygon covers a small fraction of it and does not trip this.
_COVERED_FRACTION = 0.5

# A segmenter's only block covering at least this fraction of the image's width and
# height is the page, not a region (blla wraps every line in one such block).
_FULL_PAGE_FRACTION = 0.9


class SegmentMergeService:
    """Apply canonical segment output to a part without touching what a human did."""

    def apply_sync(
        self,
        session: Session,
        *,
        part_id: UUID,
        canonical_segment: CanonicalSegmentResult,
        job_id: UUID,
        commit: bool = True,
        model_identity: ModelIdentity | None = None,
    ) -> SegmentMergeSummary:
        part = self._load_part(session, part_id)
        transcribed_ids = self._human_transcribed_line_ids(session, part)
        run = self._record_run(
            session,
            part=part,
            canonical_segment=canonical_segment,
            job_id=job_id,
            model_identity=model_identity or EMPTY_MODEL_IDENTITY,
        )
        pruned = _prune_unprotected(session, part, transcribed_ids)
        blocks_by_external_id = _add_blocks(session, part, canonical_segment.blocks)

        # A kept line whose block was just deleted joins the fresh region it sits in, so
        # it is placed among that region's lines rather than after every region.
        kept_block_ids = {block.id for block in pruned.kept_blocks}
        fresh_blocks = sorted(blocks_by_external_id.values(), key=lambda block: block.order)
        for line in pruned.kept_lines:
            if line.block_id is not None and line.block_id in kept_block_ids:
                continue
            container = region_containing(line.points, fresh_blocks)
            line.block_id = container.id if container is not None else None

        kept_geometry = _polygons(line.points for line in pruned.kept_lines)
        model_lines: list[Line] = []
        added_lines = 0
        skipped_covered_lines = 0
        for line_data in canonical_segment.lines:
            if _mostly_covered(line_data.points, kept_geometry):
                skipped_covered_lines += 1
                continue
            added_lines += 1
            line = _kraken_line(part.id, line_data, blocks_by_external_id, job_id, run.id)
            session.add(line)
            model_lines.append(line)

        # Kept lines still carry the numbers of the page they were kept from, and the
        # segmenter numbers its own lines from 0, so the two collide. One reading order:
        # the segmenter's sequence, with each kept line placed where it sits (ADR 0008).
        model_lines.sort(key=lambda line: line.order)
        kept_lines = sorted(pruned.kept_lines, key=lambda line: (line.order, line.created_at))
        # Region-first (ADR 0011): a human's regions and the segmenter's, by block order,
        # a kept region before a fresh one of the same order since it is older.
        kept_blocks = sorted(pruned.kept_blocks, key=lambda block: (block.order, block.created_at))
        block_orders = block_ranks(
            sorted([*kept_blocks, *blocks_by_external_id.values()], key=lambda b: b.order)
        )
        sequence = merge_reading_order(kept_lines, model_lines, block_orders)
        orders = renumber(sequence, block_orders)
        for line in sequence:
            line.order = orders[line.id]

        if commit:
            session.commit()
        return SegmentMergeSummary(
            blocks_count=len(canonical_segment.blocks),
            lines_count=len(canonical_segment.lines),
            added_lines=added_lines,
            pruned_lines=pruned.pruned_lines,
            preserved_manual_lines=pruned.preserved_manual_lines,
            preserved_transcribed_lines=pruned.preserved_transcribed_lines,
            skipped_covered_lines=skipped_covered_lines,
            segmentation_run_id=run.id,
        )

    def _record_run(
        self,
        session: Session,
        *,
        part: DocumentPart,
        canonical_segment: CanonicalSegmentResult,
        job_id: UUID,
        model_identity: ModelIdentity,
    ) -> SegmentationRun:
        """Store what the segmenter said, before anything on the page is changed.

        Written first and flushed, so the live lines below can carry its id, and in
        the caller's transaction so that the proposal and the page it produced commit
        or roll back together. There is never a run row describing lines that were
        not written, and never lines whose origin points at nothing.

        The whole canonical result goes in verbatim, including lines the merge is
        about to skip as covered. What the model proposed is the thing being
        recorded; what the merge chose to do with it is a separate fact, already in
        the job result.
        """
        run = SegmentationRun(
            part_id=part.id,
            created_by_job_id=job_id,
            model_name=model_identity.name,
            model_provider=model_identity.provider,
            model_artifact_ref=model_identity.artifact_ref,
            registry_model_id=model_identity.registry_model_id,
            registry_tag=model_identity.registry_tag,
            params=dict(model_identity.params or {}),
            image_width=part.width,
            image_height=part.height,
            line_count=len(canonical_segment.lines),
            block_count=len(canonical_segment.blocks),
            lines=[line.model_dump(mode="json") for line in canonical_segment.lines],
            blocks=[block.model_dump(mode="json") for block in canonical_segment.blocks],
        )
        session.add(run)
        session.flush()
        return run

    def _human_transcribed_line_ids(self, session: Session, part: DocumentPart) -> set[UUID]:
        """Lines on ``part`` that carry approved text or a pairing.

        Only the ground-truth layer counts: a model's own output on a line is what a
        re-segment is allowed to replace. A pairing counts on its own because pairing
        writes the ground-truth row and un-pairing removes it, so the two normally agree,
        and when they do not (imported text paired before any approval, a ground-truth
        layer created later) the pairing is still a human's decision about that line.
        """
        line_ids = [line.id for line in part.lines]
        if not line_ids:
            return set()
        transcribed = session.scalars(
            select(LineTranscription.line_id)
            .join(Transcription, Transcription.id == LineTranscription.transcription_id)
            .where(
                LineTranscription.line_id.in_(line_ids),
                Transcription.kind == TranscriptionKind.ground_truth,
            )
        )
        paired = session.scalars(
            select(PageTranscriptionLine.paired_line_id).where(
                PageTranscriptionLine.part_id == part.id,
                PageTranscriptionLine.paired_line_id.is_not(None),
            )
        )
        return set(transcribed) | {line_id for line_id in paired if line_id is not None}

    def _load_part(self, session: Session, part_id: UUID) -> DocumentPart:
        # Lock the part row for the length of the merge. Two segment jobs on the
        # same part each delete-then-reinsert its machine geometry; the callback's
        # FOR UPDATE covers only the Job row, not the part, so without this lock
        # the two merges interleave their delete/insert and leave a corrupted mix
        # of both runs' blocks and lines. The second job blocks here until the
        # first commits, then reads the fresh state and cleanly replaces it.
        result = session.execute(
            select(DocumentPart)
            .where(DocumentPart.id == part_id)
            .with_for_update()
            .options(selectinload(DocumentPart.blocks))
            .options(selectinload(DocumentPart.lines))
        )
        part = result.scalar_one_or_none()
        if part is None:
            raise NotFoundError("Part not found")
        return part


def _covers_page(box: dict, part: DocumentPart) -> bool:
    """Whether ``box`` spans at least ``_FULL_PAGE_FRACTION`` of the image both ways.

    A part with no recorded size cannot be measured, and its block is kept.
    """
    points = box_polygon(box)
    if not points or not part.width or not part.height:
        return False
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        max(xs) - min(xs) >= _FULL_PAGE_FRACTION * part.width
        and max(ys) - min(ys) >= _FULL_PAGE_FRACTION * part.height
    )


@dataclass
class _PruneResult:
    kept_lines: list[Line] = field(default_factory=list)
    kept_blocks: list[Block] = field(default_factory=list)
    preserved_manual_lines: int = 0
    preserved_transcribed_lines: int = 0
    pruned_lines: int = 0


def _prune_unprotected(
    session: Session, part: DocumentPart, transcribed_ids: set[UUID]
) -> _PruneResult:
    """Delete the machine geometry nobody touched; keep and count what a human did."""
    result = _PruneResult()
    for line in list(part.lines):
        if line.manual_geometry:
            result.preserved_manual_lines += 1
            result.kept_lines.append(line)
        elif line.id in transcribed_ids:
            result.preserved_transcribed_lines += 1
            result.kept_lines.append(line)
        else:
            session.delete(line)
            result.pruned_lines += 1
    # Blocks carry no transcription of their own, so the geometry rule is the whole
    # rule for them. A kept line whose block goes loses only its block_id (SET NULL),
    # which is the state kraken lines without a block are already in.
    for block in list(part.blocks):
        if block.manual_geometry:
            result.kept_blocks.append(block)
        else:
            session.delete(block)
    session.flush()
    return result


def _add_blocks(
    session: Session, part: DocumentPart, blocks: list[CanonicalBlock]
) -> dict[str, Block]:
    blocks_by_external_id: dict[str, Block] = {}
    for block_data in blocks:
        if len(blocks) == 1 and _covers_page(block_data.box, part):
            # Not a region: its lines are stored with no block (ADR 0011), so drawing
            # a region or a line on the page works as on a page never segmented.
            continue
        block = Block(
            part_id=part.id,
            order=block_data.order,
            box=block_data.box,
            manual_geometry=False,
        )
        session.add(block)
        blocks_by_external_id[block_data.external_id] = block
    session.flush()
    return blocks_by_external_id


def _kraken_line(
    part_id: UUID,
    line_data: CanonicalLine,
    blocks_by_external_id: dict[str, Block],
    job_id: UUID,
    run_id: UUID,
) -> Line:
    block = (
        blocks_by_external_id.get(line_data.block_external_id)
        if line_data.block_external_id is not None
        else None
    )
    return Line(
        id=uuid4(),
        part_id=part_id,
        block_id=block.id if block is not None else None,
        baseline=line_data.baseline,
        mask=line_data.mask,
        kind=line_data.kind,
        points=line_data.points,
        source=LineSource.kraken,
        source_metadata={
            **line_data.source_metadata,
            "external_id": line_data.external_id,
            "job_id": str(job_id),
        },
        # The origin link, as columns rather than as two more keys in the
        # metadata blob beside them. ``source_metadata`` keeps carrying
        # ``external_id`` and ``job_id`` for the clients that already read
        # them; these two are what the provenance diff joins on, and unlike
        # the blob they are in no request schema, so no bulk save can
        # rewrite them.
        segmentation_run_id=run_id,
        run_external_id=line_data.external_id,
        kraken_ceiling=line_data.kraken_ceiling,
        manual_geometry=False,
        order=line_data.order,
    )


def _polygons(point_lists: Iterable[list[list[float]] | None]) -> list[Polygon]:
    polygons: list[Polygon] = []
    for points in point_lists:
        polygon = _polygon(points)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def _polygon(points: list[list[float]] | None) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    # buffer(0) repairs the self-touching rings kraken's simplification can leave; a
    # polygon it cannot repair has no area and is treated as covering nothing.
    polygon = Polygon(points).buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _mostly_covered(points: list[list[float]], kept: list[Polygon]) -> bool:
    """Whether some single kept line claims at least ``_COVERED_FRACTION`` of ``points``.

    One kept line, not their union. "Is this a redraw of a line I kept" is a claim
    about one line, and summing what several neighbours claim gets it wrong in the
    direction nobody can see: two generously drawn kept lines above and below, each
    dipping a quarter into a fresh line that is neither of them, would swallow it and
    leave the page silently missing a line. The price is the other way round: a fresh
    box drawn across two kept lines is added and shows up as a visible overlap, which
    someone can delete.
    """
    if not kept:
        return False
    fresh = _polygon(points)
    if fresh is None:
        return False
    best = max(
        (polygon.intersection(fresh).area for polygon in kept if polygon.intersects(fresh)),
        default=0.0,
    )
    return best / fresh.area >= _COVERED_FRACTION
