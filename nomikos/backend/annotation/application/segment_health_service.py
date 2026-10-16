"""Offer kraken's systematic fixes to a reviewer, and apply the one they pick.

`segment_health` is pure geometry and knows nothing about the database. This is
the layer between it and a page: it reads the part's lines, measures the page,
asks for the three kinds of fix, and writes back exactly the one a human chose.

Two rules shape the whole module.

**The server recomputes the geometry; the client only names a choice.** An apply
request carries line ids and an operation, never a polygon. The fix is derived
again from the rows as they are at that moment and refused if it is no longer on
offer. A client that has been open in a tab since before someone else edited the
page therefore cannot write geometry measured against a page that no longer
exists, which is the failure a "here are the points, save them" endpoint invites.

**Nothing that carries human work is destroyed.** A merge keeps the primary
row's id so its transcription and pairing survive. A delete refuses on any line
with text or a pairing. A pairing is read from ``page_transcription_lines``, not
inferred from the transcription layers, because the two can disagree: see
``_is_paired``. Geometry this module writes is marked
``manual_geometry`` so a later re-segment treats it as a human's work rather
than kraken's, which is the flag re-segmentation reads before it clears a page.

Both destructive paths re-check the has-someone-worked-on-this rule that
`segment_health` already applied when it decided what to offer. Those second checks are backstops, not
the live defence: as the finders stand today neither can fire, because a
fragment or suspect carrying text is never offered in the first place. They are
here so that relaxing a *flagging* rule cannot silently become permission to
delete, and each says so at the point it is written.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application import segment_health
from backend.annotation.application.line_geometry import geometry_points
from backend.annotation.application.segment_health import (
    FragmentMerge,
    OverlapTrim,
    Segment,
    SpanningSplit,
    Suspect,
)
from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.application.document_service import DocumentService
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    DocumentPart,
    Line,
    LineGeometryKind,
    TranscriptionKind,
)
from backend.users.infrastructure.orm_models import User


@dataclass(frozen=True)
class SegmentHealthReport:
    """Everything wrong with one page, with the fix for each already built."""

    part_id: UUID
    page_width: float
    page_height: float
    #: False when the part carries no stored dimensions and the page had to be
    #: measured from the segments themselves. Every threshold is relative, so
    #: the findings stay meaningful, but a page whose ink stops short of its
    #: edges reads as narrower than it is and the column bands shift with it.
    measured_page: bool
    line_count: int
    considered_count: int
    suspects: list[Suspect]
    spanning: list[SpanningSplit]
    fragments: list[FragmentMerge]
    overlaps: list[OverlapTrim]

    @property
    def finding_count(self) -> int:
        return len(self.suspects) + len(self.spanning) + len(self.fragments) + len(self.overlaps)


class SegmentHealthService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService(documents=self._documents)

    # -- reading ---------------------------------------------------------

    async def report(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> SegmentHealthReport:
        part = await self._require_part(session, user, project_id, document_id, part_id)
        lines = await self._documents.list_part_lines(session, part_id)
        paired_ids = await self._documents.paired_line_ids(session, part_id)
        return await self._compute_report(part, lines, paired_ids, part_id)

    async def _compute_report(
        self,
        part: DocumentPart,
        lines: list[Line],
        paired_ids: set[UUID],
        part_id: UUID,
    ) -> SegmentHealthReport:
        """The pure-geometry findings, derived off the event loop.

        The finders are O(n²) over the page's lines; a large page costs seconds
        of interpreter time, which would stall every other request on the worker.
        """
        return await asyncio.to_thread(
            self._findings_report,
            part,
            [self._to_segment(line, paired_ids) for line in lines],
            len(lines),
            part_id,
        )

    def _findings_report(
        self,
        part: DocumentPart,
        segments: list[Segment],
        line_count: int,
        part_id: UUID,
    ) -> SegmentHealthReport:
        usable = [segment for segment in segments if len(segment.points) >= 3]
        width, height, measured = self._page_size(part, usable)
        if not usable or width <= 0 or height <= 0:
            return SegmentHealthReport(
                part_id=part_id,
                page_width=width,
                page_height=height,
                measured_page=measured,
                line_count=line_count,
                considered_count=len(usable),
                suspects=[],
                spanning=[],
                fragments=[],
                overlaps=[],
            )
        stats = segment_health.page_stats(usable, width, height)
        return SegmentHealthReport(
            part_id=part_id,
            page_width=width,
            page_height=height,
            measured_page=measured,
            line_count=line_count,
            considered_count=len(usable),
            suspects=segment_health.find_suspects(usable, stats),
            spanning=segment_health.find_spanning(usable, stats),
            fragments=segment_health.find_fragments(usable, stats),
            overlaps=segment_health.find_overlaps(usable, stats),
        )

    # -- applying --------------------------------------------------------

    async def apply_split(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
    ) -> list[Line]:
        """Cut a segment that swallowed two columns, at the gutter midpoint."""
        # No pairing set: neither path deletes a row, so neither can destroy one.
        report, lines, _paired_ids = await self._recompute(
            session, user, project_id, document_id, part_id
        )
        split = next((item for item in report.spanning if item.line_id == str(line_id)), None)
        if split is None or not split.pieces:
            raise ValidationError("This segment is no longer offered a column split")

        by_id = {str(line.id): line for line in lines}
        original = by_id[str(line_id)]
        first_points, first_baseline = split.pieces[0]
        original.points = first_points
        original.baseline = {"points": first_baseline}
        original.mask = {"points": first_points}
        original.manual_geometry = True

        fresh: list[Line] = []
        for points, baseline in split.pieces[1:]:
            fresh.append(
                Line(
                    part_id=part_id,
                    block_id=original.block_id,
                    baseline={"points": baseline},
                    mask={"points": points},
                    kind=LineGeometryKind.polygon,
                    points=points,
                    source=original.source,
                    source_metadata=original.source_metadata,
                    manual_geometry=True,
                    order=original.order,
                )
            )
        for line in fresh:
            session.add(line)
        # New pieces sit immediately after the row they came from, and everything
        # below shifts down. Appending at max(order) instead would put the right
        # column at the end of the page, and reusing the original's order would
        # leave two lines claiming one position (#114).
        self._insert_after(lines, original, fresh)
        await session.commit()
        return await self._documents.list_part_lines(session, part_id)

    async def apply_merge(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        primary_id: UUID,
        fragment_id: UUID,
    ) -> list[Line]:
        """Fold a fragment back into the line it broke off, primary keeps its id."""
        report, lines, paired_ids = await self._recompute(
            session, user, project_id, document_id, part_id
        )
        merge = next(
            (
                item
                for item in report.fragments
                if item.primary_id == str(primary_id) and item.fragment_id == str(fragment_id)
            ),
            None,
        )
        if merge is None:
            raise ValidationError("This pair is no longer offered a merge")

        by_id = {str(line.id): line for line in lines}
        primary = by_id[str(primary_id)]
        fragment = by_id[str(fragment_id)]
        # Unreachable as `find_fragments` stands: it skips any fragment with
        # text or a pairing, so a merge carrying one is never offered and the
        # lookup above has already raised. Kept because this is the line that
        # actually deletes a row, and the rule that makes it safe lives in
        # another module that nothing forces to keep it.
        if self._has_text(fragment) or self._is_paired(fragment, paired_ids):
            raise ValidationError(
                "The fragment carries transcribed text or a pairing; merging would "
                "delete it. Move the text to the larger piece first."
            )

        primary.points = merge.points
        primary.baseline = {"points": merge.baseline}
        primary.mask = {"points": merge.points}
        primary.manual_geometry = True
        await session.delete(fragment)
        self._close_gap(lines, fragment)
        await session.commit()
        return await self._documents.list_part_lines(session, part_id)

    async def apply_trim(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        upper_id: UUID,
        lower_id: UUID,
    ) -> list[Line]:
        """Cut two overlapping masks apart, midway between their baselines."""
        # No pairing set: neither path deletes a row, so neither can destroy one.
        report, lines, _paired_ids = await self._recompute(
            session, user, project_id, document_id, part_id
        )
        trim = next(
            (
                item
                for item in report.overlaps
                if item.upper_id == str(upper_id) and item.lower_id == str(lower_id)
            ),
            None,
        )
        if trim is None:
            raise ValidationError("This pair is no longer offered a trim")
        if trim.duplicate or not trim.upper_points or not trim.lower_points:
            # One line drawn twice. Cutting between the baselines halves a
            # duplicate rather than separating two lines, so the module offers
            # no trim and neither do we; a human deletes one of them.
            raise ValidationError(
                "These two are one line drawn twice, not two that overlap. "
                "Delete one instead of trimming both."
            )

        by_id = {str(line.id): line for line in lines}
        upper = by_id[str(upper_id)]
        lower = by_id[str(lower_id)]
        upper.points = trim.upper_points
        lower.points = trim.lower_points
        upper.mask = {"points": trim.upper_points}
        lower.mask = {"points": trim.lower_points}
        upper.manual_geometry = True
        lower.manual_geometry = True
        await session.commit()
        return await self._documents.list_part_lines(session, part_id)

    async def apply_delete(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
    ) -> list[Line]:
        """Delete a flagged suspect, only ever on an explicit request."""
        report, lines, paired_ids = await self._recompute(
            session, user, project_id, document_id, part_id
        )
        if not any(item.line_id == str(line_id) for item in report.suspects):
            raise ValidationError("This segment is not flagged as a suspect")

        by_id = {str(line.id): line for line in lines}
        target = by_id[str(line_id)]
        # Same backstop as the merge path, and unreachable for the same reason:
        # find_suspects already skips lines with text or a pairing. The two
        # checks answer different questions, though. That one decides what to
        # show; this one decides what may be destroyed, and a future change to
        # the flagging rule must not quietly become a deletion rule.
        if self._has_text(target) or self._is_paired(target, paired_ids):
            raise ValidationError("This line carries transcribed text and will not be deleted")

        await session.delete(target)
        self._close_gap(lines, target)
        await session.commit()
        return await self._documents.list_part_lines(session, part_id)

    # -- internals -------------------------------------------------------

    async def _recompute(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[SegmentHealthReport, list[Line], set[UUID]]:
        """Re-derive the offer, under a lock held until the caller commits.

        The lock is taken before anything is read. Recomputing already stops a
        stale *client* from writing geometry measured against an older page, but
        on its own it leaves the server's own window open: between this read and
        the commit below, a re-segment, an editor line edit or a second apply on
        the same part can land, and the edit is then written over geometry it
        was never derived from. Holding the part row from before the read makes
        them wait for each other instead. All three now take it; see
        ``DocumentRepository.lock_part`` for the full set, which is the thing
        that has to stay complete for any of this to hold.

        Every apply path goes through here and then writes, so the lock lives
        here rather than in the four callers: that is what makes it impossible
        to add a fifth that forgets to take it.

        Access is checked before the lock is taken, not after. Locking first
        would let any signed-in caller hold a row on a part they cannot read,
        just by naming its id.
        """
        part = await self._require_part(session, user, project_id, document_id, part_id)
        await self._documents.lock_part(session, part_id)
        lines = await self._documents.list_part_lines(session, part_id)
        paired_ids = await self._documents.paired_line_ids(session, part_id)
        report = await self._compute_report(part, lines, paired_ids, part_id)
        return report, lines, paired_ids

    def _to_segment(self, line: Line, paired_ids: set[UUID]) -> Segment:
        return Segment(
            id=str(line.id),
            points=[list(point) for point in (line.points or [])],
            baseline=geometry_points(line.baseline),
            manual_geometry=bool(line.manual_geometry),
            has_text=self._has_text(line),
            is_paired=self._is_paired(line, paired_ids),
        )

    @staticmethod
    def _has_text(line: Line) -> bool:
        """Some text on some layer, not just the approved one.

        A draft nobody has approved yet is still someone's work, and a model
        prediction is the thing a reviewer is about to correct. Restricting this
        to ground truth would make every un-approved line deletable.
        """
        return any(
            isinstance(entry.text, str) and entry.text.strip()
            for entry in (line.transcriptions or [])
        )

    @staticmethod
    def _is_paired(line: Line, paired_ids: set[UUID]) -> bool:
        """A human's decision about this line, in either of the two places it lands.

        A non-blank ground-truth transcription is the usual one. A row in
        ``page_transcription_lines`` pointing here is the other, and the first
        does not imply it: pairing writes the ground-truth row and un-pairing
        removes it, so the two agree most of the time, but text imported and
        paired before anyone approved it carries the pairing without the layer.
        Reading only the layers would call such a line untouched and offer it up
        for deletion, discarding the pairing with it.

        ``segment_merge_service._human_transcribed_line_ids`` takes the same union for
        the same reason, and re-segmentation trusts it to decide what to keep.
        """
        if line.id in paired_ids:
            return True
        return any(
            entry.transcription is not None
            and entry.transcription.kind == TranscriptionKind.ground_truth
            and isinstance(entry.text, str)
            and entry.text.strip()
            for entry in (line.transcriptions or [])
        )

    @staticmethod
    def _page_size(part: DocumentPart, segments: list[Segment]) -> tuple[float, float, bool]:
        width = part.width
        height = part.height
        if width and height:
            return float(width), float(height), True
        # Dimensions are nullable, and a part imported before they were recorded
        # has none. Measuring the ink is worse than reading the file, but it is
        # far better than refusing to look at the page at all.
        extent_x = max((point[0] for segment in segments for point in segment.points), default=0.0)
        extent_y = max((point[1] for segment in segments for point in segment.points), default=0.0)
        return float(width or extent_x), float(height or extent_y), False

    @staticmethod
    def _insert_after(lines: list[Line], anchor: Line, fresh: list[Line]) -> None:
        """Give `fresh` the positions just after `anchor`, shifting the rest down."""
        if not fresh:
            return
        for line in lines:
            if line is not anchor and line.order > anchor.order:
                line.order += len(fresh)
        for offset, line in enumerate(fresh, start=1):
            line.order = anchor.order + offset

    @staticmethod
    def _close_gap(lines: list[Line], removed: Line) -> None:
        """Pull everything below a deleted row up, so orders stay contiguous."""
        for line in lines:
            if line is not removed and line.order > removed.order:
                line.order -= 1

    async def _require_part(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> DocumentPart:
        document = await self._document_service.get_document(session, user, project_id, document_id)
        part = await self._documents.get_part(session, part_id)
        if part is None or part.document_id != document.id:
            raise NotFoundError("Part not found")
        return part


__all__ = ["SegmentHealthReport", "SegmentHealthService"]
