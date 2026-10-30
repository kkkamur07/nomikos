"""Merge the lines a human picked, however many and wherever they are.

Segment health merges one pair at a time, and only a pair its own fragment
finder is still offering. That is the right rule for a fix the detector
proposed and the wrong one for a person looking at the page: the finder
demands a shared baseline, a small gap and a lopsided width ratio, so the
merges a reviewer actually wants are usually not on the list, and asking for
one comes back 422. This module is the other half: the caller names the lines,
the server decides nothing about *whether* to merge and everything about *how*.

What is kept from the segment-health path, because these are the parts that
make a destructive edit safe:

**The server computes the geometry.** The request carries ids and one optional
choice, never a polygon, so a tab open since before somebody else edited the
page cannot write an outline measured against a page that no longer exists.

**The part is locked before anything is read.** Every other writer of a part's
lines takes the same row first (see ``DocumentRepository.lock_part``), so a
re-segment or a line edit landing between the read and the commit waits instead
of interleaving.

**Nothing that carries human work is destroyed without being asked.** Here that
rule is stricter than a refusal: the line carrying the work is the one that
survives, keeping its id and therefore its transcriptions and its pairing. Only
when more than one of the selection carries work is there a real choice to make,
and then the caller is asked rather than guessed at.

"Human work" is ground-truth text or a pairing, not any text at all. A page that
has been through OCR has machine text on every line, so treating that as work
would make every merge a conflict. The machine text on the lines that do not
survive goes with them, which is the point: a prediction measured against a
polygon that no longer exists is not worth keeping.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application import line_work, segment_health
from backend.annotation.application.line_geometry import geometry_points
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.document.application.document_service import DocumentService
from backend.document.application.reading_order import block_ranks, renumber
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import DocumentPart, Line
from backend.users.infrastructure.orm_models import User

#: Enough for any real broken line, low enough that a runaway client cannot ask
#: the server to fold a whole page into one polygon. A page of 40 lines merged
#: into one is never what somebody meant, and it is not undoable.
MAX_MERGE_LINES = 20


class LineMergeTextConflict(ConflictError):
    """More than one of the chosen lines carries work, so the caller must choose.

    Carries the ids rather than only a sentence, because the client has to show
    the reviewer what each of them says before they can answer.
    """

    def __init__(self, line_ids: list[UUID]) -> None:
        self.line_ids = line_ids
        super().__init__(
            "More than one of the selected lines carries transcribed text or a pairing. "
            "Name the one to keep with keep_text_from, or merge them in smaller steps."
        )


class LineMergeService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._document_service = document_service or DocumentService(documents=self._documents)

    async def merge_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_ids: list[UUID],
        keep_text_from: UUID | None = None,
    ) -> list[Line]:
        """Fold every named line into one, and answer with the part's whole list."""
        self._check_request(line_ids, keep_text_from)
        # Access first, lock second: locking first would let any signed-in caller
        # hold a row on a part they cannot read just by naming its id.
        await self._require_part(session, user, project_id, document_id, part_id)
        await self._documents.lock_part(session, part_id)

        lines = await self._documents.list_part_lines(session, part_id)
        paired_ids = await self._documents.paired_line_ids(session, part_id)
        by_id = {line.id: line for line in lines}
        if any(line_id not in by_id for line_id in line_ids):
            # Deliberately not "which one": an id from another page is either a
            # stale tab or a probe, and neither is owed a map of what exists.
            raise NotFoundError("Line not found")

        chosen = set(line_ids)
        # Reading order, taken from the list the repository already sorted, so
        # "the first" means the same thing here as it does on the canvas.
        selected = [line for line in lines if line.id in chosen]
        survivor = self._survivor(selected, paired_ids, keep_text_from)
        removed = [line for line in selected if line.id != survivor.id]

        survivor.points = segment_health.merge_polygon_group(
            [[list(point) for point in (line.points or [])] for line in selected]
        )
        survivor.baseline = {
            "points": segment_health.merge_baseline_group(
                [geometry_points(line.baseline) for line in selected]
            )
        }
        # ``mask`` is the outline every consumer of a line reads for its bounding
        # box, and ``resolve_line_baseline_and_mask`` keeps the two in step on
        # every other write. Leaving it behind would give the merged row a box
        # around the piece it used to be.
        survivor.mask = {"points": survivor.points}
        # The flag re-segmentation reads before it clears a page. Without it the
        # next auto-segment silently undoes the merge (and takes the survivor's
        # transcriptions with it).
        survivor.manual_geometry = True
        # ``updated_at`` looks after itself (``onupdate`` on the column), as it does on
        # every other geometry path; the editor is the half that needs the caller. The
        # survivor's origin columns are left alone: it is still the line the model
        # proposed, now redrawn by a human.
        survivor.updated_by = user.id

        blocks = sorted(
            await self._documents.list_part_blocks(session, part_id), key=lambda block: block.order
        )
        self._renumber(lines, survivor, selected[0], removed, block_ranks(blocks))
        for line in removed:
            # Plain delete, exactly as ``LayoutService.delete_part_line`` does it:
            # ``page_transcription_lines.paired_line_id`` is ``ON DELETE SET NULL``,
            # so the pairing is released and the imported page text survives.
            await session.delete(line)

        await session.commit()
        return await self._documents.list_part_lines(session, part_id)

    # -- internals -------------------------------------------------------

    @staticmethod
    def _check_request(line_ids: list[UUID], keep_text_from: UUID | None) -> None:
        """The guards that need no database, so a bad request costs no lock."""
        if len(line_ids) < 2:
            raise ValidationError("Merging needs at least two lines")
        if len(set(line_ids)) != len(line_ids):
            raise ValidationError("Line ids must be distinct")
        if len(line_ids) > MAX_MERGE_LINES:
            raise ValidationError(f"Cannot merge more than {MAX_MERGE_LINES} lines at once")
        if keep_text_from is not None and keep_text_from not in set(line_ids):
            raise ValidationError("keep_text_from must be one of the lines being merged")

    @staticmethod
    def _survivor(selected: list[Line], paired_ids: set[UUID], keep_text_from: UUID | None) -> Line:
        """Which row keeps its id, and therefore its text and its pairing."""
        worked = [line for line in selected if line_work.is_paired(line, paired_ids)]
        if keep_text_from is not None:
            keeper = next(line for line in selected if line.id == keep_text_from)
            if worked and keeper not in worked:
                # Honouring this would delete every line that does carry work and
                # keep the one that does not: the exact loss the 409 exists to stop.
                raise ValidationError(
                    "keep_text_from must name a line that carries transcribed text or a pairing"
                )
            return keeper
        if len(worked) > 1:
            raise LineMergeTextConflict([line.id for line in worked])
        if worked:
            return worked[0]
        return selected[0]

    @staticmethod
    def _renumber(
        lines: list[Line],
        survivor: Line,
        first: Line,
        removed: list[Line],
        block_orders: dict,
    ) -> None:
        """Give what is left a contiguous run of positions again, region first.

        The merged line takes the topmost position of the selection rather than
        the survivor's own: its outline now starts where that line started, and a
        polygon covering rows 3 to 7 sitting at position 7 reads as out of order
        in every export. It takes that line's region with it. A selection can
        cross regions, and a survivor kept for its text may sit in a later one;
        left in its own region at an earlier position it would be numbered ahead
        of lines the region-first order (ADR 0011) puts before it, and the canvas
        would disagree with PAGE XML, which groups lines by region.

        Subtracting one gap per deleted row above, the way ``_close_gap`` does,
        is right for a single delete and wrong here: the survivor also vacates
        its own old position, so a line that sat *between* two merged ones ends
        up sharing a position with the merged line (#114 again, by a third
        route). Walking what remains and numbering it is the version that cannot
        produce a duplicate, and it leaves a page whose orders were already
        contiguous exactly where ``_close_gap`` would have.
        """
        removed_ids = {line.id for line in removed}
        survivor.order = first.order
        survivor.block_id = first.block_id
        remaining = sorted(
            (line for line in lines if line.id not in removed_ids), key=lambda line: line.order
        )
        orders = renumber(remaining, block_orders)  # type: ignore[type-var]
        for line in remaining:
            line.order = orders[line.id]

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


__all__ = ["MAX_MERGE_LINES", "LineMergeService", "LineMergeTextConflict"]
