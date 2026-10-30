"""Page geometry: the blocks and lines drawn over a document part.

One responsibility: where the writing is on the page. The *text* of a line belongs to the
transcription module; the only reason this one knows ground truth exists at all is
``replace_part_lines``, which accepts an ``approved_text`` alongside each polygon so the
editor can save a redrawn page and its corrections in one request. That single crossing is
served by an injected :class:`GroundTruthText`, not by inheriting it.

``manual_geometry`` is the through-line. Every single-shape edit here asserts human
authorship, because a later model run must not silently overwrite a shape a researcher
drew. The two bulk paths, ``replace_part_lines`` and ``reset_part_layout``, deliberately
do not, since they exist to hand geometry back to the model.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.annotation.application.line_geometry import resolve_line_baseline_and_mask
from backend.core.exceptions import NotFoundError, ValidationError
from backend.document.application.document_access import DocumentAccess
from backend.document.application.ground_truth import GroundTruthText
from backend.document.application.patch_fields import (
    BLOCK_PATCH_FIELDS,
    LINE_PATCH_FIELDS,
    reject_unknown_fields,
)
from backend.document.application.reading_order import (
    BlockOrders,
    block_ranks,
    insertion_index,
    region_sequence,
    renumber,
)
from backend.document.application.regions import line_in_box, region_containing, region_position
from backend.document.application.script_detection import detect_script, is_rtl
from backend.document.infrastructure.document_repository import DocumentRepository
from backend.document.infrastructure.orm_models import (
    Block,
    Line,
    LineGeometryKind,
    LineSource,
    LineTranscription,
)
from backend.project.infrastructure.project_repository import ProjectRepository
from backend.users.infrastructure.orm_models import User

MAX_REPLACE_PART_LINES = 10_000

# Lines whose text the page-direction guess reads; a page's script shows in a few lines.
_SCRIPT_SAMPLE_LINES = 200

# The fields that move a line's shape. A PATCH carrying only others (a region, an
# order) files the line without claiming a human drew it.
_GEOMETRY_PATCH_FIELDS = frozenset({"points", "baseline", "mask"})


def _preserved(data: dict, field: str, prior: Line | None, default: object) -> object:
    """Read an optional field from a bulk-replace payload.

    ``model_dump(exclude_unset=True)`` strips any field the client did not send,
    including ones with a schema default, so presence in ``data`` is the only
    signal of intent. Absent means keep what the row already had; the default is
    for rows that do not exist yet.
    """
    if field in data:
        return data[field]
    if prior is not None:
        return getattr(prior, field)
    return default


def _apply_reading_order(lines: list[Line], block_orders: BlockOrders | None = None) -> None:
    orders = renumber(lines, block_orders)
    for line in lines:
        line.order = orders[line.id]


def _sorted_blocks(blocks: list[Block]) -> list[Block]:
    # The repository lists by (order, created_at); a block added in this transaction has
    # no created_at yet and is the newest, so a stable sort on order puts it last of its
    # order, where the repository will list it once committed.
    return sorted(blocks, key=lambda block: block.order)


class LayoutService:
    def __init__(
        self,
        documents: DocumentRepository | None = None,
        projects: ProjectRepository | None = None,
        access: DocumentAccess | None = None,
        ground_truth: GroundTruthText | None = None,
    ) -> None:
        self._documents = documents or DocumentRepository()
        self._projects = projects or ProjectRepository()
        self._access = access or DocumentAccess(documents=self._documents, projects=self._projects)
        self._ground_truth = ground_truth or GroundTruthText(documents=self._documents)

    async def list_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> list[Line]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        return await self._documents.list_part_lines(session, context.part.id)

    async def list_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
    ) -> tuple[list[Block], list[Line]]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        blocks = await self._documents.list_part_blocks(session, context.part.id)
        lines = await self._documents.list_part_lines(session, context.part.id)
        return blocks, lines

    async def create_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int | None = None,
        box: dict,
        label: str | None = None,
        assign_contained: bool = True,
    ) -> Block:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # A block is part of the same layout segment health renumbers.
        await self._documents.lock_part(session, part_id)
        part = context.part
        blocks = _sorted_blocks(await self._documents.list_part_blocks(session, part.id))
        lines = await self._documents.list_part_lines(session, part.id)
        if order is None:
            # No explicit order: the region reads where it sits (ADR 0011), and the
            # page's regions are renumbered 0..n-1 around it.
            rtl = await self._part_reads_rtl(session, part.id)
            position = region_position([other.box for other in blocks], box, rtl=rtl)
            for rank, other in enumerate(blocks):
                other.order = rank if rank < position else rank + 1
            order = position
        block = Block(
            id=uuid4(), part_id=part.id, order=order, box=box, label=label, manual_geometry=True
        )
        session.add(block)
        if assign_contained:
            # Only lines in no region yet: drawing a region never steals lines from
            # another one; moving a line between regions is an explicit PATCH.
            for line in lines:
                if line.block_id is None and line_in_box(line.points, box):
                    line.block_id = block.id
        _apply_reading_order(lines, block_ranks(_sorted_blocks([*blocks, block])))
        await session.commit()
        await session.refresh(block)
        return block

    async def patch_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
        **updates: object,
    ) -> Block:
        reject_unknown_fields(updates, BLOCK_PATCH_FIELDS, "block patch")
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # A block is part of the same layout segment health renumbers.
        await self._documents.lock_part(session, part_id)
        block = await self._block_or_404(session, context.part.id, block_id)
        # ``updates`` already went through ``exclude_unset``: every key present was sent
        # by the client, so applying it verbatim is what makes explicit nulls meaningful.
        for key, value in updates.items():
            setattr(block, key, value)
        block.manual_geometry = True
        if "order" in updates:
            # Lines are numbered region-first (ADR 0011), so moving a region moves its lines.
            await self._renumber_part(session, context.part.id)
        await session.commit()
        await session.refresh(block)
        return block

    async def delete_part_block(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        block_id: UUID,
    ) -> None:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # A block is part of the same layout segment health renumbers.
        await self._documents.lock_part(session, part_id)
        block = await self._block_or_404(session, context.part.id, block_id)
        # The lines stay where they are and become unblocked. The foreign key would null
        # them too, but only after the renumber below has read the rows in memory.
        lines = await self._documents.list_part_lines(session, context.part.id)
        for line in lines:
            if line.block_id == block.id:
                line.block_id = None
        remaining = [
            other
            for other in await self._documents.list_part_blocks(session, context.part.id)
            if other.id != block.id
        ]
        await session.delete(block)
        _apply_reading_order(lines, block_ranks(remaining))
        await session.commit()

    async def create_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        order: int | None = None,
        kind: LineGeometryKind,
        points: list[list[float]],
        block_id: UUID | None = None,
        baseline: dict | None = None,
        mask: dict | None = None,
    ) -> Line:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # `lines.order` has only a plain index behind it, no unique constraint, so a line created
        # inside a segment health apply's snapshot keeps its order while that apply renumbers
        # around it, and the two collide with nothing raised. This is the #114 duplicate-order bug
        # arriving by a different route.
        await self._documents.lock_part(session, part_id)
        part = context.part
        blocks = _sorted_blocks(await self._documents.list_part_blocks(session, part.id))
        if block_id is not None:
            await self._block_or_404(session, part.id, block_id)
        else:
            # A line drawn without a region joins the one it sits in (ADR 0011), so a
            # line drawn on a page of regions is numbered among its neighbours.
            container = region_containing(points, blocks)
            block_id = container.id if container is not None else None
        line = Line(
            id=uuid4(),
            part_id=part.id,
            block_id=block_id,
            order=0,
            kind=kind,
            points=points,
            baseline=baseline or {"points": points},
            mask=mask or {"points": points},
            source=LineSource.manual,
            manual_geometry=True,
            updated_by=user.id,
        )
        # The server owns reading order (ADR 0008). Without an ``order`` the line goes
        # where it sits on the page; with one (undo restoring a deleted line) it goes
        # before the first line numbered at least that. Either way every line after it
        # moves down one, so the page stays numbered 0..n-1.
        block_orders = block_ranks(blocks)
        existing = region_sequence(
            await self._documents.list_part_lines(session, part.id), block_orders
        )
        if order is None:
            index = insertion_index(existing, line, block_orders)
        else:
            index = sum(1 for other in existing if other.order < order)
        session.add(line)
        _apply_reading_order([*existing[:index], line, *existing[index:]], block_orders)
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def patch_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
        **updates: object,
    ) -> Line:
        # Any geometry edit through PATCH marks the line as manually overridden; filing it
        # into a region or reordering it does not (the shape is still the model's).
        # Bulk PUT /lines and POST /layout/reset are the paths that preserve kraken source.
        reject_unknown_fields(updates, LINE_PATCH_FIELDS, "line patch")
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # Lock the part for the rest of this transaction. Segment health derives
        # a fix from the whole line list and then renumbers or deletes rows, and
        # re-segmentation replaces the list wholesale; both hold this same row.
        # A line write that skipped it would interleave with either of them, and
        # the one that committed second would win on rows it never read.
        await self._documents.lock_part(session, part_id)
        part = context.part
        line = await self._line_or_404(session, part.id, line_id)
        if "block_id" in updates and updates["block_id"] is not None:
            await self._block_or_404(session, part.id, updates["block_id"])
        # ``updates`` already went through ``exclude_unset``: every key present was sent by
        # the client. Applying it verbatim is what lets a client clear a nullable field
        # (``block_id``/``mask``); the request schema rejects nulls for the rest.
        moved_region = "block_id" in updates and updates["block_id"] != line.block_id
        for key, value in updates.items():
            setattr(line, key, value)
        if _GEOMETRY_PATCH_FIELDS & updates.keys():
            line.manual_geometry = True
            line.source = LineSource.manual
        if moved_region:
            # A line moved to another region is placed among that region's lines by
            # where it sits, as a newly drawn line would be (ADR 0011).
            block_orders = await self._block_orders(session, part.id)
            others = region_sequence(
                [
                    other
                    for other in await self._documents.list_part_lines(session, part.id)
                    if other.id != line.id
                ],
                block_orders,
            )
            others.insert(insertion_index(others, line, block_orders), line)
            _apply_reading_order(others, block_orders)
        # ``updated_at`` looks after itself (``onupdate`` on the column); this is the
        # half that needs a caller, and every path that sets it already holds the
        # user. The origin columns are untouched on purpose: a line a human redrew is
        # still the line the model proposed, and losing that link is exactly what
        # made a correction indistinguishable from a fresh drawing.
        line.updated_by = user.id
        await session.commit()
        return await self._line_or_404(session, part.id, line.id)

    async def delete_part_line(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        line_id: UUID,
    ) -> None:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # Lock the part for the rest of this transaction. Segment health derives
        # a fix from the whole line list and then renumbers or deletes rows, and
        # re-segmentation replaces the list wholesale; both hold this same row.
        # A line write that skipped it would interleave with either of them, and
        # the one that committed second would win on rows it never read.
        await self._documents.lock_part(session, part_id)
        line = await self._line_or_404(session, context.part.id, line_id)
        remaining = [
            other
            for other in await self._documents.list_part_lines(session, context.part.id)
            if other.id != line.id
        ]
        await session.delete(line)
        # Close the gap: the lines after it move up one.
        _apply_reading_order(remaining, await self._block_orders(session, context.part.id))
        await session.commit()

    async def reset_part_layout(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        line_ids: list[UUID] | None = None,
    ) -> tuple[list[Block], list[Line]]:
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # ``manual_geometry`` is the flag re-segmentation reads before it clears a
        # page, and this path and segment health set it in opposite directions.
        # Without the lock the two interleave and the flag ends up describing
        # neither of them.
        await self._documents.lock_part(session, part_id)
        part = context.part
        lines = await self._documents.list_part_lines(session, part.id)
        selected_ids = set(line_ids) if line_ids is not None else {line.id for line in lines}
        if line_ids is not None and selected_ids - {line.id for line in lines}:
            raise NotFoundError("Line not found")
        for line in lines:
            if line.id in selected_ids:
                line.manual_geometry = False
        if line_ids is None:
            blocks = await self._documents.list_part_blocks(session, part.id)
            for block in blocks:
                block.manual_geometry = False
        await session.commit()
        return await self.list_part_layout(session, user, project_id, document_id, part_id)

    async def replace_part_lines(
        self,
        session: AsyncSession,
        user: User,
        project_id: UUID,
        document_id: UUID,
        part_id: UUID,
        *,
        lines: list[dict],
        allow_new_ids: bool = False,
        restore_origin: bool = False,
    ) -> list[Line]:
        if len(lines) > MAX_REPLACE_PART_LINES:
            raise ValidationError(
                f"Cannot replace more than {MAX_REPLACE_PART_LINES} lines at once"
            )
        context = await self._access.require_part(session, user, project_id, document_id, part_id)
        # Lock the part for the rest of this transaction. Segment health derives
        # a fix from the whole line list and then renumbers or deletes rows, and
        # re-segmentation replaces the list wholesale; both hold this same row.
        # A line write that skipped it would interleave with either of them, and
        # the one that committed second would win on rows it never read.
        await self._documents.lock_part(session, part_id)
        part = context.part
        ground_truth = await self._ground_truth.layer_for(session, context.document)

        requested_ids = [line["id"] for line in lines if line.get("id") is not None]
        if len(set(requested_ids)) != len(requested_ids):
            raise ValidationError("Line ids must be unique")

        existing = await self._documents.list_part_lines(session, part.id)
        existing_by_id = {line.id: line for line in existing}
        if not allow_new_ids:
            new_supplied_ids = [
                line_id for line_id in requested_ids if line_id not in existing_by_id
            ]
            if new_supplied_ids:
                raise ValidationError("New line ids are server-generated")
        requested_id_set = set(requested_ids)
        for line in existing:
            if line.id not in requested_id_set:
                await session.delete(line)

        written: list[Line] = []
        for data in lines:
            line_id = data.get("id")
            prior = existing_by_id.get(line_id) if line_id is not None else None
            line = prior
            if line is None:
                line = Line(id=line_id or uuid4(), part_id=part.id, baseline={}, mask=None)
                session.add(line)
            written.append(line)

            points = data["points"]
            if "block_id" in data:
                block_id = data["block_id"]
                if block_id is not None:
                    await self._block_or_404(session, part.id, block_id)
                line.block_id = block_id
            elif prior is not None:
                line.block_id = prior.block_id
            # ``order`` and ``points`` are required by the request schema, so they are
            # always present. ``kind`` and ``source`` are optional and carry schema
            # defaults, which ``exclude_unset`` strips from an omitting client's payload -
            # so an absent key means "leave it alone", exactly as it does for
            # ``block_id``, ``source_metadata`` and ``kraken_ceiling`` below. Falling back
            # to the schema default instead flipped every pre-existing Kraken line on the
            # page to ``source=manual`` while keeping its ``kraken_ceiling``: a row that
            # claims a human drew it and that a model measured it. The editor sends the
            # whole page on every save, so redrawing one line rewrote the provenance of
            # all of them.
            line.order = data["order"]
            line.kind = _preserved(data, "kind", prior, LineGeometryKind.polygon)
            line.points = points
            line.baseline, line.mask = resolve_line_baseline_and_mask(
                points=points,
                payload_baseline=data.get("baseline"),
                payload_mask=data.get("mask"),
                existing_baseline=prior.baseline if prior is not None else None,
                existing_mask=prior.mask if prior is not None else None,
            )
            source = _preserved(data, "source", prior, LineSource.manual)
            line.source = source
            if "source_metadata" in data:
                line.source_metadata = data["source_metadata"]
            elif prior is not None:
                line.source_metadata = prior.source_metadata
            if "kraken_ceiling" in data:
                line.kraken_ceiling = data["kraken_ceiling"]
            elif prior is not None:
                line.kraken_ceiling = prior.kraken_ceiling
            source_value = source.value if hasattr(source, "value") else source
            line.manual_geometry = source_value == "manual"
            # A row that existed keeps its origin columns untouched, because they are
            # not in ``LineUpsertRequest`` at all. ``source_metadata`` is, which is
            # why the link does not live there: the editor posts the whole page on
            # every save, so a client that sent its own metadata would otherwise
            # rewrite the provenance of every line on the page.
            line.updated_by = user.id
            # A history restore is the one caller trusted to put an origin back: the
            # values come from a snapshot the server took, never from a request body.
            # A snapshot line with no origin leaves the row's own link alone.
            if restore_origin and data.get("segmentation_run_id") is not None:
                line.segmentation_run_id = data["segmentation_run_id"]
                line.run_external_id = data.get("run_external_id")

            await self._ground_truth.write(session, line, ground_truth, data.get("approved_text"))

        # The client's orders say the sequence; the server makes it dense. The sort is
        # stable, so lines sent with the same order keep the order they were sent in.
        _apply_reading_order(
            sorted(written, key=lambda line: line.order),
            await self._block_orders(session, part.id),
        )
        await session.commit()
        return await self._documents.list_part_lines(session, part.id)

    async def _part_reads_rtl(self, session: AsyncSession, part_id: UUID) -> bool:
        """Whether the page reads right to left, from the script of its transcribed text.

        A document has no direction field; the text's script is the only signal. A page
        with no text yet reads left to right.
        """
        texts = await session.scalars(
            select(LineTranscription.text)
            .join(Line, Line.id == LineTranscription.line_id)
            .where(Line.part_id == part_id, LineTranscription.text != "")
            .limit(_SCRIPT_SAMPLE_LINES)
        )
        return is_rtl(detect_script("".join(texts)))

    async def _block_orders(self, session: AsyncSession, part_id: UUID) -> BlockOrders:
        return block_ranks(await self._documents.list_part_blocks(session, part_id))

    async def _renumber_part(self, session: AsyncSession, part_id: UUID) -> None:
        """Regroup the page region-first, keeping each region's lines in their sequence."""
        lines = await self._documents.list_part_lines(session, part_id)
        blocks = _sorted_blocks(await self._documents.list_part_blocks(session, part_id))
        _apply_reading_order(lines, block_ranks(blocks))

    async def _block_or_404(self, session: AsyncSession, part_id: UUID, block_id: UUID) -> Block:
        block = await self._documents.get_block_in_part(session, part_id, block_id)
        if block is None:
            raise NotFoundError("Block not found")
        return block

    async def _line_or_404(self, session: AsyncSession, part_id: UUID, line_id: UUID) -> Line:
        line = await self._documents.get_line_in_part(session, part_id, line_id)
        if line is None:
            raise NotFoundError("Line not found")
        return line
