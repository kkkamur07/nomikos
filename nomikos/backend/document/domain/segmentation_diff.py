"""What the model proposed for a page against what the humans left behind.

Pure geometry and set arithmetic over two inputs: one immutable
``segmentation_runs`` row, and the page's live ``lines`` as they stand now. No
session, no ORM, no clock. Everything the retraining side wants to know about a
page is derivable from that pair, which is why there is no tombstone table:

* a run line whose ``external_id`` has no live line was **deleted** - either a
  false positive somebody removed, or a fragment a merge folded into its
  neighbour. Both look the same from here, and deliberately so: the record that
  distinguishes them is the neighbour, which shows up as *corrected* with a
  polygon that grew. Its original geometry is carried on the entry either way,
  so the deletion is never a silent one;
* a live line with no origin in this run was **added** by a human: geometry the
  model missed, or a line kept from an older run that this one did not redraw;
* a live line whose origin is in the run but whose polygon differs was
  **corrected**, and both geometries come back so the pair can be scored;
* everything else is **unchanged**, which is the signal that the model got the
  line right and nobody had to touch it.

A split is one run line against several live lines. It is a correction, not two
of anything, so ``CorrectedLine`` carries a *list* of live ids and their
polygons rather than one: ``split`` is what a one-to-many correction is called,
and inventing a fifth class for it would mean every consumer had to handle the
same case twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from uuid import UUID

Points = list[list[float]]


@dataclass(frozen=True)
class RunLine:
    """One line as the segmenter proposed it, read out of the run's JSONB."""

    external_id: str
    order: int
    points: Points


@dataclass(frozen=True)
class LiveLine:
    """One line as it stands on the page now."""

    id: UUID
    order: int
    points: Points
    #: The run this line came from, and its id inside that run. Both null for a
    #: line a human drew and for anything that predates the run record.
    segmentation_run_id: UUID | None = None
    run_external_id: str | None = None


@dataclass(frozen=True)
class UnchangedLine:
    external_id: str
    line_id: UUID


@dataclass(frozen=True)
class CorrectedLine:
    external_id: str
    model_points: Points
    #: One id for an ordinary correction, several when the line was split.
    line_ids: tuple[UUID, ...]
    human_points: tuple[Points, ...]


@dataclass(frozen=True)
class DeletedLine:
    external_id: str
    order: int
    model_points: Points


@dataclass(frozen=True)
class AddedLine:
    line_id: UUID
    order: int
    points: Points


@dataclass(frozen=True)
class SegmentationDiff:
    unchanged: tuple[UnchangedLine, ...]
    corrected: tuple[CorrectedLine, ...]
    deleted: tuple[DeletedLine, ...]
    added: tuple[AddedLine, ...]


def _normalised(points: Points | None) -> tuple[tuple[float, float], ...]:
    """Compare polygons as the numbers they are, not as the containers they arrived in.

    The two sides come from different round trips - one out of a JSONB column
    written by the merge, one out of a JSONB column written by the editor - and a
    list against a tuple, or an int against the same value as a float, is not a
    correction anybody made.
    """
    if not points:
        return ()
    return tuple((float(point[0]), float(point[1])) for point in points)


def diff_segmentation_run(
    run_lines: Iterable[RunLine],
    live_lines: Iterable[LiveLine],
    *,
    run_id: UUID,
) -> SegmentationDiff:
    """Classify every proposed and every live line on one page against one run."""
    proposed: list[RunLine] = list(run_lines)
    live: list[LiveLine] = list(live_lines)

    by_external: dict[str, list[LiveLine]] = {}
    orphans: list[LiveLine] = []
    for line in live:
        # The run id is checked as well as the external id. External ids are only
        # unique inside one run, so matching on the string alone would pair a line
        # from last week's run with this week's proposal of the same name and
        # report a correction nobody made.
        if line.segmentation_run_id == run_id and line.run_external_id:
            by_external.setdefault(line.run_external_id, []).append(line)
        else:
            orphans.append(line)

    unchanged: list[UnchangedLine] = []
    corrected: list[CorrectedLine] = []
    deleted: list[DeletedLine] = []

    for proposal in sorted(proposed, key=lambda item: (item.order, item.external_id)):
        matches = sorted(
            by_external.get(proposal.external_id, ()), key=lambda item: (item.order, str(item.id))
        )
        if not matches:
            deleted.append(
                DeletedLine(
                    external_id=proposal.external_id,
                    order=proposal.order,
                    model_points=proposal.points,
                )
            )
            continue
        model_points = _normalised(proposal.points)
        if len(matches) == 1 and _normalised(matches[0].points) == model_points:
            unchanged.append(UnchangedLine(external_id=proposal.external_id, line_id=matches[0].id))
            continue
        corrected.append(
            CorrectedLine(
                external_id=proposal.external_id,
                model_points=proposal.points,
                line_ids=tuple(match.id for match in matches),
                human_points=tuple(match.points for match in matches),
            )
        )

    added = tuple(
        AddedLine(line_id=line.id, order=line.order, points=line.points)
        for line in sorted(orphans, key=lambda item: (item.order, str(item.id)))
    )
    return SegmentationDiff(
        unchanged=tuple(unchanged),
        corrected=tuple(corrected),
        deleted=tuple(deleted),
        added=added,
    )


def run_lines_from_payload(payload: Sequence[dict] | None) -> list[RunLine]:
    """Read the stored canonical output back into the shape the diff takes.

    Entries without an ``external_id`` are skipped rather than raised on. The
    column is written from a validated ``CanonicalSegmentResult``, so this cannot
    happen today; a run written by an older or a hand-rolled writer would
    otherwise take down the whole page's diff over one unusable entry.
    """
    lines: list[RunLine] = []
    for index, entry in enumerate(payload or ()):
        external_id = entry.get("external_id")
        if not external_id:
            continue
        lines.append(
            RunLine(
                external_id=str(external_id),
                order=int(entry.get("order", index)),
                points=list(entry.get("points") or []),
            )
        )
    return lines


__all__ = [
    "AddedLine",
    "CorrectedLine",
    "DeletedLine",
    "LiveLine",
    "RunLine",
    "SegmentationDiff",
    "UnchangedLine",
    "diff_segmentation_run",
    "run_lines_from_payload",
]
