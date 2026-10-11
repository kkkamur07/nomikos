"""Reading order of the lines on a page - where a line goes, and how the page is numbered.

The server owns ``Line.order``. A client that draws a line says where it is, not what
number it gets: the number is its position among the lines already on the page. Every
surface that shows a segment number reads it from this order, so the order must be
dense (0..n-1, no gaps, no duplicates) after every write.

Position is decided by horizontal overlap first, then height. A page can be a two-folio
spread or carry two columns, so "sort by y" interleaves the columns. A line of the other
column or folio never shares much of its x-range with the new line, so only lines of the
same column take part in the comparison; a line that overlaps nothing goes last.

Regions come first (ADR 0011). When a page has blocks, its lines are numbered region by
region: every line of the block with the lowest order, then the next block, and the lines
in no block last. Inside one region the geometric rule above applies unchanged. Callers
pass ``block_orders`` (block id to its rank on the page); without it, or with an empty
one, every line is in the same group and nothing differs from a page with no regions.

Everything here is pure: it takes anything with ``points`` and ``baseline`` (an ORM
``Line`` qualifies; ``block_id`` is read when present) and never touches a session.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

# Two lines are in the same column when they share at least this fraction of the
# narrower one's width. Lines of one column are rarely the same length (a short last
# line of a paragraph, a numeral in the margin), so the narrower one is the yardstick.
_COLUMN_OVERLAP = 0.3


class ReadingLine(Protocol):
    """A line as ordering reads it. ``block_id`` is optional and read with ``getattr``."""

    points: list[list[float]] | None
    baseline: dict | None


class IdentifiedLine(ReadingLine, Protocol):
    id: Hashable


L = TypeVar("L", bound=ReadingLine)
I = TypeVar("I", bound=IdentifiedLine)  # noqa: E741


@dataclass
class LineShape:
    """A line as reading order sees it, for callers that have no ORM row yet."""

    id: Hashable = None
    order: int = 0
    points: list[list[float]] | None = field(default_factory=list)
    baseline: dict | None = None
    block_id: Hashable = None


BlockOrders = Mapping[Hashable, int]


def block_ranks(blocks: Iterable[object]) -> dict[Hashable, int]:
    """Rank of each block on its page, from blocks already sorted by ``(order, created_at)``.

    A rank rather than the stored ``order``: two blocks can share an order, and the
    lines of each must still form one contiguous run.
    """
    return {block.id: rank for rank, block in enumerate(blocks)}  # type: ignore[attr-defined]


def _group(line: object, block_orders: BlockOrders | None) -> float:
    """The region a line is numbered in: its block's rank, or after every block."""
    if not block_orders:
        return 0
    block_id = getattr(line, "block_id", None)
    if block_id is None or block_id not in block_orders:
        return math.inf
    return block_orders[block_id]


def region_sequence(lines: Sequence[L], block_orders: BlockOrders | None = None) -> list[L]:
    """``lines`` regrouped region-first, keeping the given sequence inside each region."""
    if not block_orders:
        return list(lines)
    return sorted(lines, key=lambda line: _group(line, block_orders))


def _x_range(line: ReadingLine) -> tuple[float, float] | None:
    points = line.points or _baseline_points(line)
    if not points:
        return None
    xs = [point[0] for point in points]
    return min(xs), max(xs)


def _baseline_points(line: ReadingLine) -> list[list[float]]:
    baseline = line.baseline or {}
    points = baseline.get("points") if isinstance(baseline, dict) else None
    return points or []


def _y_range(line: ReadingLine) -> tuple[float, float] | None:
    points = line.points or _baseline_points(line)
    if not points:
        return None
    ys = [point[1] for point in points]
    return min(ys), max(ys)


def _centre_y(line: ReadingLine) -> float | None:
    """The baseline's mean height, or the polygon's middle row when there is no baseline."""
    baseline = _baseline_points(line)
    if baseline:
        return sum(point[1] for point in baseline) / len(baseline)
    if not line.points:
        return None
    ys = [point[1] for point in line.points]
    return (min(ys) + max(ys)) / 2


def _same_column(a: tuple[float, float], b: tuple[float, float]) -> bool:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return False
    narrower = min(a[1] - a[0], b[1] - b[0])
    if narrower <= 0:
        return True
    return overlap >= _COLUMN_OVERLAP * narrower


def _beside_wins(gap: float, height: float, column_gap: float) -> bool:
    """Whether a line on the new line's row decides its place over its own column.

    With no same-column line at all, the row neighbour always does. Otherwise only one
    right beside it (no further off sideways than it is tall; the gutter of a spread is
    wider) and nearer than the column's nearest line is vertically, so a line low on a
    short verso stays on the verso although recto lines share its height.
    """
    if math.isinf(column_gap):
        return True
    return gap <= height and gap < column_gap


def insertion_index(
    existing_lines: Sequence[ReadingLine],
    new_line: ReadingLine,
    block_orders: BlockOrders | None = None,
) -> int:
    """Where ``new_line`` belongs in ``existing_lines``, which are in reading order.

    After the nearest same-column line above it; failing that, before the nearest
    same-column line below it; failing that, at the end. A line beside another one (a
    note in the margin of line 2) goes right after that one instead: a margin note
    shares no column, or only one far down the page, but it does share a row (see
    ``_beside_wins``). With ``block_orders`` only
    lines of the new line's own region are compared, and "the end" is the end of that
    region (``existing_lines`` must then already be region-first).
    """
    group = _group(new_line, block_orders)
    end = next(
        (index for index, line in enumerate(existing_lines) if _group(line, block_orders) > group),
        len(existing_lines),
    )
    new_x = _x_range(new_line)
    new_y = _centre_y(new_line)
    if new_x is None or new_y is None:
        return end

    above: tuple[float, int] | None = None
    below: tuple[float, int] | None = None
    # The nearest line of the same column, by vertical distance between centres.
    column_gap = math.inf
    # The nearest line of another column whose height spans the new line's centre, by
    # horizontal gap, with its height; on a tie the later one in reading order wins.
    beside: tuple[float, int, float] | None = None
    for index, line in enumerate(existing_lines):
        if _group(line, block_orders) != group:
            continue
        x_range = _x_range(line)
        y = _centre_y(line)
        if x_range is None or y is None:
            continue
        if not _same_column(new_x, x_range):
            y_range = _y_range(line)
            if y_range is not None and y_range[0] <= new_y <= y_range[1]:
                gap = max(x_range[0] - new_x[1], new_x[0] - x_range[1], 0.0)
                if beside is None or gap <= beside[0]:
                    beside = (gap, index, y_range[1] - y_range[0])
            continue
        column_gap = min(column_gap, abs(y - new_y))
        if y < new_y:
            # Closest from above; on a tie the later one in reading order wins, so a
            # line goes after the whole run it sits under.
            if above is None or y >= above[0]:
                above = (y, index)
        elif below is None or y < below[0]:
            below = (y, index)

    if beside is not None and _beside_wins(beside[0], beside[2], column_gap):
        return beside[1] + 1
    if above is not None:
        return above[1] + 1
    if below is not None:
        return below[1]
    return end


def renumber(lines: Sequence[I], block_orders: BlockOrders | None = None) -> dict[Hashable, int]:
    """Dense 0-based orders for ``lines``: region-first, else the sequence given."""
    return {line.id: index for index, line in enumerate(region_sequence(lines, block_orders))}


def merge_reading_order(
    kept_lines: Sequence[L],
    model_lines: Sequence[L],
    block_orders: BlockOrders | None = None,
) -> list[L]:
    """One reading order for a re-segmented page.

    The segmenter's lines keep the order it gave them (regrouped region-first); each
    line a human kept is placed among them by position, in its own previous order so
    that two kept lines in one gap stay in the order they had.
    """
    sequence = region_sequence(model_lines, block_orders)
    for line in kept_lines:
        sequence.insert(insertion_index(sequence, line, block_orders), line)
    return sequence
