"""Reading order of the lines on a page - where a line goes, and how the page is numbered.

The server owns ``Line.order``. A client that draws a line says where it is, not what
number it gets: the number is its position among the lines already on the page. Every
surface that shows a segment number reads it from this order, so the order must be
dense (0..n-1, no gaps, no duplicates) after every write.

Position is decided by horizontal overlap first, then height. A page can be a two-folio
spread or carry two columns, so "sort by y" interleaves the columns. A line of the other
column or folio never shares much of its x-range with the new line, so only lines of the
same column take part in the comparison; a line that overlaps nothing goes last.

Everything here is pure: it takes anything with ``points`` and ``baseline`` (an ORM
``Line`` qualifies) and never touches a session.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

# Two lines are in the same column when they share at least this fraction of the
# narrower one's width. Lines of one column are rarely the same length (a short last
# line of a paragraph, a numeral in the margin), so the narrower one is the yardstick.
_COLUMN_OVERLAP = 0.3


class ReadingLine(Protocol):
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


def insertion_index(existing_lines: Sequence[ReadingLine], new_line: ReadingLine) -> int:
    """Where ``new_line`` belongs in ``existing_lines``, which are in reading order.

    After the nearest same-column line above it; failing that, before the nearest
    same-column line below it; failing that, at the end.
    """
    new_x = _x_range(new_line)
    new_y = _centre_y(new_line)
    if new_x is None or new_y is None:
        return len(existing_lines)

    above: tuple[float, int] | None = None
    below: tuple[float, int] | None = None
    for index, line in enumerate(existing_lines):
        x_range = _x_range(line)
        y = _centre_y(line)
        if x_range is None or y is None or not _same_column(new_x, x_range):
            continue
        if y < new_y:
            # Closest from above; on a tie the later one in reading order wins, so a
            # line goes after the whole run it sits under.
            if above is None or y >= above[0]:
                above = (y, index)
        elif below is None or y < below[0]:
            below = (y, index)

    if above is not None:
        return above[1] + 1
    if below is not None:
        return below[1]
    return len(existing_lines)


def renumber(lines: Sequence[I]) -> dict[Hashable, int]:
    """Dense 0-based orders for ``lines``, keeping the sequence they are given in."""
    return {line.id: index for index, line in enumerate(lines)}


def merge_reading_order(kept_lines: Sequence[L], model_lines: Sequence[L]) -> list[L]:
    """One reading order for a re-segmented page.

    The segmenter's lines keep the order it gave them; each line a human kept is
    placed among them by position, in its own previous order so that two kept lines
    in one gap stay in the order they had.
    """
    sequence = list(model_lines)
    for line in kept_lines:
        sequence.insert(insertion_index(sequence, line), line)
    return sequence
