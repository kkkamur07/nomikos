"""Which lines a region contains (ADR 0011).

A region is a block: a polygon in ``box["points"]`` drawn around one folio or column.
Drawing one in the editor assigns it every line of the page whose centre lies inside
it, so a researcher marks a folio with one rectangle instead of one click per line.
Exports read the same grouping to print a page region by region.

Pure, like ``reading_order``: no session, anything with ``points`` qualifies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

B = TypeVar("B")
T = TypeVar("T")


def box_polygon(box: dict | None) -> list[list[float]]:
    points = box.get("points") if isinstance(box, dict) else None
    return [list(point) for point in points or [] if len(point) >= 2]


def centroid(points: Sequence[Sequence[float]] | None) -> tuple[float, float] | None:
    """The mean of a polygon's vertices; good enough to say which region a line is in."""
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def contains(polygon: Sequence[Sequence[float]], x: float, y: float) -> bool:
    """Even-odd ray casting; a polygon of fewer than three points contains nothing."""
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = current[0], current[1]
        x0, y0 = previous[0], previous[1]
        if (y1 > y) != (y0 > y):
            crossing = x1 + (y - y1) * (x0 - x1) / (y0 - y1)
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def line_in_box(points: Sequence[Sequence[float]] | None, box: dict | None) -> bool:
    centre = centroid(points)
    if centre is None:
        return False
    return contains(box_polygon(box), *centre)


def _bounds(box: dict | None) -> tuple[float, float, float, float] | None:
    points = box_polygon(box)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def reads_before(first: dict | None, second: dict | None, *, rtl: bool = False) -> bool:
    """Whether region ``first`` is read before region ``second``, by where they are.

    Regions that do not overlap vertically read top to bottom, by their top edge.
    Regions side by side (a spread, two columns) read left to right, or right to left
    on a right-to-left page.
    """
    a, b = _bounds(first), _bounds(second)
    if a is None or b is None:
        return False
    if a[3] <= b[1] or b[3] <= a[1]:
        return a[1] < b[1]
    if rtl:
        return a[2] > b[2]
    return a[0] < b[0]


def region_position(existing_boxes: Sequence[dict | None], box: dict | None, *, rtl: bool) -> int:
    """Where a new region goes among ``existing_boxes`` (in page order), by position.

    Before the first existing region it reads before; after all of them otherwise.
    """
    return next(
        (index for index, other in enumerate(existing_boxes) if reads_before(box, other, rtl=rtl)),
        len(existing_boxes),
    )


def region_containing(points: Sequence[Sequence[float]] | None, blocks: Sequence[B]) -> B | None:
    """The first of ``blocks`` (in page order) whose box holds the line's centroid."""
    return next((block for block in blocks if line_in_box(points, block.box)), None)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RegionHeading:
    """The label a region is printed under in a text export (docx, plain text)."""

    label: str


def lines_by_region(blocks: Sequence[B], lines: Sequence[T]) -> list[tuple[B | None, list[T]]]:
    """``lines`` grouped by region, for exports.

    ``blocks`` in page order, each with its lines in their given (reading) order, then
    the lines in no region, under ``None``. A page with no blocks is one group, so an
    export of it is exactly what it was before regions existed. A block with no lines
    is dropped; a line whose block is not in ``blocks`` counts as in no region.
    """
    if not blocks:
        return [(None, list(lines))]
    grouped: dict[object, list[T]] = {block.id: [] for block in blocks}  # type: ignore[attr-defined]
    loose: list[T] = []
    for line in lines:
        grouped.get(getattr(line, "block_id", None), loose).append(line)
    groups: list[tuple[B | None, list[T]]] = [
        (block, grouped[block.id])  # type: ignore[attr-defined]
        for block in blocks
        if grouped[block.id]  # type: ignore[attr-defined]
    ]
    if loose:
        groups.append((None, loose))
    return groups


def region_text(
    blocks: Sequence[object], lines: Sequence[T], text_of: Callable[[T], str]
) -> list[str | RegionHeading]:
    """A page's transcribed lines, each labelled region's preceded by its heading.

    Lines with no text are dropped, as the text exports always have; a region whose
    lines are all empty gets no heading either, so it does not print as an empty folio.
    """
    out: list[str | RegionHeading] = []
    for block, members in lines_by_region(blocks, lines):
        texts = [text for text in (text_of(line) for line in members) if text.strip()]
        label = getattr(block, "label", None) if block is not None else None
        if label and texts:
            out.append(RegionHeading(label))
        out.extend(texts)
    return out
