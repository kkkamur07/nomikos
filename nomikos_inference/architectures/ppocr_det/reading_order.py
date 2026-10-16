"""Column-aware reading order for PP-OCRv6 detection quads.

Manuscript pages are often spreads: two text columns side by side that must
read left page first, then right page, each top to bottom. Sorting by y alone
interleaves the columns, so quads are first grouped into columns by
horizontal overlap, columns are ordered across the page, and inside a column
quads are grouped into rows by vertical overlap and read row by row. A quad
that spans two or more columns (a running head or caption) reads as its own
band between the rows above and below it.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from nomikos_inference.architectures.ppocr_det.postprocessing import DetectedQuad

# A quad counts as narrow (a marginal note, page number or initial) when its
# width is below half the page's median quad width. Such quads must not split
# or reorder a column: they attach to the nearest column instead.
NARROW_WIDTH_FRACTION = 0.5
# Two quads share a column when their x ranges overlap by at least half of
# the narrower quad's width, merged transitively.
COLUMN_OVERLAP_FRACTION = 0.5
# A non-narrow quad wider than this multiple of the median width never founds
# a column: it joins the one column it overlaps, or reads as its own band
# when it spans two or more columns.
WIDE_WIDTH_FRACTION = 1.5
# A wide quad overlaps a column when their x ranges share more than this
# fraction of the column's width.
SPANNING_OVERLAP_FRACTION = 0.10
# Two quads in a column share a row when their y ranges overlap by at least
# half of the shorter height and they sit side by side (x overlap below half
# of the narrower width). Stacked quads in the same x range are different
# lines however much they overlap vertically.
ROW_OVERLAP_FRACTION = 0.5
# A quad taller than this multiple of its column's median line height (an
# enlarged initial or figure) joins the first row it overlaps without
# extending that row, so it cannot chain stacked lines into one row.
TALL_HEIGHT_FRACTION = 1.8


def _extents(quads: list[DetectedQuad]) -> list[tuple[float, ...]]:
    """Per quad (xmin, xmax, cx, cy, width, ymin, ymax, height)."""
    extents: list[tuple[float, ...]] = []
    for quad in quads:
        xs = [point[0] for point in quad.points]
        ys = [point[1] for point in quad.points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        extents.append(
            (
                xmin,
                xmax,
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                xmax - xmin,
                ymin,
                ymax,
                ymax - ymin,
            )
        )
    return extents


def _overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return min(max_a, max_b) - max(min_a, min_b)


def _nearest_column(ranges: list[tuple[float, float]], xmin: float, xmax: float) -> int:
    best, best_gap = 0, None
    for position, (column_xmin, column_xmax) in enumerate(ranges):
        gap = max(0.0, column_xmin - xmax, xmin - column_xmax)
        if best_gap is None or gap < best_gap:
            best, best_gap = position, gap
    return best


def _shares_row(quad: tuple[float, ...], row: list[float]) -> bool:
    """Whether a quad (its extents) belongs to a row ``[ymin, ymax, xmin, xmax]``."""
    xmin, xmax, ymin, ymax, height = quad[0], quad[1], quad[5], quad[6], quad[7]
    row_ymin, row_ymax, row_xmin, row_xmax = row
    shorter_h = min(height, row_ymax - row_ymin)
    if _overlap(ymin, ymax, row_ymin, row_ymax) < ROW_OVERLAP_FRACTION * shorter_h:
        return False
    # The side-by-side rule reuses the half-width threshold of the column
    # rule: stacked quads sharing an x range are separate lines.
    narrower_w = min(xmax - xmin, row_xmax - row_xmin)
    return _overlap(xmin, xmax, row_xmin, row_xmax) < COLUMN_OVERLAP_FRACTION * narrower_w


def _place_in_rows(
    index: int,
    quad: tuple[float, ...],
    tall: bool,
    rows: list[list[int]],
    row_ranges: list[list[float]],
) -> None:
    """Add one quad to the row it shares, or open a new row.

    A tall quad joins the first row it shares without extending it; any
    other quad is compared with the current (last) row only and widens it.
    """
    xmin, xmax, ymin, ymax = quad[0], quad[1], quad[5], quad[6]
    if tall:
        target = next(
            (position for position, row in enumerate(row_ranges) if _shares_row(quad, row)), None
        )
        if target is not None:
            rows[target].append(index)
            return
    elif row_ranges and _shares_row(quad, row_ranges[-1]):
        rows[-1].append(index)
        last = row_ranges[-1]
        last[0], last[1] = min(last[0], ymin), max(last[1], ymax)
        last[2], last[3] = min(last[2], xmin), max(last[3], xmax)
        return
    rows.append([index])
    row_ranges.append([ymin, ymax, xmin, xmax])


def _row_groups(
    members: list[int],
    extents: list[tuple[float, ...]],
    geoms: list[tuple[float, ...]],
    column_builders: set[int],
    direction: str,
) -> list[list[int]]:
    """Group one column's members into rows, top to bottom.

    Rows group transitively along the centre-y sequence: each quad is
    compared with the current row's ranges, so a tall initial met midway
    cannot chain the lines above and below it into one row. Tall quads join
    the first row they overlap and never extend it. Every tie-break is
    geometric (centre, then coordinates), never the input index, so the
    result does not depend on input order.
    """
    by_y = sorted(members, key=lambda i: (extents[i][3], extents[i][2], geoms[i]))
    heights = sorted(extents[i][7] for i in members if i in column_builders)
    median_height = heights[len(heights) // 2]
    rows: list[list[int]] = []
    row_ranges: list[list[float]] = []
    for index in by_y:
        tall = median_height > 0 and extents[index][7] > TALL_HEIGHT_FRACTION * median_height
        _place_in_rows(index, extents[index], tall, rows, row_ranges)
    # Rows read by their top-most centre y, ties broken geometrically.
    ranked = sorted(
        range(len(rows)),
        key=lambda position: (
            min(extents[i][3] for i in rows[position]),
            row_ranges[position][0],
            row_ranges[position][1],
            tuple(sorted(geoms[i] for i in rows[position])),
        ),
    )
    sign = 1.0 if direction == "ltr" else -1.0
    return [
        sorted(rows[position], key=lambda i: (sign * extents[i][2], geoms[i]))
        for position in ranked
    ]


@dataclass(frozen=True)
class ColumnLayout:
    """One column: its members, row groups and reading sequence.

    ``builders`` are the full-width members that found the column; the
    text band is their x range, while attached narrow or wide members may
    stick out past it.
    """

    members: tuple[int, ...]
    builders: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    sequence: tuple[int, ...]
    xmin: float
    xmax: float


@dataclass(frozen=True)
class PageLayout:
    """Column and row structure refinement shares with ordering."""

    columns: tuple[ColumnLayout, ...]
    column_order: tuple[int, ...]
    spanning: tuple[int, ...]
    direction: str
    centre_y: tuple[float, ...]


def _width_classes(extents: list[tuple[float, ...]]) -> tuple[list[int], list[int], list[int]]:
    """Split quad indices into (narrow, wide, builders) by the median width."""
    widths = sorted(extent[4] for extent in extents)
    median_width = widths[len(extents) // 2]
    narrow: list[int] = []
    wide: list[int] = []
    builders: list[int] = []
    for index, extent in enumerate(extents):
        if median_width > 0 and extent[4] < NARROW_WIDTH_FRACTION * median_width:
            narrow.append(index)
        elif median_width > 0 and extent[4] > WIDE_WIDTH_FRACTION * median_width:
            wide.append(index)
        else:
            builders.append(index)
    return narrow, wide, builders


def _builder_components(builders: list[int], extents: list[tuple[float, ...]]) -> list[list[int]]:
    """Connected components of builders that share a column, in builder order.

    Two builders share a column when their x ranges overlap by at least
    ``COLUMN_OVERLAP_FRACTION`` of the narrower width, merged transitively.
    A breadth-first walk tests each reached builder against all unreached
    ones at once with numpy, so a page of N stacked lines costs N vector
    passes rather than N squared Python comparisons; the components are the
    same whatever order the relation is explored in. Each component lists
    its members in ``builders`` order, and components come in the order of
    their first member.
    """
    xmins = np.array([extents[i][0] for i in builders], dtype=float)
    xmaxs = np.array([extents[i][1] for i in builders], dtype=float)
    widths = np.array([extents[i][4] for i in builders], dtype=float)
    component = np.full(len(builders), -1, dtype=int)
    for seed in range(len(builders)):
        if component[seed] >= 0:
            continue
        component[seed] = seed
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            unreached = np.flatnonzero(component < 0)
            overlap = np.minimum(xmaxs[current], xmaxs[unreached]) - np.maximum(
                xmins[current], xmins[unreached]
            )
            narrower = np.minimum(widths[current], widths[unreached])
            joined = unreached[overlap >= COLUMN_OVERLAP_FRACTION * narrower]
            component[joined] = seed
            frontier.extend(joined.tolist())
    grouped: dict[int, list[int]] = {}
    for position, index in enumerate(builders):
        grouped.setdefault(int(component[position]), []).append(index)
    return list(grouped.values())


def _column_range(extents: list[tuple[float, ...]], members: list[int]) -> tuple[float, float]:
    return min(extents[i][0] for i in members), max(extents[i][1] for i in members)


def _column_key(
    extents: list[tuple[float, ...]], members: list[int], xmin: float, xmax: float
) -> tuple[float, float, tuple[tuple[float, float], ...]]:
    # Member centres break range ties without depending on input order.
    return (xmin, xmax, tuple(sorted((extents[i][2], extents[i][3]) for i in members)))


def _attach_wide(
    wide: list[int],
    columns: list[list[int]],
    ranges: list[tuple[float, float]],
    extents: list[tuple[float, ...]],
) -> list[int]:
    """Attach wide quads in place and return the ones that span columns.

    Wide quads never merge columns: one overlapped column means membership,
    two or more means a band of its own. ``ranges`` follows each attachment,
    so a later wide quad sees the column as the earlier ones widened it.
    """
    spanning: list[int] = []
    for index in wide:
        xmin, xmax = extents[index][0], extents[index][1]
        qualifying = [
            position
            for position, (low, high) in enumerate(ranges)
            if _overlap(xmin, xmax, low, high) > SPANNING_OVERLAP_FRACTION * (high - low)
        ]
        if len(qualifying) >= 2:
            spanning.append(index)
            continue
        target = qualifying[0] if qualifying else _nearest_column(ranges, xmin, xmax)
        columns[target].append(index)
        low, high = ranges[target]
        ranges[target] = (min(low, xmin), max(high, xmax))
    return spanning


def layout_lines(quads: list[DetectedQuad], *, direction: str = "ltr") -> PageLayout:
    """Group quads into columns and rows, shared by ordering and refinement.

    Columns are in canonical left to right order with ``column_order`` giving
    the reading sequence; each column carries its row groups top to bottom
    and its flat reading ``sequence``. Spanning quads are sorted top to
    bottom. Every tie-break is geometric, so equal inputs give equal outputs
    whatever order they arrive in.
    """

    if direction not in ("ltr", "rtl"):
        raise ValueError('reading direction must be "ltr" or "rtl"')
    if not quads:
        return PageLayout(
            columns=(), column_order=(), spanning=(), direction=direction, centre_y=()
        )
    extents = _extents(quads)
    geoms = [tuple(float(c) for point in quad.points for c in point) for quad in quads]
    # The median quad itself can never be narrow or wide, so at least one
    # column builder always exists for the rest to attach to.
    narrow, wide, builders = _width_classes(extents)
    ordered_columns = sorted(
        _builder_components(builders, extents),
        key=lambda members: _column_key(extents, members, *_column_range(extents, members)),
    )
    builder_sets = [set(members) for members in ordered_columns]
    ranges = [_column_range(extents, members) for members in ordered_columns]
    spanning = _attach_wide(wide, ordered_columns, ranges, extents)
    for index in narrow:
        ordered_columns[_nearest_column(ranges, extents[index][0], extents[index][1])].append(index)

    column_order = sorted(
        range(len(ordered_columns)),
        key=lambda position: _column_key(extents, ordered_columns[position], *ranges[position]),
        reverse=(direction == "rtl"),
    )
    built_columns = []
    for position, members in enumerate(ordered_columns):
        rows = _row_groups(members, extents, geoms, builder_sets[position], direction)
        built_columns.append(
            ColumnLayout(
                members=tuple(members),
                builders=tuple(sorted(builder_sets[position])),
                rows=tuple(tuple(row) for row in rows),
                sequence=tuple(index for row in rows for index in row),
                xmin=ranges[position][0],
                xmax=ranges[position][1],
            )
        )
    return PageLayout(
        columns=tuple(built_columns),
        column_order=tuple(column_order),
        spanning=tuple(sorted(spanning, key=lambda i: (extents[i][3], extents[i][2], geoms[i]))),
        direction=direction,
        centre_y=tuple(extent[3] for extent in extents),
    )


def order_lines(quads: list[DetectedQuad], *, direction: str = "ltr") -> list[int]:
    """Return quad indices in reading order.

    Columns group transitively by x overlap; narrow quads attach to the
    nearest column without splitting it, and wide quads either join their
    one overlapped column or, when spanning two or more, read as bands of
    their own. Columns run left to right for ``"ltr"`` and right to left for
    ``"rtl"``; inside a column quads run row by row, each row across the
    page. Every tie-break is geometric, so equal inputs give equal outputs
    whatever order they arrive in.
    """

    layout = layout_lines(quads, direction=direction)
    # Spanning quads split the page into horizontal bands: everything above a
    # band's centre reads first, then the band, then everything below. The
    # spanning quads are sorted by centre y, so a line's band is the number
    # of band centres strictly above its own.
    span_cys = [layout.centre_y[i] for i in layout.spanning]
    bands: list[list[int]] = [[] for _ in range(len(layout.spanning) + 1)]
    for position in layout.column_order:
        for index in layout.columns[position].sequence:
            bands[bisect_left(span_cys, layout.centre_y[index])].append(index)
    reading: list[int] = []
    for band, members in enumerate(bands):
        reading.extend(members)
        if band < len(layout.spanning):
            reading.append(layout.spanning[band])
    return reading


__all__ = ["ColumnLayout", "PageLayout", "layout_lines", "order_lines"]
