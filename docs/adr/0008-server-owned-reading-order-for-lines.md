# 0008. Server-owned reading order for lines

- Status: Accepted
- Date: 2026-10-11

## Context

`Line.order` is a plain integer with no unique constraint on `(part_id, order)`. Three writers
set it, and none of them kept it meaningful:

- **Create.** The editor sent `max(order) + 1` and the server stored it verbatim. A line drawn
  between lines 3 and 4 of a page was numbered 24 on a 23-line page, on every surface: the
  segment nav (`24 / 24`), the selected-segment header, the public transcript panel
  (`Line 24`) and the export filename.
- **Delete.** Nothing renumbered, so a page carried a gap for every deleted line. The editor
  hid it with a dense rank; the public panel and the segment export showed `order + 1` and
  so did not agree with the editor.
- **Re-segmentation.** `SegmentMergeService.apply_sync` keeps lines with manual geometry or
  approved text at their old orders and inserts the segmenter's lines numbered from 0. Kept
  and fresh lines collided and interleaved.

A pure sort by baseline height would fix none of this on our pages: many are two-folio
spreads side by side, and some carry two columns, so height alone interleaves the folios.

## Decision

The server owns reading order, and it is dense (`0..n-1`) after every write. The logic is
pure, in `backend/document/application/reading_order.py`:

- `insertion_index(existing, new)`: the candidates are the existing lines whose x-range
  overlaps the new line's by at least 30 % of the narrower one (the same column or folio).
  The new line goes after the nearest candidate above its centre, else before the nearest
  candidate below it, else at the end. Height is the baseline's mean y, or the polygon's
  middle row when there is no baseline. A line beside another one (a margin note level
  with line 2) goes right after that one instead: always when no line shares its column,
  and otherwise when it is no further off sideways than that line is tall and nearer than
  the column's nearest line is vertically (so a line low on a short verso stays on it).
- `renumber(lines)`: `0..n-1` in the given sequence.
- `merge_reading_order(kept, model)`: the segmenter's sequence, each kept line inserted with
  `insertion_index`.

Wiring, all under the existing `lock_part` row lock:

- `POST .../lines` takes `order` as optional. Omitted, the line is placed with
  `insertion_index`. Sent (undo and redo restore a line to its old place), it is the
  position: before the first line whose order is at least that. Either way the lines after
  it move down one.
- `DELETE .../lines/{id}` renumbers the rest, closing the gap.
- `PUT .../lines` takes the client's orders as the sequence and renumbers it densely.
- `apply_sync` numbers kept and fresh lines with `merge_reading_order`.
- The segment export numbers by position in list order, not `order + 1`; so does the public
  transcript panel. The editor keeps its dense rank, and reads the shifted orders back after a
  create or delete because the endpoint returns only the row it wrote.

Existing pages are fixed once with `scripts/platform/renumber_lines.py` (dry run by default,
`--apply` writes), which closes gaps and breaks ties without moving any line within its
region; it ranks the part's blocks as the API does, so a page not yet region-first is
regrouped now rather than on its next write.

Revised 2026-10-11 after the backend critique. The beside rule only applies to a new line
that is a margin note: when a line of its own column lies within one line height of it (the
larger of its height and that line's), the column decides, and a beside neighbour wins only
when it is no further off sideways than the new line itself is tall. So a main-column line
next to a tall gloss, or in a column 15 px from the next one, stays in its column. On a
re-segment, a kept line whose block was deleted first joins the fresh region that contains
its centroid and is then placed inside that region, so approved lines are no longer moved
to the end of the page.

## Consequences

- One line has one number everywhere, and adding or removing a line shifts the numbers after
  it the way a reader expects.
- Placement is a heuristic. A line that overlaps no column and sits level with no line goes
  last; a line that straddles two columns attaches to whichever it overlaps enough. Both
  are visible and fixable by redrawing. (Before the row rule, a margin note beside line 2 of
  a 51-line page went last, as line 52.)
- A create or delete now writes every later line of the page, a few hundred rows at most.
- A line drawn under the old rule stays last on its page after the renumber script; it has to
  be redrawn to move. No migration: `order` stays an int, and nothing enforces density in the
  schema, so a writer that bypasses `LayoutService` (a script, segment health) is responsible
  for keeping it dense itself, as segment health already does.
