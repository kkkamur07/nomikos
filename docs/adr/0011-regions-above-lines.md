# 0011. Regions are the unit of layout above lines

- Status: Accepted
- Date: 2026-10-11

## Context

Many of our photographs show two folios side by side (a spread such as 4v and 5r), and
some pages carry two columns. The editor kept a page's lines in one flat list, so a
researcher could not say "these lines are folio 4v, those are 5r", and every export
flattened both folios into one page of text.

The schema already had the unit for this: a `blocks` table (box, order, manual flag) and a
nullable `lines.block_id`, with create, patch and delete routes. Nothing used it. The
segmenters wrapped every line in one full-page block or emitted none, the editor could not
draw or assign one, and reading order (ADR 0008) and the exports ignored `block_id`.

## Decision

A block is a **region**: the layout unit above lines, typically one folio or one column.
It is used, not replaced.

- **Label.** `blocks.label` (nullable, up to 64 characters, migration `008_block_label`)
  holds the researcher's name for the region, typically a folio ("4v"). Create and patch
  accept it; an explicit `null` or empty string clears it.
- **Drawing assigns.** `POST .../blocks` takes `assign_contained` (default true): every
  line of the page whose polygon centroid lies inside the box and that is in no region
  yet joins the new one. A region never takes lines from another region; moving a line
  is an explicit `PATCH .../lines/{id}` with `block_id`.
- **Reading order is region-first.** Lines are numbered region by region: all lines of
  the block with the lowest `(order, created_at)`, then the next, then the lines in no
  region. Inside a region the ADR 0008 rule applies unchanged (`insertion_index` compares
  only that region's lines, and "at the end" means the end of the region). Block create,
  block order change, block delete and a line's `block_id` change renumber the page under
  the part lock; a line moved to another region is placed in it by position. Segment runs
  (`apply_sync`) number the segmenter's and the kept lines the same way. Responses are
  unchanged: the editor reads the shifted orders back with its existing line reload.
- **Exports group by region** when a page has at least one block: regions in order, a
  labelled region's lines preceded by the label (its own line in the text export, a bold
  paragraph in docx, a small heading over the region's box in the document PDF), then the
  lines in no region. A page with no blocks exports exactly as before.
- **Any source.** Regions from a segmenter run, from the editor's Region tool or from a
  hand-written row behave the same; nothing checks where a block came from.

## Consequences

- A spread can be split into its folios with two rectangles, and the transcript and the
  exports then say which lines are which folio.
- Deleting a region keeps its lines where they are on the page; they become unblocked
  and are numbered after the remaining regions.
- A region is kept across re-segmentation only if a human drew or edited it
  (`manual_geometry`), but the lines in it are not: a re-segment replaces model lines,
  and the fresh lines join the segmenter's own regions, not the kept one. Re-drawing the
  region (or a region-aware segmenter) reassigns them.
- The per-page PDF and the PAGE XML export still write one region per page; they do not
  print labels yet.
- Production applies `scripts/platform/sql/2026-10-11_block_label.sql` before deploying
  the API that reads `blocks.label`.
