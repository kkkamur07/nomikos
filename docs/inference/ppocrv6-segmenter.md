# PP-OCRv6 medium detection segmenter

## What the model is

Text line detection for manuscript pages with `PaddlePaddle/PP-OCRv6_medium_det`
at Hub revision `8e0f56fb2ef86b461d99cfc7ac5c137738985f61`, served through the
`ppocr-det` runtime adapter in `nomikos_inference/architectures/ppocr_det/` on
ONNX Runtime CPU. The served artifact is the ONNX export
`pp-ocrv6-medium-det.onnx` (SHA-256
`09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`),
whose export and Paddle parity are proved in
[ppocrv6-onnx-parity-2026-09-19.md](ppocrv6-onnx-parity-2026-09-19.md).
The model's Hub page states its licence as Apache-2.0.

## Why it was adopted

Line localisation benchmark (Coptic, 12 pages, 1018 target lines, micro
averaged line localisation, PP box mid axes against annotated polygons, not a
baseline placement score, a development set and not a held out gate):

| Model | P | R | F1 |
|-------|-----|-----|-----|
| BLLA | 61.1 | 84.9 | 71.1 |
| PP-OCRv6 medium cap 960 | 91.8 | 99.1 | 95.3 |
| PP-OCRv6 medium cap 1920 | 90.6 | 99.0 | 94.6 |

Polygon IoU at 0.75: medium 1920 F1 71.6, medium 960 F1 64.1. Greek manual
targets: medium 1920 found 23 of 23 and 13 of 13, medium 960 found 22 of 23
and 11 of 13; the Greek sample is too small for a general claim.

The detector finds nearly every target line where BLLA misses about one in
seven, so it was adopted as a candidate replacement for the default
segmenter, pending the end to end validation below and complete OCR crop
validation.

## How serving works

`run_ppocr_det_segment(image_bytes, model_path, artifact_sha256, params)`
is the entry point production uses. The artifact digest is verified before
the file is opened.

Preprocessing (`preprocessing.py`) reproduces PaddleX 3.7.0 exactly: PIL
decode to RGB, BGR channel order, scale down only when the longer side
exceeds `limit_side_len` with each side rounded to a multiple of 32 (floor
32) via bilinear `cv2.resize`, ImageNet normalisation per stored channel,
CHW float32 batch tensor.

DB postprocess (`postprocessing.py`) mirrors PaddleX 3.7.0 `DBPostProcess`
in quad mode: binarise at `thresh`, `cv2.findContours` over at most
`max_candidates` contours, minimum-area rectangles with a minimum side of 3,
`box_score_fast` gated at `box_thresh`, unclip expansion, a second
minimum-area rectangle with a minimum side of 5, then scaling back to source
coordinates with rounding and clipping. The unclip expansion is PaddleX
3.7.0's own algorithm (OpenCV area and perimeter for the offset distance,
`pyclipper` with `JT_ROUND` for the offsetting), declared in
`[project].dependencies` as `pyclipper>=1.4.0`.

Params, defaults and bounds:

| Param | Default | Bounds | Meaning |
|-------|---------|--------|---------|
| `limit_side_len` | 1920 | integer 320 to 4000 | max side the page is scaled down to |
| `thresh` | 0.2 | 0 to 1 | probability map binarisation threshold |
| `box_thresh` | 0.45 | 0 to 1 | minimum mean probability inside a quad |
| `unclip_ratio` | 1.4 | 0 to 5 | box expansion factor |
| `max_candidates` | 3000 | integer 1 to 10000 | most contours considered |
| `baseline_fraction` | 0.75 | 0 to 1 | where the synthetic baseline sits down the quad |
| `reading_direction` | `ltr` | `ltr` or `rtl` | column order |
| `merge_fragments` | true | boolean | merge row fragments into visual lines |
| `resolve_overlaps` | true | boolean | cut stacked neighbours, drop duplicates |
| `noise_policy` | `flag` | `flag`, `drop` or `off` | what happens to suspect lines |
| `merge_gap_ratio` | 0.25 | 0 to 10 | max merge gap in page median line heights |
| `merge_max_overlap_ratio` | 0.1 | 0 to 2 | max member overlap in median heights inside a merge |
| `merge_max_height_ratio` | 2.0 | 1 to 10 | max member height ratio inside a merge |
| `overlap_cut_threshold` | 0.20 | 0.05 to 1 | shared area fraction that counts as overlap |

Non-finite values, bools where a number belongs and out-of-range values
are rejected with a message naming the param and the bound.

Reading order (`reading_order.py`): quads group transitively into columns by
horizontal overlap, columns run left to right (`ltr`) or right to left
(`rtl`), and inside a column quads group into rows by vertical overlap and
read row by row. A quad spanning two or more columns (a running head or
caption) reads as its own band between the rows above and below it. Narrow
quads (marginal notes, page numbers, initials) attach to the nearest column
without splitting it.

Regions, 2026-10-11: both response builders emit one `SegmentBlock` per
column in `column_order` sequence, so a two-folio spread reaches the
platform as two regions. A region id is `ppocr-det-region-<n>` with `order`
n-1, n counting from 1 in reading order (under `rtl` the right column is
region 1). A region box is the axis-aligned bounds of its column: member
quad bounds on the plain path, refined line bounds on the refined path.
Every line of a column carries its region id; a spanning line carries
`block_external_id` null and the platform merge treats it as blockless. One
column yields one bounding region, and an empty page yields no regions.

Sessions are created with `ORT_ENABLE_EXTENDED`, sequential execution, 1
inter-op thread and 4 intra-op threads: EXTENDED is about 2x faster than
the ALL default with identical boxes, and 4 threads measured 5.85 s on c13
against 11.0 s at 2 threads and 3.6 s at 8
(docs/inference/ppocrv6-onnx-performance-2026-09-20.md). The intra-op
count comes from `NOMIKOS_PPOCR_DET_THREADS` (integer 1 to 64, default 4);
an invalid value fails the run with a clear error.

## Decisions and their reasons

Boxes, not baselines: PP-OCRv6 emits boxes and the segment contract needs a
baseline per line, so the adapter synthesises one across the quad at
`baseline_fraction` 0.75 of the way from the top edge to the bottom edge,
where kraken baselines sit near the bottom of the letter bodies.

Pyclipper instead of shapely: the unclip call was first written with
shapely `buffer` to avoid a native dependency, but the parity measurement
showed up to 3.162 px corner error with a mean near 0.85 px, so it was
replaced by the exact PaddleX algorithm for full parity (see below).

## End to end validation

`scripts/segmentation/ppocr/verify_adapter.py` feeds each of the 14
reference pages as bytes through the production entry point with refinement
switched off (`merge_fragments` false, `resolve_overlaps` false,
`noise_policy` off, so the gate measures the detector and nothing else)
and compares the returned quads with the Paddle pipeline
fixtures (`_ppocr-parity/fixtures/`, cap 1920): counts, then greedy
one-to-one matching by quad centre with corner-SET distances. Gate:
identical counts, no unmatched boxes, max corner distance at most 0.5 px.

| page | fixture | adapter | unmatched | max px | mean px |
|------|---------|---------|-----------|--------|---------|
| vat-1r | 78 | 78 | 0 / 0 | 0.000 | 0.000 |
| vat-1v | 71 | 71 | 0 / 0 | 0.000 | 0.000 |
| vat-2r | 69 | 69 | 0 / 0 | 0.000 | 0.000 |
| vat-2v | 64 | 64 | 0 / 0 | 0.000 | 0.000 |
| vat-3r | 69 | 69 | 0 / 0 | 0.000 | 0.000 |
| vat-7v | 67 | 67 | 0 / 0 | 0.000 | 0.000 |
| c10 | 65 | 65 | 0 / 0 | 0.000 | 0.000 |
| c11 | 131 | 131 | 0 / 0 | 0.000 | 0.000 |
| c12 | 130 | 130 | 0 / 0 | 0.000 | 0.000 |
| c13 | 135 | 135 | 0 / 0 | 0.000 | 0.000 |
| c14 | 138 | 138 | 0 / 0 | 0.000 | 0.000 |
| c21 | 130 | 130 | 0 / 0 | 0.000 | 0.000 |
| grec-p1 | 29 | 29 | 0 / 0 | 0.000 | 0.000 |
| grec-p4 | 57 | 57 | 0 / 0 | 0.000 | 0.000 |

The gate passes on all 14 pages: identical counts, nothing unmatched, and
0.000 px corner distance throughout. The PIL decode was checked against a
cv2 decode and is pixel-identical, so the decode contributes nothing. Per-page overlays (`<page>.overlay.jpg` beside `results.json` in
`_ppocr-parity/adapter-e2e/`) show quads in reading order with synthetic
baselines; no fixture box went unmatched on any page.

Reading order spot check on the overlays: two-column spreads read left page
first, each column top to bottom (c13: 1-33, 34-65, then 66-100,
101-102, 103-135; grec-p4: left page 1-26, right page 27-56). Page headers
(shelfmarks, c13 box 1, grec-p1 boxes 1-2, vat-1r boxes 38-39) read before
their column and footers (copyright bar vat-1r 76-78, Gallica bar grec-p1
29 and grec-p4 57) read last. Enlarged initials stay inside their line
boxes (grec-p1 line 3, whose first line is split into boxes 3 and 4 but
reads left to right). Interlinear insertions get their own number in
vertical position (grec-p4 box 49). Watermarks, page labels and marginal
text are returned as ordinary lines interleaved with body text (vat-1r
diagonal library stamps 1-4 and right-margin fragments 57, 59, 62, 64),
which is expected from the known gaps below.

## Refinement stage

Between detection and the response the adapter refines raw quads into
manuscript lines (`refinement.py`, pure geometry over the page's own
medians and the column bands `reading_order.py` builds, which refinement
shares through `layout_lines`). Order: detect, merge, resolve overlaps,
classify suspects, order, build the response.

Merge: inside one column, quads that the row grouping puts in the same row
are one visual line. Row mates merge along x while the gap is at most
`merge_gap_ratio` (0.25) page median quad heights and the members overlap
by at most `merge_max_overlap_ratio` (0.1) median heights, unless one member
is taller than `merge_max_height_ratio` (2.0) times another. Fragments of one
visual line sit side by side; row mates that overlap belong to the
overlap stage, so the merge stage refuses them. A tall
multi-line initial never merges: it stays its own line with
`source_metadata.role = "initial"`. A merged boundary is the polygon union
of its members plus a bridge over the gap inside their shared y range (a
valid simple polygon, never much taller than one text line, because the
transcription crop is the boundary's bounding box with a polygon mask).
Its baseline runs from the outer left end of the leftmost member's
baseline to the outer right end of the rightmost member's. Score is the
area-weighted mean; `source_metadata.merged_from` records the count.

Overlaps: after merging, pairs sharing at least `overlap_cut_threshold`
(0.20) of the smaller polygon are stacked (cut apart at the mid-baseline
line until they share zero area, unless the cut would remove more than
half of either polygon), duplicates (coinciding baselines with shared area
over the larger polygon above half: only the higher score survives, the
one case that removes a detection by default), containment pairs (shared
over the smaller above the threshold but over the larger at or below half:
both marked `overlap_unresolved` and left) or marked `overlap_unresolved`
and left. A work with role `initial` is never dropped. On the 14
fixture pages the 0.20 default behaves: 194 pre-merge pairs in 20-50%
collapse to zero-area cuts, and 6 lines carry the mark (a vat-2r pair that
still shares above 0.20, plus one pair each on c13 and grec-p4 where a
later cut with a third line removed the shared area after the mark was
set), so zero unmarked pairs at or above 0.20 remain (plus 5 pre-merge
pairs above 50%, all grec-p4 duplicates).

Suspects: a singleton that is not an initial is a suspect when it lies
outside every builder column band (`suspect_reason` `outside_bands`),
unless it sits strictly between two bands inside the text block top to
bottom (a gutter numeral keeps its place and is never a suspect by angle
either); any other non-initial whose bounding-box angle is over 20 degrees
off the page dominant angle is a suspect (`suspect_reason` `angle`).
Measured on the 12 Coptic pages against the benchmark matching: 0 of 1009
real lines flagged with 52 of 106 unmatched detections caught (49%; 71% of
the non-ignore ones; fewer merges than at the old 1.5 gap leave more
singletons to flag). `noise_policy` `flag` (default) orders suspects after
every body line; `drop` removes them; `off` changes nothing. With all three
refinement switches off the adapter output is byte identical to the
unrefined path.

## Refinement evaluation

`scripts/segmentation/ppocr/evaluate_refinement.py` runs the adapter on
the 12 Coptic pages with refinement off, defaults and `drop`, scoring each
against the benchmark target lines with the benchmark's matching (a port
of `axis`, `assign` and `finish` from
`server/ppocrv6-quality_eval_20260919.py`; the port recovers 1002 of 1018
care targets where the benchmark tool recovers 1008, so it is a consistent
off-vs-defaults comparator rather than identical tooling):

| variant | detections | P | R | F1 | alone | pairs | suspects | wrong |
|---------|------------|---|---|----|-------|-------|----------|-------|
| off | 1147 | 0.9065 | 0.9902 | 0.9465 | 1008 | 181 | 0 | 0 |
| defaults | 1136 | 0.9164 | 0.9912 | 0.9523 | 1009 | 2 | 52 | 0 |
| drop | 1084 | 0.9619 | 0.9912 | 0.9763 | 1009 | 2 | 0 | 0 |

Alone means care targets covered by exactly one detection; pairs means
line pairs sharing at least 20% area; wrong means real lines flagged as
suspects. Pass gate, line by line: recall under defaults (0.9912) is ABOVE
refinement off (0.9902), so the gate PASSES with 0 lines lost and 1
fragmentation fixed on c21; precision under drop (0.9619) is strictly
higher than off (0.9065); the 2 remaining pairs are vat-2r lines 32-33
and c14 lines 30-31 with both sides marked unresolved and zero unmarked
pairs left; 0 real lines flagged against the 1% budget; c13 suspects
(129-133 of 133 lines) and vat-1r suspects (lines 70-77 of 77) read after
every body line, while the tall diagonal watermarks (role initial) and
the right-margin strip stay in body order.

Overlap sweep behind the 0.1 default (`--merge-max-overlap-ratio
0.0,0.05,0.1,0.2,0.3` at gap 0.25): recall 0.9902, 0.9902, 0.9912,
0.9902, 0.9902 with 0, 0, 0, 1, 1 lines lost against off and 0, 0, 1, 1,
1 fragmentations fixed. The default is the largest value that loses zero
lines: at 0.2 the c13 overlapping pair (gap minus 0.18 median heights)
merges again and loses its target, while the vat-1r pair (gap minus 0.45)
stays split at every swept value and the c21 fix holds from 0.1 up.

Gap sweep behind the 0.25 default (`--merge-gap-ratio
0.25,0.5,0.75,1.0,1.5`, one run scores defaults and drop at each ratio):
recall 0.9892, 0.9882, 0.9882, 0.9872, 0.9843 with 2, 3, 4, 5, 8 lines lost
against off and 1, 1, 2, 2, 2 fragmentations fixed. No ratio reaches off
recall (a loss already happens at 0.25), so the default is the ratio with
the fewest lost lines. Of the 8 losses at 1.5, six are gap-fixable (gaps
0.47, 0.62, 0.93, 1.08 twice and 1.35 median heights); two of those six
join two ground-truth lines side by side on one row (vat-2v gap 1.35,
vat-3r gap 1.08 killing both members), which is a labelling convention,
not a merge error, and the rule was not changed to chase it. The 2
persistent losses have overlapping members, so no gap ratio can separate
them: each joins a real line to a fragment matching no care target, and
the union axis covers neither target polygon. That was the union-axis stop
condition from the review; the decision taken was the overlap lower bound
above, which keeps both pairs split at the 0.1 default.

Refined overlays (`<page>.refined.overlay.jpg` in
`_ppocr-parity/refinement/`) draw body quads green with order numbers,
baselines yellow and suspects red.

## Publication and registry

The artifact is published at
`hf://nomikos-project/segmentation-ppocrv6-det@stable` (file
`ppocrv6-det.onnx`), pinned in `nomikos_inference/registry.yaml` under
model id `ppocr-segment` by Hub commit
`2274480436df3f1717dcf570eea01cc294294dc6` (README-only follow-up renaming
the card's model id row; the ONNX blob is unchanged) and artifact SHA-256
`09e4c827c5bb20a0344374bbf8b88d41b7c8bf2be3a0db82ffed3bf090eacfe3`.
The download was verified back from the Hub at the pinned revision before
registering.

How to make it selectable: a row in the platform `inference_models`
catalog table is what exposes a model, inserted by the maintainer; the
page editor has no segment-model picker yet, and the segment endpoint
already accepts `model_id`. Nothing is deployed and this model is not the
default segmenter.

## Known gaps

* Column regions are emitted (2026-10-11): one `ppocr-det-region-<n>` block
  per column, spanning lines unblocked. The gap below is closed.
* Refinement merges row fragments that share a row, refusing members
  that overlap by more than 0.1 median heights (those belong to the
  overlap stage). On the benchmark this loses no target line and fixes
  one fragmentation on c21: recall 0.9912 under defaults against 0.9902
  with refinement off; the evaluation table above.
* Tall non-text shapes take `role` `initial` and are exempt from suspect
  flagging, so the vat-1r diagonal watermarks (lines 1-3) and the
  right-margin strip (line 52) stay in body order, as do the c13 top
  shelfmark (inside its column band) and the centred footer bars.
* Small inside-band edge fragments are geometrically indistinguishable
  from narrow real lines and are never flagged; the suspect rule catches
  52 of 106 unmatched detections and no more by design.
* The model is not the default segmenter until complete OCR crops are
  validated.
