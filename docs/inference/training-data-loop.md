# The training-data loop

How a page somebody corrected in the editor becomes a folder a segmenter trainer
can read, what counts as "corrected", and which parts of the trip are still done
by hand.

## The trip a corrected page makes

1. **Segment.** A page is uploaded and segmented. The BLLA adapter
   (`nomikos_inference/architectures/blla/blla_runtime.py`) returns lines with a
   real baseline and a boundary polygon, and **one block covering the whole
   page**. `SegmentMergeService.apply_sync` writes them as `Line` rows with
   `source=kraken`, `manual_geometry=false`.
2. **Correct.** A person opens the page editor and fixes the geometry: drags
   vertices, splits, merges, trims, draws a missing line by hand. Every one of
   those paths sets `manual_geometry=true` on the row it touched. Hand-drawn
   lines get `source=manual`.
3. **Transcribe and approve.** Approved text is a `LineTranscription` row in the
   document's single `kind=ground_truth` `Transcription` layer. There is no
   `approved` column: the existence of that row *is* the approval.
4. **Sign off.** The person ticks the page as reviewed, which sets
   `DocumentPart.reviewed = true`.
5. **Export.** `GET /projects/{p}/documents/{d}/export/page-xml` streams a zip of
   every page as `NNNN.xml` next to `NNNN.webp`, plus a `manifest.json`.
   `?reviewed_only=true` narrows it to signed-off pages.
6. **Train.** Unpack the zip somewhere outside the repo and point a trainer at
   the folder. See below.

## What "corrected" means

    part.reviewed is true
    and the page has at least one line carrying ground-truth text

This is the rule the manifest states and the rule `corrected_page_count` counts.
It is deliberately the conservative reading of what the data model offers.

Rejected alternatives, and why:

| Candidate signal | Why it is not the rule |
| --- | --- |
| `manual_geometry` on any line | A page can have hand-drawn boxes because somebody started and stopped. Editing geometry is not signing off. |
| A paired `PageTranscriptionLine` | Pairing an imported transcription to lines is a step *towards* ground truth, not a signature under it. `SegmentMergeService` treats it as "human touched", which is a different and looser question. |
| `Document.workflow == published` | Publishing is about who may read the page, not about whether it is correct. |
| `reviewed` alone | A page can be ticked reviewed with only model output on it. The export would then ship machine guesses as labels. |

The endpoint is **not** narrowed to this rule. `reviewed_only` is the one filter
three document exports share (PAGE XML, transcription PDF, plain text), and
tightening it here alone would make the three disagree about what a reviewed
chapter is. The archive therefore carries every selected page and the manifest
says, per page, whether it is corrected. A trainer drops the rest with one pass
over `manifest.json` and no second request. **Whether the endpoint should enforce
the rule instead is an open decision for the owner.**

## What the XML now says

`Coords` is the line polygon: the mask, else `points`, else the bounding
rectangle of whatever geometry the row has. `Baseline` is a baseline: an open
polyline with strictly increasing x, inside the polygon.

From 2026-07-09 (commit `891dbc5f`) until 2026-09-18, `Baseline` held the **mask
polygon** whenever a mask existed. Every export made in those two months was
useless for training a segmenter, because kraken, eScriptorium, Transkribus and
Orli all read `Baseline` as an open polyline under the letter bodies. No consumer
ever wanted the polygon there: the editor canvas draws `line.baseline` from the
JSON API, and nothing in the repository reads an exported `<Baseline>` back.

### Derived baselines

Three write paths store something in `Line.baseline` that is not a measurement:

* `LayoutService.create_part_line` stores `{"points": points}`, the polygon the
  box was drawn as.
* `resolve_line_baseline_and_mask`, used by the bulk `PUT /lines` route and by
  PATCH, calls `default_baseline_from_polygon`, which returns a two-point segment
  along the polygon's **bottom edge**.
* Nothing at all, on rows that predate those paths.

All three are replaced at export time by a derived baseline, and the line is
marked `custom="nomikos {...; baseline:derived;}"` so a trainer can tell it from
a measurement. The manifest counts them per page under
`lines.derived_baseline`.

**The fraction is 0.68 of the polygon's height, measured downwards from the top.**
Measured on the 33 recon snapshots of the Coptic ingest (Napoli and Vatican,
`scripts/coptic/pairing-tools-2026-09-18.tar.gz`, `human-guide/v4/recon/`): the
1010 lines carrying both a kraken baseline and a kraken polygon put the baseline
at

| statistic | value |
| --- | --- |
| mean | 0.679 |
| median | 0.680 |
| stdev | 0.061 |
| p5 / p95 | 0.573 / 0.766 |

The two manuscripts agree closely: Napoli 0.690 median over 529 kraken lines,
Vatican 0.674 over 476. That gap of 0.016 is a quarter of the within-manuscript
standard deviation, which is why a single constant is defensible here where pixel
thresholds are not: the calibration work on these same manuscripts found that no
pixel rule transfers across them, but a *ratio* of the line's own height does.

The derived baseline follows the polygon rather than being one straight segment:
at eight sampled x positions the polygon's own top and bottom edges are found and
the point is placed between them, so a line that sags keeps its sag and every
point is inside the polygon by construction. Points that add nothing are dropped,
so a rectangle's baseline comes back out as two points.

Run over all 2831 lines in those snapshots, the exporter keeps all 1010 kraken
baselines and derives for 948 of the 1821 manual lines; all 33 pages pass
`validate_page_xml` with zero problems.

### Baselines that overhang their polygon

kraken's BLLA baselines routinely run past the simplified boundary it ships them
with: 199 of those 1010 lines overhang, by a median of 1px but up to 194px. The
export clips the baseline back to the polygon's bounding box, interpolating y at
the cut so a sloping baseline keeps its slope. The overhang is not information a
trainer can use, because the crop it learns from is the polygon.

### Regions and reading order

One `TextRegion` per `Block`, in `block.order`, with lines in `line.order`
inside. The exception is a page with one block or none, which stays one region:
the segmenter emits exactly one whole-page block, and splitting there would
invent a structure nobody drew. A page with two or more regions also gets a PAGE
`ReadingOrder` element (an `OrderedGroup` of `RegionRefIndexed`), written before
the regions as the 2019-07-15 schema requires. Lines belonging to no block go
into a trailing `region_<part>_unassigned`.

Every `TextRegion` carries a `Coords`, even a region holding no lines: kraken
puts a boundary-less region in `missing_region_ids` and drops it, taking its
lines with it, so a two-column page would read back as no regions at all. The
boundary is the block's box when there is one, otherwise the bounding box of the
lines inside, otherwise the page.

### What kraken 7 will and will not parse

Two rules that a schema validator does not state but kraken enforces, both found
by running `kraken.lib.xml.XMLPage` over a real export:

* **`custom` is Transkribus grammar**, `tag {key:value; key:value;}`.
  `parse_page_custom` splits on `}` and then on `{` and unpacks two values, so a
  bare `source:kraken; kind:polygon` raises `ValueError` and fails the whole
  file, not just that line. Everything this export has to say about a line goes
  under one `nomikos` tag: `custom="nomikos {source:manual; kind:polygon;
  baseline:derived;}"`.
* **Coordinates are integers.** PAGE's `PointsType` is integer pairs and kraken
  calls `int()` on each component, so one decimal fails the file. Coordinates are
  rounded, and clamped into the image, at the single place points are formatted.
  Rounding can break the invariants that make a baseline a baseline, so it is
  followed by: vertices that collapse onto one x are dropped, a polygon left with
  fewer than three distinct vertices is squared off to its bounding rectangle,
  and the baseline is pulled back inside the polygon's box.

The tags kraken gives meaning to are deliberately **not** written. `structure
{type:...;}` sets the class a segmenter trains per line or region, and there is
no line typology in this data model to put there; `readingOrder {index:N;}`
duplicates the `ReadingOrder` element, and kraken 7.1.1's line-level branch is
unreachable anyway (it tests `'index' not in reg_cus['readingOrder']` against a
list of dicts, which is always true). With no `structure` tag, `_get_type`
returns its default, so **`ketos segtrain` sees every line as class `default`**
and each region as its `type` attribute, `paragraph`.

## Checking an export

Without kraken, in the test suite or a script:

```python
from backend.annotation.application.page_xml_validation import validate_page_xml

result = validate_page_xml(open("0001.xml", "rb").read())
assert result.problems == []
```

It checks the shape of the file, never the manuscript. Lines with an empty
`Unicode` are counted in `result.lines_without_text` rather than faulted: a page
can be segmented and only half transcribed, and that export still trains a
segmenter.

With kraken 7, in an environment that has it (the repo deliberately does not
depend on it):

```
python scripts/platform/check_page_xml_with_kraken.py path/to/unpacked-export/
```

It loads each file with `kraken.lib.xml.XMLPage`, prints line and region counts,
and names any line kraken reads without a baseline or a boundary. Run on kraken
7.1.1 over the 33 recon pages, all 33 load with no findings and kraken counts
back exactly what went in: 2831 lines and 33 regions.

## Training

Neither command below has been run from this repository. They are recorded so
that the shape of the hand-off is written down, not because anything here
automates them.

**kraken segmentation**, which is what this export exists for:

```
ketos segtrain -f page -o models/segmenter --resize both unpacked-export/*.xml
```

`-f page` is the PAGE XML reader; it takes the XML files and resolves each
`imageFilename` relative to the XML, which is why the archive ships the image
next to it under a matching stem.

**kraken recognition**, from the same files:

```
ketos train -f page -o models/recognizer unpacked-export/*.xml
```

**Orli** (`https://github.com/mittagessen/orli`), from the same author as kraken.
It detects baselines *and* emits them in reading order, so it is the one consumer
here that cares about our `ReadingOrder` element. Its trainer does not read PAGE
XML directly: per its README, "Orli trains on Arrow datasets compiled from
PageXML or ALTO files, storing each line as a normalized baseline polyline in
source-file order", and "compilation respects the implicit reading order from
source file line sequences" - which is why the order lines are written in matters
as much as the geometry.

```
orli compile -o dataset.arrow --allow-textless -r 1920 1440 unpacked-export/*.xml
orli --config finetune.yaml train --load "$MODEL"
```

Two things to know before trying it: the base model "*only* works in bfloat16
precision", and a fine-tuned `best_*.safetensors` plugs back into kraken 7 the
same way the base model does
(`kraken ... -x segment -bl --model orli_base.safetensors`). The base model is
fetched with `kraken get 10.5281/zenodo.20558179`.

## What is still manual

* Unpacking the archive and putting it somewhere outside the repo. There is no
  "send to training" button and no job that trains.
* Filtering to corrected pages. The manifest says which they are; nothing acts on
  it.
* Deciding the train/validation split. Note
  that the Syriac splitter in `src/` is unshuffled, and `src/` is not ours to
  fix.
* Everything after `ketos`. No model produced this way is registered, published
  or bound to a project by any of this.
* Re-exporting after a correction. The export is a pull, not a subscription; a
  page corrected after the zip was made is not in it.

## Traps

* A re-segment deletes approved text on lines it replaces: `manual_geometry`
  tracks geometry edits only, so it does not protect the transcription. Export before re-segmenting, not after.
* `Line.order` has no unique constraint, so two lines on a page can share an
  order (#114). Reading order in the XML is then whatever `(order, created_at)`
  decides.
* A page whose `DocumentPart.width`/`height` are null has them read from the
  image header at export time. Cheap, but it means the bundle route reads the
  full-resolution object for a page the bare XML route would not.
