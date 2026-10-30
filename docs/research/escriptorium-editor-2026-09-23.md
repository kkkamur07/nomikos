# How the eScriptorium editor binds segments to text

## Summary

The transcription UI is not one editor but three panels over one Vuex store: a paper.js canvas segmentation panel, an SVG visual transcription panel, and a contenteditable diplomatic panel, plus a modal for edit-in-place.
The hypothesis holds, with a correction: text is bound to the baseline path (not the mask polygon) via SVG `<text>` on `<textPath>`; the mask is drawn underneath only as a polygon backdrop and hover overlay.
Every line is identified by its backend integer `pk`, carried through the REST JSON, the `lines` store, and `data-line-pk` DOM attributes, so selection and editing stay joined across panels.
Font size is derived per line from mask area divided by baseline length (`area / pathLength * 0.3 * fontSizeRatio`); diagonal lines work for free because the glyphs follow the path.
For Nomikos, the portable ideas are the textPath binding, the pk-keyed store with `currentTrans` pointer, and the geometry-driven ordering; the paper.js segmenter itself is the expensive part to copy.

## Repository layout and stack

Clone: `https://gitlab.com/scripta/escriptorium.git` at commit `fdb75db3ae7abf2f5d64981a71136af28c7d94c3` (shallow, 2026-09-23).

Backend is Django (see `app/apps/core/models.py`, `app/apps/core/merger.py`, `app/apps/api/serializers.py`, `app/apps/api/views.py`). Frontend is Vue 2 with Vuex 3 and webpack, per `front/package.json` (`"vue": "^2.7.16"`, `"vuex": "^3.6.0"`, `"paper": "^0.12.17"`, plus `@recogito/recogito-js` for text annotations and `sortablejs` for drag reorder). There are two frontend generations in the tree: the current editor lives in `front/src/editor/` (store, API, mixins, `front/src/baseline.editor.js`) with components in `front/vue/components/`; the older `front/src/` scripts are legacy.

Key editor files and their roles:

- `front/src/baseline.editor.js` (3435 lines): the `Segmenter` class (line 654) and `SegmenterLine` (line 251), built on paper.js (`paper.setup(this.canvas)`, lines 1266 to 1267). Owns all canvas interaction: drawing, selecting, splitting (`splitLinesByPath`, line 2853), merging, reversing (line 516), masks.
- `front/vue/components/SegPanel.vue` (1389 lines): segmentation panel shell. Instantiates the `Segmenter` (import line 377) and bridges it to the Vuex store in both directions.
- `front/vue/components/VisuPanel.vue` (275 lines) and `front/vue/components/VisuLine.vue` (224 lines): the visual (coordinate-bound) transcription panel. Pure SVG, no canvas, no contenteditable.
- `front/vue/components/DiploPanel.vue` (1351 lines) and `front/vue/components/DiploLine.vue` (72 lines): the diplomatic panel, a single `contenteditable` div whose children are plain divs keyed by line pk, with Recogito (`@recogito/recogito-js`, import line 277) layered for text annotations.
- `front/vue/components/TranscriptionModal.vue`: edit-in-place modal opened by clicking a line in the visual panel. Shows a deskewed line-image crop plus a text input.
- `front/vue/components/OverviewPanel/OverviewPanel.vue`: Elements panel listing regions and lines; clicking a row commits `lines/setSelected` or `regions/setSelected` (lines 253 to 258).
- `front/vue/components/Editor.vue` (336 lines): shell. Fetches the part, wires the selected transcription, mounts one global `TranscriptionModal` when `state.lines.editedLine` is set (lines 51 to 54), and listens for background mask recalculation events (`part:mask`, lines 204 to 212 and 320 to 329).
- `front/vue/components/GroupedLine/GroupedLine.vue` (183 lines): line rows for the region-grouped reorder mode.

## Data model (backend + JSON + store)

Backend models (`app/apps/core/models.py`):

- `Block` (line 1896): a region. Fields include `box` (JSON polygon, line 1903), `typology` FK to `BlockType`, `document_part` FK, `order` (django-ordered-model, ordered within the part), `external_id`, `locked`.
- `Line` (line 1984): a segment. `mask` is a closed JSON polygon or null (lines 1991 to 1993); `baseline` is an open JSON polyline of at least 2 points or null (lines 1995 to 1997); `document_part` FK; `block` FK nullable with `on_delete=SET_NULL` (lines 2001 to 2003); `typology` FK to `LineType`; `order` within the part; `external_id`. The derived `box` property (lines 2033 to 2049) is computed from the mask if present, else from the baseline. Validation requires at least one of baseline and mask (`clean`, lines 2054 to 2057).
- `Transcription` (line 2069): a named layer per document (`unique_together = ["name", "document"]`, line 2084), with a protected `manual` default.
- `LineTranscription` (line 2104): one line's text in one layer. `transcription` FK, `content` (plain string up to 2048 chars, line 2112), `graphs` (optional per-character polygons plus confidence, lines 2114 to 2145), `avg_confidence`, and `line` FK nullable with `on_delete=CASCADE` (lines 2151 to 2153; the comment "nullable in case we re-segment ?? for now we lose data" is theirs). Uniqueness is per `(line, transcription)` (line 2158). Content is versioned (`Versioned` mixin) and HTML-sanitized to `em, strong, s, u` tags on write (`LineTranscriptionSerializer.cleanup`, `app/apps/api/serializers.py`, lines 766 to 770).

REST JSON (`app/apps/api/serializers.py`): `LineSerializer` (line 793) exposes `pk, document_part, external_id, order, region (source block), baseline, mask, typology`; `DetailedLineSerializer` (line 854) appends `transcriptions` (all layers for the line). Line order moves go through `LineOrderSerializer` (line 839, `pk, order`). The part payload delivers `data.lines` and `data.regions` arrays which the store splits apart (`front/src/editor/store/parts.js`, lines 104 to 125).

Vuex store (`front/src/editor/store/`): `lines.js` holds `all` (each entry carries `pk, baseline, mask, region, order, type, transcriptions` dict keyed by layer pk, plus a resolved `currentTrans` pointer for the selected layer), `editedLine` (the line open in the modal), `selectedPk`/`selectedToken`, and `autoOrder`. `regions.js` mirrors selection for blocks. `transcriptions.js` holds the layer list, the selected layer pk, and the fetch/update actions. `document.js` holds direction settings (`defaultTextDirection`, `mainTextDirection`, `readDirection`), confidence display state, and panel config. `parts.js` holds the image and part pk. The `currentTrans` pointer is the single join point: every panel renders `line.currentTrans.content` and never looks up layers itself (`lines.js` mutation `updateCurrentTrans`, lines 88 to 96).

## The panels

### SegPanel (segmentation, left): paper.js canvas

Renders the page image on a paper.js canvas with baselines as paper `Path`s and masks as paper paths (`SegmenterLine` constructor, `baseline.editor.js`, lines 251 to 320). Supports line, mask, and region modes, line numbering, direction hints, and an undo manager. All geometry edits (draw, drag points, split by stroke via `splitLinesByPath` at line 2853, cut, merge, reverse at line 516, link or unlink lines to regions) happen here first as canvas operations, then persist through the bulk line APIs. Selection changes propagate outward: the panel watches `state.lines.selectedToken` and `state.regions.selectedToken` and selects the matching canvas element (`SegPanel.vue`, lines 564 to 582; `selectElement`, lines 1262 to 1268).

### VisuPanel and VisuLine (visual transcription, right): SVG text on baseline paths

`VisuPanel.vue` renders one `<svg>` sized to the image aspect (`updateView`, lines 232 to 238) containing one `<VisuLine>` per line, keyed `'VL' + line.pk` (lines 135 to 147). Each `VisuLine` (template lines 1 to 50) draws a `<g>` with three children: a `<polygon>` for the mask, a `<path>` for the baseline, and a `<text>` whose single `<textPath :href="'#' + textPathId">` carries `line.currentTrans.content` (lines 27 to 32), where `textPathId` is `"textPath" + line.pk` (lines 74 to 76). There is no rotation transform anywhere in this component; diagonal lines render diagonally because the glyphs follow the baseline path. Clicking the group calls `edit()` (lines 209 to 213), which dispatches `lines/toggleLineEdition` unless the pan tool is active. Hover shows the mask in the other panels through the shared `.panel-overlay` mechanism (`LineBase.showOverlay`, `mixins.js`, lines 75 to 89).

### DiploPanel and DiploLine (diplomatic text, plain list): one contenteditable

Renders one `DiploLine` component per line (key `'DL' + line.pk`) plus a single `contenteditable` div `#diplomatic-lines` (line 212) that DiploLine instances populate imperatively: on mount each calls `this.$parent.appendLine(null, this.line.pk)`, creating `<div data-line-pk="...">` children (DiploPanel lines 729 to 757; DiploLine lines 53 to 58). Text sync is DOM-first: `setElContent` writes `textContent` (DiploLine lines 60 to 63), and panel-level `changed`/`save` handlers diff the divs back into `updatedLines`/`createdLines` and persist via `transcriptions/bulkUpdate` and `transcriptions/bulkCreate` (DiploPanel lines 724, 1156 to 1173). Reordering uses SortableJS drag and drop (`onDraggingEnd`, line 829; `moveLines`, line 861) or the region-grouped `GroupedLine` rows. Arrow keys move between lines (`onKeyPress`, lines 941 to 962; `focusNextLine`/`focusPreviousLine`). Recogito annotations overlay the same divs (line 612).

### TranscriptionModal (edit-in-place): deskewed crop plus input

Opened globally by `state.lines.editedLine` (Editor.vue lines 51 to 54). Shows a rotated, translated crop of the line image with the mask polygon overlaid, plus a text input bound with `v-model.lazy="localTranscription"` (line 188) whose setter dispatches `transcriptions/updateLineTranscriptionVersion` (lines 506 to 528). Prev/next buttons and arrow/enter shortcuts call `editLine(direction)`, which saves then moves `editedLine` to the adjacent line in store order (`lines.js` `editLine`, lines 455 to 467; modal lines 601 to 605). RTL-aware: prev/next buttons swap with `readDirection` (lines 90 and 99).

### OverviewPanel (Elements list) and Editor.vue shell

OverviewPanel lists regions and lines with filter and ordering display; `onSelectElement` commits `lines/setSelected` or `regions/setSelected` by pk, and `onEditLine` dispatches `lines/toggleLineEdition` (lines 252 to 259). Editor.vue bootstraps everything: it fetches the part, restores the last used transcription layer, mounts the modal on `editedLine`, and applies background `part:mask` websocket events to the store via `lines/update` (lines 170 to 186, 320 to 329).

## The segment-to-text binding

The binding is baseline-anchored SVG text, re-computable from three stored values per line (`baseline`, `mask`, `currentTrans.content`) plus one display ratio. Concretely, for each line with backend pk `P`:

1. Path construction. The baseline JSON array (open polyline in image pixels) becomes an SVG path scaled by `ratio = panelWidth / imageWidth` (`BasePanel.setRatio`, `mixins.js`, lines 59 to 63):

```
baselinePoints() {
    ...
    return "M "+baseline.map((pt) => ptToStr(pt)).join(" L ");
}
```

with `ptToStr(pt) = Math.round(pt[0]*ratio)+" "+Math.round(pt[1]*ratio)` (VisuLine.vue, lines 120 to 131). If `baseline` is null, a fake two-point baseline is built from the mask's leftmost and rightmost points (`fakeBaseline`, lines 107 to 112). The path gets `id="textPathP"` (`textPathId`, lines 74 to 76).

2. Text attachment. The transcription string is placed in `<text><textPath href="#textPathP">` (template lines 20 to 33). No `x`, `y`, `rotate`, or `transform` is used: placement and angle come entirely from the path geometry, so diagonal and curved lines place text correctly with zero extra code. RTL is handled by `text-anchor="end"` when the document default direction is rtl (line 23) and by choosing the path direction; vertical (`ttb`) scripts get a second `<text>` branch with `writing-mode="tb"` (lines 35 to 49).

3. Mask backdrop. The mask polygon is drawn under the text as `<polygon :points="maskPoints">` with the same ratio scaling (lines 7 to 11, 103 to 106). Its fill doubles as the confidence heatmap: when confidence display is on, fill becomes `hsl(hue, 100%, 50%, 50%)` with `hue = pow(avgConfidence, confidenceScale) * 120`, i.e. red to green (lines 94 to 102). Stroke is `none` once the line has text, else lightgrey; the baseline path stroke is `none` once text exists, else blue (lines 77 to 83, 113 to 119).

4. Font sizing formula. `computeLineHeight` (lines 145 to 183) estimates the strip height as mask area over baseline length, then takes 30 percent of it times the user font ratio:

```
area = |shoelace(mask)| * ratio        (lines 148 to 172)
lineHeight = area / pathElement.getTotalLength()   (line 173)
lineHeight = max(round(lineHeight), 5) * 0.3       (line 179)
textElement.font-size = lineHeight * fontSizeRatio  (lines 181 to 182)
```

`fontSizeRatio` defaults to 0.25 per document and persists in the user profile, adjustable with plus/minus buttons (VisuPanel lines 190, 239 to 248). Without a mask the fallback is 30 (line 176). Then `computeTextLength` (lines 184 to 195) forces the rendered text to exactly fill the baseline by setting `textLength = pathLength` with `lengthAdjust="spacingAndGlyphs"` (template line 24), so short text stretches and long text compresses along the line.

5. Cross-panel identity. The join key everywhere is the backend `pk`: store objects (`lines.js`), component keys (`'VL' + pk`, `'DL' + pk`, `textPath + pk`), DOM attributes (`data-line-pk` in the diplomatic panel, `data-line-pk`/`data-order` in GroupedLine), and canvas contexts (`Segmenter` option `idField: "pk"`, SegPanel line ~607). Hover in any panel highlights the mask in the others via `showOverlay`/`hideOverlay` copying `maskPoints` into `.panel-overlay` polygons (`mixins.js`, lines 75 to 96). Selection from the Elements list flows store to canvas through the `selectedToken` watcher (SegPanel lines 572 to 579). Editing flows the other way: click in VisuLine or DiploPanel sets `editedLine` via `toggleLineEdition` (`lines.js` lines 451 to 453), which opens the modal.

6. What is NOT coordinate-bound. The mask never shapes or clips the text (no clipPath, no per-character `graphs` rendering in the editor; `graphs` polygons exist in the model only for confidence display). Rotation is never computed in the visual panel; the only angle math in the frontend is `getLineAngle` in TranscriptionModal (`atan2(p2 - p1)` on baseline endpoints) used to deskew the modal's image crop with `translate(...) rotate(angle)` (TranscriptionModal `getLineAngle`, `getRotatedLineBBox`, `computeImgStyles`). So the orchestrator hypothesis is confirmed for the baseline and refuted for the mask: text follows the baseline path, the mask is backdrop plus sizing input.

## Editing lifecycle

Create: canvas strokes become lines via `lines/bulkCreate` (lines.js lines 164 to 237), posting `{pk, document_part, baseline, region, typology}` to `bulk_create` (views.py line 1489), which re-creates lines plus their transcriptions through `_bulk_create_helper` (lines 1495 to 1536). Empty `currentTrans` shells are attached client side.

Move and resize: dragging points or whole baselines persists via `lines/bulkUpdate` (lines 239 to 315) to `bulk_update` (views.py line 1538). Changed baselines trigger async server-side mask recalculation (`recalculateMasks`, lines 404 to 426, hitting `reset_masks`, views.py line 1184); results return over websocket/`part:mask` and commit through `lines/update` (Editor.vue lines 320 to 329). Region reassignment sets `hasToRecalculateOrdering` and re-runs ordering when `autoOrder` is on (lines 289 to 312).

Delete: `lines/bulkDelete` (lines 317 to 340) to `bulk_delete` (views.py line 1547), which returns the deleted JSON (used for undo) and deletes rows; `LineTranscription.line` is CASCADE so its texts vanish with it. The diplomatic panel marks divs beyond the line count as danger (`constrainLineNumber`, lines 759 to 806).

Split: canvas-side only in `splitLinesByPath` (baseline.editor.js line 2853), which cuts baseline and mask paper paths with projected normals; each resulting piece is then created as a new line (new pk, empty transcription) while the original is deleted. There is no transcription carryover at split time; the text must be retyped or re-recognized.

Merge (join): `lines/merge` (lines 342 to 392) to the `merge` endpoint (views.py lines 1558 to 1587), implemented by `merge_lines` (merger.py line 115): baselines are ordered by brute-force shortest Hamiltonian path over endpoint distances (`find_order`, `build_dist_matrix`, lines 24 to 57, capped at `MAX_MERGE_SIZE = 8`, line 13), concatenated end to end (`merge_baseline`, lines 59 to 62), texts joined with the script blank character or a space (`merge_transcriptions`, lines 71 to 103, RTL-aware by reversing line order for RTL scripts), region re-detected from the merged baseline midpoint (`find_block`, lines 106 to 111), then the old lines are deleted and one new line plus merged transcriptions created.

Order: source of truth is the integer `order` field per part. Automatic order comes from `DocumentPart.recalculate_ordering` (models.py line 1195), a geometry algorithm: line origins (baseline start, or nearest mask point to the page origin corner; RTL mirrors the origin and uses the baseline end), DBSCAN column clustering on x origins, per-column y sorting. It runs after create, region change, delete, and merge when `autoOrder` is true, and on explicit request (`recalculate_ordering` endpoint, views.py line 1195). Manual order moves post `{pk, order}` pairs to the `move` endpoint (views.py line 1590) and commit `updateOrder` (lines.js lines 74 to 87). `enforce_line_order` (models.py line 1854) compacts order to a dense 0-based sequence. Keyboard prev/next and the modal step through `state.all` order (lines.js lines 455 to 467).

Versioning: every transcription save snapshots the previous content (`LineTranscriptionViewSet.update` calls `new_version`, views.py lines 1620 to 1630); the modal writes through `updateLineTranscriptionVersion` (transcriptions.js lines 143 to 180), which optimistically commits `updateTranscriptionVersion` then PUTs or POSTs. Orphan policy: deleting a line deletes its transcriptions (CASCADE); the model comment admits resegmentation loses data (models.py line 2150).

## Comparison with the Nomikos page editor

`PageEditorCanvas.tsx` (1063 lines) and `PageEditorTranscriptionStrip.tsx` (290 lines) implement a split guidance editor rather than a coordinate-bound one. The canvas overlays mask polygons and baseline polylines on the image in reading order (`orderedLines`, line 188; polygon/baseline rendering, lines 298 to 384), keyed by string `line.id` with `selectedSegmentId` driving highlight (lines 299 to 350), which is the same pk-identity idea as eScriptorium but with local selection state instead of a global store token. The transcription strip edits exactly one selected segment at a time (ground-truth textarea, model-output review, approve and promote actions, segment prev/next navigation) instead of eScriptorium's three parallel panels (visual overlay plus full-page contenteditable plus modal). There is no textPath or rotated overlay: text never sits on the image at line position, and no font size is derived from geometry. Ordering comes from precomputed `segmentsInNumberOrder`, not from a geometry recompute after each edit. Layer handling is similar in spirit (ground-truth vs model layers resemble Transcription layers, with per-line lookup helpers in `hooks/utils`), but eScriptorium resolves the active layer once into `currentTrans` while Nomikos resolves per render. Net: Nomikos has the identity key and the single-line edit loop; it lacks the geometry-placed text, the full-page diplomatic editing surface, and the auto-reordering and mask-recalculation feedback loops.

## What to reuse

1. SVG textPath binding for the overlay (VisuLine pattern). Render each selected or all lines as `<path id>` from the baseline plus `<text><textPath>` with the transcription, scaled by `panelWidth / imageWidth`. Effort: 1 to 2 days for a read-only overlay; 1 week with hover, confidence tint, and RTL branches.
2. The pk plus currentTrans store shape. Keep one canonical line object keyed by backend id carrying `baseline, mask, region, order`, with a resolved pointer to the active layer's text, so canvas, overlay, and strip can never disagree. Effort: 2 to 3 days refactor of line state.
3. The font-size formula (`maskArea / baselineLength * 0.3 * userRatio`, min clamp, `textLength` fit). It makes overlay text legible on tall lines and compact on small print without manual tuning. Effort: half a day.
4. Geometry-driven ordering after edits (origin corner plus column clustering, dense recompaction). Adopt the trigger points (after create, region change, delete, merge) even if the algorithm starts simpler (sort by baseline-start y, then x). Effort: 1 day for simple sort; 1 week for the DBSCAN column version.
5. Mask recalculation as an async job with store patch-back (`part:mask` pattern). Geometry edits stay instant on canvas while the server refines masks and pushes updates keyed by pk. Effort: 3 to 5 days including the job and socket plumbing.
6. The modal deskew crop (`getLineAngle` plus rotate/translate crop). High value for transcribers on skewed or diagonal lines and cheap to build on existing segment geometry. Effort: 2 to 3 days.
7. Do not copy the paper.js segmenter wholesale. It is 3400 lines of canvas interaction (draw, split, cut, undo) tightly coupled to Vue 2. For Nomikos, keep the current canvas and port only the split-by-stroke and merge endpoints with their transcription join rules. Effort saved by not porting: several weeks.

## Sources

eScriptorium commit `fdb75db3ae7abf2f5d64981a71136af28c7d94c3`, files under `/var/folders/h3/pvdfbwyx1txcx5ftyxsw_chr0000gn/T/herdr-orchestrator/escriptorium-editor/escriptorium-src/`:

- `front/package.json`: Vue 2.7, Vuex 3, paper.js 0.12, recogito-js, sortablejs.
- `front/src/baseline.editor.js`: `SegmenterLine` (251), `reverse` (516), `Segmenter` (654), paper setup (1266 to 1267), `splitLinesByPath` (2853).
- `front/src/editor/mixins.js`: `BasePanel` ratio (59 to 67), `LineBase` overlay and maskPoints (75 to 110).
- `front/src/editor/store/lines.js`: selection and edition (28 to 31, 451 to 467), ordering (74 to 87), create/update/delete/merge/move/masks/ordering actions (164 to 449).
- `front/src/editor/store/parts.js`: part fetch, lines and regions commit (104 to 125).
- `front/src/editor/store/transcriptions.js`: layer state, `fetchContent` (95), `updateLineTranscriptionVersion` (143 to 180).
- `front/src/editor/store/regions.js`: region selection (6 to 20).
- `front/vue/components/SegPanel.vue`: Segmenter import (377), selection watchers (564 to 582), merge (1136 to 1150), `selectElement` (1262 to 1268).
- `front/vue/components/VisuPanel.vue`: svg container (135 to 147), font ratio (190), sizing (232 to 248).
- `front/vue/components/VisuLine.vue`: full binding (1 to 224), esp. template (1 to 50), `textPathId` (74 to 76), colors (77 to 102), points (103 to 131), sizing (145 to 195), `edit` (209 to 213).
- `front/vue/components/DiploPanel.vue`: contenteditable (212), Recogito (277, 612), `changed`/`appendLine` (724 to 757), reorder (829 to 878), keys (941 to 962), persistence (1156 to 1173).
- `front/vue/components/DiploLine.vue`: mount, `getEl`, `setElContent`, order watcher (whole file, 72 lines).
- `front/vue/components/TranscriptionModal.vue`: input binding (188, 506 to 528), shortcuts (193 to 195, 222), `editLine` (601 to 605), angle and deskew (getLineAngle, getRotatedLineBBox, computeImgStyles, computeInputStyles).
- `front/vue/components/OverviewPanel/OverviewPanel.vue`: select and edit dispatch (252 to 259).
- `front/vue/components/GroupedLine/GroupedLine.vue`: region-grouped reorder rows (whole file).
- `front/vue/components/Editor.vue`: shell wiring, modal mount, mask events (51 to 54, 170 to 186, 204 to 212, 320 to 329).
- `app/apps/core/models.py`: `Block` (1896), `Line` (1984), `Transcription` (2069), `LineTranscription` (2104), `recalculate_ordering` (1195), `enforce_line_order` (1854).
- `app/apps/core/merger.py`: cap (13), distance matrix and order (24 to 57), baseline join (59 to 62), text join (71 to 103), region find (106 to 111), `merge_lines` (115 to 144).
- `app/apps/api/serializers.py`: `LineSerializer` (793), `LineOrderSerializer` (839), `DetailedLineSerializer` (854), transcription cleanup (766 to 770).
- `app/apps/api/views.py`: mask and ordering endpoints (1184, 1195), bulk create/update/delete (1489, 1538, 1547), merge (1558), move (1590), transcription versioning (1620 to 1630).
- Nomikos skimmed (not modified): `nomikos/frontend/src/components/page-editor/PageEditorCanvas.tsx`, `nomikos/frontend/src/components/page-editor/PageEditorTranscriptionStrip.tsx`.

