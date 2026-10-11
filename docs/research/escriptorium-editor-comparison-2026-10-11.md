# eScriptorium diplomatic panel vs the Nomikos transcript column

Date: 2026-10-11. Follows `docs/research/escriptorium-editor-2026-09-23.md` (on the unmerged branch `escriptorium-editor/research`, commit `a97e6d67`), which covers the store shape, the SVG visual panel and the editing lifecycle.
This note compares the line-by-line text editing surface only: eScriptorium's DiploPanel against the Nomikos side-by-side transcript column (draft PR #164).

eScriptorium source: `https://gitlab.com/scripta/escriptorium.git`, shallow clone at `ce909abbbb573165eb12983f2d69ad096d9698cb`, read only. Paths below are relative to that clone (`E:`) or to `nomikos/frontend/src` (`N:`).

## How eScriptorium binds segments to text in the diplomatic panel

- One Vue `DiploLine` per line, keyed `'DL' + line.pk` (E: `front/vue/components/DiploPanel.vue` 200 to 206). The component renders nothing itself (`<template />`); on mount it asks the panel to append a `<div data-line-pk>` into one shared `contenteditable` div (E: `DiploLine.vue` mounted, `DiploPanel.vue` `appendLine` 731 to 757).
- Text is DOM-first: `setElContent` writes `textContent` (E: `DiploLine.vue`), and the panel diffs every div against `line.currentTrans.content` on save (`addToList`, `DiploPanel.vue` 1174 onward).
- Row order follows `line.order`: a watcher moves the div to `parent.children[order]` (E: `DiploLine.vue` watch `line.order`). `constrainLineNumber` (E: `DiploPanel.vue` 759 to 815) runs on every keyup and focus-out: it strips stray `<br>`, appends divs until there is one per line, re-stamps `data-line-pk` by position, and marks surplus divs `alert-danger` "Line not present in segmentation".
- Identity is therefore by pk for store updates but by DOM position inside the editable div: pressing Enter in the contenteditable creates a new div that `constrainLineNumber` re-stamps, so text can shift between lines.

## Comparison

| Behaviour | eScriptorium | Nomikos before this branch | This branch |
|---|---|---|---|
| Row identity on add/delete | Vue key `DL+pk`, but text lives in one shared contenteditable re-stamped by position (E: `DiploPanel.vue` 759 to 815) | React key `line.id`, each row owns its own textarea and draft state (N: `components/page-editor/PageEditorTranscriptEditor.tsx`) | Adopted and tightened: rows stay keyed by `line.id` (390); an externally changed seed no longer overwrites a row holding an unsaved draft (162). Test: "keeps rows mounted and an unsaved draft when a segment is added or deleted" |
| Row order | `line.order`, reordering moves DOM nodes (E: `DiploLine.vue` watch) | geometric `readingOrder()` over baselines, which could disagree with the canvas numbers | Adopted: rows follow `segmentsInNumberOrder(lines)`, the stored `order` field the canvas numbers from (N: `PageEditorTranscriptEditor.tsx` 323), so server-owned reading order composes; the unused `readingOrder.ts` is deleted |
| Commit timing | `lodash.debounce(save, 10000)` after each input plus save on focus-out (E: `DiploPanel.vue` 429 to 434, 727, 822 to 829) | explicit commit on Enter and blur | Kept explicit (no debounce, by design): Enter commits and advances, blur commits, Ctrl/Cmd+Enter commits and stays (260) |
| Save path | `transcriptions/bulkUpdate` for all changed divs; errors only `console.log`, the store is not updated ("No store update ????", E: `front/src/editor/store/transcriptions.js` 77 to 97) | two copies: the page's `handleCommitText` and the hook's `saveApprovedText` | One path: `commitLineText(lineId, text)` in the pairing hook (N: `components/page-editor/hooks/usePairingState.ts` 172); `saveApprovedText` and the transcript column both call it |
| Dirty and failure state | a single panel-wide "unsaved" notification toggled by `changed()`/`save()` (E: `DiploPanel.vue` 727, 881) | error text under the row, no dirty marker | Adopted per row: `data-status` dirty (●), saving (…), saved (✓), unsaved (!) and "Not saved: <reason>" while the draft is kept (N: `PageEditorTranscriptEditor.tsx` 200 onward, `styles/page-editor-transcript.css`) |
| Conflicts with the server | none; last bulk write wins, no versions compared on write (versions are snapshotted server side) | last write wins; the response text is folded into current state | Unchanged (last write wins); a row with an unsaved draft is no longer re-seeded from a refetch |
| RTL | one direction per document: class from `mainTextDirection`/`readDirection` on the container, `text-align: right` from `front/css/rtl.css` 24 to 27; regions mode sets `dir` from `mainTextDirection` (E: `DiploPanel.vue` 197, 213, 254) | hard-coded `"ltr"` on the page | Adopted per row rather than per document: `dir` and `text-align` from the row's own dominant script, empty rows take the page's dominant direction (N: `utils/scriptDirection.ts`, `PageEditorTranscriptEditor.tsx` 198). Caret and selection follow the browser bidi algorithm inside the textarea |
| Script fonts | one resolved `transcriptionFont` from a document/project/user cascade applied to the editor (E: `front/src/editor/store/document.js` 80, 192 to 205; `Editor.vue` 165) | inherited UI font | Adopted lightly: `data-script` per row selects a Syriac, Hebrew, Arabic or Coptic font stack, and RTL rows force `letter-spacing: normal` so joining letters stay joined (N: `styles/page-editor-transcript.css`). No per-document font setting |
| Font size from geometry | visual panel only: `mask area / baseline length * 0.3 * fontSizeRatio` (E: `VisuLine.vue` 145 to 183) | fixed 16 px times `fontScale` | Adopted as an option, default off ("Size transcript text by segment height"): each row's strip height (`maskStripHeight`, N: `components/page-editor/textPanelGeometry.ts` 95) against the page median, clamped 11 to 28 px |
| Keyboard between rows | ArrowUp/ArrowDown move the caret to the previous/next div (E: `DiploPanel.vue` 898 to 966); modal steps lines in store order (E: `front/src/editor/store/lines.js` 459 to 470) | Enter, Escape, ArrowUp/ArrowDown | Adopted: Tab/Shift+Tab move rows like the arrows and fall through to normal Tab at the list ends (N: `PageEditorTranscriptEditor.tsx` 280) |
| Focus between panels | hover in any panel draws the mask overlay in the others (`showOverlay`); canvas selection follows `selectedToken`; editing the diplomatic panel blocks global shortcuts (`setBlockShortcuts`) | hover mirrored, focusing a row pans the canvas, incoming canvas selection scrolls the row but never takes focus | Unchanged (accepted in round 2) |
| Text placement | SVG `<textPath>` on the baseline in the visual panel | none | Not adopted: the notepad column stays; the round-1 SVG panel (`PageEditorTextPanel.tsx`, `useLinkedViewports.ts`, `page-editor-text-panel.css`) is deleted |

## Not adopted

- Debounced background saves: a ten-second timer saves text the user has not confirmed and hides failures; explicit commit with a per-row state is easier to trust.
- A single shared `contenteditable`: it is what lets text drift between lines after an Enter or a paste, and it needs `constrainLineNumber` to repair the DOM on every keystroke.
- Drag and drop reordering (SortableJS) and region-grouped rows: reading order is moving to the server in a parallel change.
- A per-document font and direction setting: the API has no script field yet; per-row detection covers mixed pages (Syriac text with Latin folio numbers) without one.
- Optimistic concurrency (version check on write): needs a backend change; out of scope for a frontend branch.

## Files

- E: `front/vue/components/DiploPanel.vue`, `DiploLine.vue`, `VisuLine.vue`, `Editor.vue`; `front/src/editor/store/lines.js`, `transcriptions.js`, `document.js`; `front/css/rtl.css`.
- N: `components/page-editor/PageEditorTranscriptEditor.tsx`, `hooks/usePairingState.ts`, `textPanelGeometry.ts`, `pageEditorSettings.ts`, `PageEditorSettingsPanel.tsx`; `pages/PageEditorPlaceholderPage.tsx`; `utils/scriptDirection.ts`; `styles/page-editor-transcript.css`.
