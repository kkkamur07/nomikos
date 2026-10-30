import { PAGE_EDITOR_SHORTCUTS } from "./pageEditorShortcuts";

/**
 * The merge action, on screen only while a group is selected.
 *
 * It sits above the tool island rather than inside it. The island is a roving
 * toolbar with a fixed set of tools, always there and always the same size; a
 * button that appears and disappears would change its tab order under the
 * user's hands. This is a contextual bar instead: it exists exactly when a
 * merge is possible, and it says what it is about to act on.
 */

type PageEditorMergeBarProps = {
  count: number;
  busy: boolean;
  error: string | null;
  onMerge: () => void;
  onClear: () => void;
};

export function PageEditorMergeBar({
  count,
  busy,
  error,
  onMerge,
  onClear,
}: PageEditorMergeBarProps) {
  if (count < 2) return null;

  return (
    <div className="pe-merge-bar" role="status" aria-live="polite">
      <span className="pe-merge-bar__count">{count} segments selected</span>
      <button
        type="button"
        className="pe-merge-bar__btn pe-merge-bar__btn--primary"
        onClick={onMerge}
        disabled={busy}
        aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.MERGE}
        title={`Merge the selected segments into one (${PAGE_EDITOR_SHORTCUTS.MERGE})`}
      >
        {busy ? "Merging…" : `Merge (${count})`}
      </button>
      <button type="button" className="pe-merge-bar__btn" onClick={onClear}>
        Clear
      </button>
      {error ? <span className="pe-merge-bar__error">{error}</span> : null}
    </div>
  );
}
