type PageEditorBaselineBarProps = {
  onMoveDown: () => void;
  onSave: () => void;
  onReset: () => void;
};

/** The actions for a selected baseline, under the canvas. */
export function PageEditorBaselineBar({
  onMoveDown,
  onSave,
  onReset,
}: PageEditorBaselineBarProps) {
  return (
    <div className="pe-baseline-bar">
      <span>Selected baseline</span>
      <button
        type="button"
        className="btn btn--ghost btn--sm"
        onClick={onMoveDown}
      >
        Move baseline down
      </button>
      <button
        type="button"
        className="btn btn--active btn--sm"
        onClick={onSave}
      >
        Save layout
      </button>
      <button
        type="button"
        className="btn btn--danger-ghost btn--sm"
        onClick={onReset}
      >
        Reset layout
      </button>
    </div>
  );
}
