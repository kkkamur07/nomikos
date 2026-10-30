import type { SegmentHealthResponse } from "../../api/client";
import { SettingsIcon } from "./EditorIcons";
import { usePopoverDismiss } from "./hooks/usePopoverDismiss";
import type { SegmentHealthAction } from "./PageEditorSegmentHealthPanel";
import { PageEditorSettingsPanel } from "./PageEditorSettingsPanel";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";

export type ToolbarSegmentHealth = {
  report: SegmentHealthResponse | null;
  loading: boolean;
  error: string | null;
  pending: string | null;
  apply: (action: SegmentHealthAction) => void;
  refresh: () => void;
};

type SettingsPopoverProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  canvasSettings: PageEditorCanvasSettings;
  onCanvasSettingsChange: (settings: PageEditorCanvasSettings) => void;
  preferLocalInference: boolean;
  onPreferLocalInferenceChange: (preferLocal: boolean) => void;
  preferenceSaving: boolean;
  hasLocalCapacity: boolean;
};

/** The gear and the editor settings panel it opens. */
export function SettingsPopover({
  open,
  onOpenChange,
  canvasSettings,
  onCanvasSettingsChange,
  ...preference
}: SettingsPopoverProps) {
  const dismiss = usePopoverDismiss<HTMLDivElement>(open, () =>
    onOpenChange(false),
  );
  return (
    <div className="pe-dropdown-wrap" ref={dismiss.ref} onBlur={dismiss.onBlur}>
      <button
        type="button"
        className={`pe-tb-btn pe-tb-btn--icon${open ? " pe-tb-btn--on" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Editor settings"
        title="Editor settings"
        onClick={() => onOpenChange(!open)}
      >
        <SettingsIcon />
      </button>
      {open && (
        <PageEditorSettingsPanel
          settings={canvasSettings}
          onSettingsChange={onCanvasSettingsChange}
          {...preference}
        />
      )}
    </div>
  );
}
