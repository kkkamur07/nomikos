import type { SegmentHealthResponse } from "../../api/client";
import { SettingsIcon } from "./EditorIcons";
import { useDismissOnOutsideClick } from "./hooks/useDismissOnOutsideClick";
import {
  PageEditorSegmentHealthPanel,
  type SegmentHealthAction,
} from "./PageEditorSegmentHealthPanel";
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

type SegmentHealthPopoverProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  segmentHealth: ToolbarSegmentHealth;
  hasSegments: boolean;
};

/** "Check page" and the segment health panel it opens. */
export function SegmentHealthPopover({
  open,
  onOpenChange,
  segmentHealth,
  hasSegments,
}: SegmentHealthPopoverProps) {
  const wrapRef = useDismissOnOutsideClick<HTMLDivElement>(open, () =>
    onOpenChange(false),
  );
  const findingCount = segmentHealth.report?.finding_count ?? 0;
  return (
    <div className="pe-dropdown-wrap" ref={wrapRef}>
      <button
        type="button"
        className={`pe-tb-btn${open ? " pe-tb-btn--on" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Look for segmentation mistakes on this page"
        disabled={!hasSegments}
        onClick={() => onOpenChange(!open)}
      >
        Check page
        {/* The count only appears once a report has been read, so an
            unopened panel never implies the page is clean. */}
        {findingCount > 0 ? ` · ${findingCount}` : ""}
      </button>
      {open && (
        <PageEditorSegmentHealthPanel
          report={segmentHealth.report}
          loading={segmentHealth.loading}
          error={segmentHealth.error}
          pending={segmentHealth.pending}
          onApply={segmentHealth.apply}
          onRefresh={segmentHealth.refresh}
        />
      )}
    </div>
  );
}

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
  const wrapRef = useDismissOnOutsideClick<HTMLDivElement>(open, () =>
    onOpenChange(false),
  );
  return (
    <div className="pe-dropdown-wrap" ref={wrapRef}>
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
