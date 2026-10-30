import type { ReactNode } from "react";
import type { DocumentWithPartsResponse } from "../../api/client";
import { exportFileStem } from "../../utils/exportFilename";
import { MoreIcon, WorkflowIcon } from "./EditorIcons";
import { usePopoverDismiss } from "./hooks/usePopoverDismiss";
import { PageEditorDocxButton } from "./PageEditorDocxButton";
import { PageEditorInferenceStatus } from "./PageEditorInferenceStatus";
import { PageEditorPageXmlButton } from "./PageEditorPageXmlButton";
import { PageEditorSegmentHealthPanel } from "./PageEditorSegmentHealthPanel";
import { PageEditorSharingMenu } from "./PageEditorSharingMenu";
import type { ToolbarSegmentHealth } from "./ToolbarPanels";
import { toolbarButton } from "./ToolbarRunButtons";

type WorkflowRuns = {
  processing: boolean;
  segmenting: boolean;
  ocrRunning: boolean;
  segmentModelName: string;
  transcribeModelName: string;
  hasTranscribeModel: boolean;
  hasSelectedSegment: boolean;
  hasSegments: boolean;
  onRunAutoSegment: () => void;
  onRunSegmentOcr: () => void;
  onRunPageOcr: () => void;
};

type WorkflowExport = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string;
  document: DocumentWithPartsResponse;
  partIndex: number;
  onDocumentWorkflowChange: (
    workflow: DocumentWithPartsResponse["workflow"],
  ) => void;
  transcriptionPdfOpen: boolean;
  onOpenTranscriptionPdf: () => void;
  onCloseTranscriptionPdf: () => void;
};

type WorkflowHost = {
  hostPreferenceLoading: boolean;
  hasLocalCapacity: boolean;
  preferLocalInference: boolean;
};

type ToolbarWorkflowMenuProps = WorkflowRuns &
  WorkflowExport &
  WorkflowHost & {
    /**
     * "workflow" is the bar's own Workflow button; "more" is the one icon the
     * run buttons, the transcript toggle and Workflow fold into when the bar
     * runs out of room.
     */
    variant: "workflow" | "more";
    iconOnly: boolean;
    open: boolean;
    onOpenChange: (open: boolean) => void;
    healthOpen: boolean;
    onHealthOpenChange: (open: boolean) => void;
    segmentHealth: ToolbarSegmentHealth;
    /** Items above Check page: the transcript toggle, once it is folded in. */
    leadingItems?: ReactNode;
  };

/** A menu checkbox draws its tick through the class, not only aria-checked. */
export function checkableItem(checked: boolean): string {
  return checked ? "pe-dd-item pe-dd-item--checked" : "pe-dd-item";
}

/** Every menu item closes the menu before it acts. */
function closingThen(close: () => void, action: () => void) {
  return () => {
    close();
    action();
  };
}

function WorkflowRunItems({
  close,
  ...runs
}: WorkflowRuns & { close: () => void }) {
  const transcribeLabel = (idle: string) =>
    runs.ocrRunning ? "Transcribing…" : idle;
  const canTranscribeSegment =
    runs.hasTranscribeModel && runs.hasSelectedSegment;
  const canTranscribePage = runs.hasTranscribeModel && runs.hasSegments;
  return (
    <>
      <div className="pe-dd-section">Segment</div>
      <p className="pe-dd-model">
        Model <strong>{runs.segmentModelName}</strong>
      </p>
      <button
        type="button"
        role="menuitem"
        disabled={runs.processing}
        onClick={closingThen(close, () => void runs.onRunAutoSegment())}
        className="pe-dd-item"
      >
        {runs.segmenting ? "Segmenting…" : "Auto segment page"}
      </button>
      <div className="pe-dd-divider" />
      <div className="pe-dd-section">Transcribe</div>
      <p className="pe-dd-model">
        Model <strong>{runs.transcribeModelName}</strong>
      </p>
      <button
        type="button"
        role="menuitem"
        disabled={runs.processing || !canTranscribeSegment}
        onClick={closingThen(close, () => void runs.onRunSegmentOcr())}
        className="pe-dd-item"
      >
        {transcribeLabel("Selected segment")}
      </button>
      <button
        type="button"
        role="menuitem"
        disabled={runs.processing || !canTranscribePage}
        onClick={closingThen(close, () => void runs.onRunPageOcr())}
        className="pe-dd-item"
      >
        {transcribeLabel("Whole page")}
      </button>
    </>
  );
}

function WorkflowExportItems({
  close,
  processing,
  ...target
}: WorkflowExport & { close: () => void; processing: boolean }) {
  const { projectId, documentId, document } = target;
  const togglePdf = target.transcriptionPdfOpen
    ? target.onCloseTranscriptionPdf
    : target.onOpenTranscriptionPdf;
  return (
    <>
      <div className="pe-dd-section">Export</div>
      <button
        type="button"
        role="menuitemcheckbox"
        className={checkableItem(target.transcriptionPdfOpen)}
        aria-checked={target.transcriptionPdfOpen}
        onClick={closingThen(close, togglePdf)}
      >
        Transcription PDF
        <span className="pe-dd-meta">
          {target.transcriptionPdfOpen ? "open" : "preview"}
        </span>
      </button>
      {projectId && documentId && (
        <>
          <PageEditorDocxButton
            projectId={projectId}
            documentId={documentId}
            partId={target.partId}
            className="pe-dd-item"
            role="menuitem"
            onActivate={close}
            downloadFilename={`${exportFileStem(document.name, target.partIndex)}.docx`}
          />
          <PageEditorPageXmlButton
            projectId={projectId}
            documentId={documentId}
            partId={target.partId}
            className="pe-dd-item"
            role="menuitem"
            onActivate={close}
            downloadFilename={`${exportFileStem(document.name, target.partIndex)}.zip`}
          />
          <PageEditorSharingMenu
            projectId={projectId}
            documentId={documentId}
            workflow={document.workflow}
            publicShareToken={document.public_share_token ?? null}
            onWorkflowChange={target.onDocumentWorkflowChange}
            disabled={processing}
          />
        </>
      )}
    </>
  );
}

/**
 * The Workflow menu: Check page, the run variants that need a choice made
 * first, the exports, sharing, and where the work runs. One anchor holds both
 * popovers: the menu and, once Check page is picked from it, the health panel.
 */
export function ToolbarWorkflowMenu({
  variant,
  iconOnly,
  open,
  onOpenChange,
  healthOpen,
  onHealthOpenChange,
  segmentHealth,
  leadingItems,
  hostPreferenceLoading,
  hasLocalCapacity,
  preferLocalInference,
  ...items
}: ToolbarWorkflowMenuProps) {
  const close = () => onOpenChange(false);
  const dismiss = usePopoverDismiss<HTMLDivElement>(open || healthOpen, () => {
    if (open) onOpenChange(false);
    if (healthOpen) onHealthOpenChange(false);
  });
  const more = variant === "more";
  const menuLabel = more ? "More" : "Workflow";
  // The count only appears once a report has been read, so an unopened panel
  // never implies the page is clean.
  const findingCount = segmentHealth.report?.finding_count ?? 0;
  return (
    <div
      className={
        more ? "pe-dropdown-wrap" : "pe-toolbar__cluster pe-dropdown-wrap"
      }
      ref={dismiss.ref}
      onBlur={dismiss.onBlur}
    >
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={more || iconOnly ? menuLabel : undefined}
        title={
          more
            ? "More: Segment, Transcribe, Transcript, Workflow, Check page"
            : iconOnly
              ? "Workflow"
              : undefined
        }
        onClick={() => {
          onHealthOpenChange(false);
          onOpenChange(!open);
        }}
        className={`${toolbarButton(more || iconOnly)}${open ? " pe-tb-btn--on" : ""}`}
      >
        {more ? <MoreIcon /> : iconOnly ? <WorkflowIcon /> : "Workflow ▾"}
      </button>
      {open && (
        <div className="pe-dropdown" role="menu" aria-label={menuLabel}>
          {leadingItems}
          <button
            type="button"
            role="menuitem"
            className="pe-dd-item"
            disabled={!items.hasSegments}
            title="Look for segmentation mistakes on this page"
            onClick={() => {
              close();
              onHealthOpenChange(true);
            }}
          >
            {findingCount > 0 ? `Check page · ${findingCount}` : "Check page"}
          </button>
          <div className="pe-dd-divider" />
          <WorkflowRunItems close={close} {...items} />
          {/*
            Export is a workflow step, not a permanent fixture of the bar.
            PDF and XML were two of the least-pressed controls in the
            editor holding two of its most valuable slots, next to the
            settings gear. In the menu they sit beside Sharing, which is
            the same act at a different fidelity.
          */}
          <div className="pe-dd-divider" />
          <WorkflowExportItems close={close} {...items} />
          {/*
            Where the work runs belongs with the menu that starts the work.
            In the bar it announced the ordinary state permanently, which
            teaches people to stop reading the one spot a real warning would
            appear in. Here it is one glance away from the button that cares.
          */}
          <div className="pe-dd-divider" />
          <div className="pe-dd-footer">
            <PageEditorInferenceStatus
              loading={hostPreferenceLoading}
              hasLocalCapacity={hasLocalCapacity}
              preferLocalInference={preferLocalInference}
            />
          </div>
        </div>
      )}
      {healthOpen && (
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
