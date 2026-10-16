import type { DocumentWithPartsResponse } from "../../api/client";
import { exportFileStem } from "../../utils/exportFilename";
import { useDismissOnOutsideClick } from "./hooks/useDismissOnOutsideClick";
import { PageEditorInferenceStatus } from "./PageEditorInferenceStatus";
import { PageEditorPageXmlButton } from "./PageEditorPageXmlButton";
import { PageEditorSharingMenu } from "./PageEditorSharingMenu";

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
    open: boolean;
    onOpenChange: (open: boolean) => void;
  };

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
        className="pe-dd-item"
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
 * The Workflow menu: the run variants that need a choice made first, the
 * exports, sharing, and where the work runs.
 */
export function ToolbarWorkflowMenu({
  open,
  onOpenChange,
  hostPreferenceLoading,
  hasLocalCapacity,
  preferLocalInference,
  ...items
}: ToolbarWorkflowMenuProps) {
  const close = () => onOpenChange(false);
  const wrapRef = useDismissOnOutsideClick<HTMLDivElement>(open, close);
  return (
    <div className="pe-toolbar__cluster pe-dropdown-wrap" ref={wrapRef}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => onOpenChange(!open)}
        className={`pe-tb-btn${open ? " pe-tb-btn--on" : ""}`}
      >
        Workflow ▾
      </button>
      {open && (
        <div className="pe-dropdown" role="menu" aria-label="Workflow">
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
    </div>
  );
}
