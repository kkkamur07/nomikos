import { useCallback, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type {
  DocumentWithPartsResponse,
  InferenceModelResponse,
  LineResponse,
  SegmentHealthResponse,
} from "../../api/client";
import { PageEditorBackLink } from "./PageEditorNavHeader";
import { FeedbackDialog } from "../layout/FeedbackDialog";
import { useFeedbackButton } from "../layout/useFeedbackButton";
import { PageEditorModelSelect } from "./PageEditorModelSelect";
import { PageEditorPager } from "./PageEditorPager";
import { PageEditorSharingMenu } from "./PageEditorSharingMenu";
import { PageEditorPageXmlButton } from "./PageEditorPageXmlButton";
import { PageEditorDocxButton } from "./PageEditorDocxButton";
import { exportFileStem } from "../../utils/exportFilename";
import {
  CheckIcon,
  CheckPageIcon,
  FeedbackIcon,
  MoreIcon,
  SegmentIcon,
  SettingsIcon,
  TranscribeIcon,
  WorkflowIcon,
} from "./EditorIcons";
import { useToolbarCollapse } from "./useToolbarCollapse";
import { PageEditorSettingsPanel } from "./PageEditorSettingsPanel";
import {
  PageEditorSegmentHealthPanel,
  type SegmentHealthAction,
} from "./PageEditorSegmentHealthPanel";
import { PageEditorInferenceStatus } from "./PageEditorInferenceStatus";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";

type PageEditorToolbarProps = {
  projectId: string | undefined;
  documentId: string | undefined;
  document: DocumentWithPartsResponse;
  partIndex: number;
  /** How many pages the document has, for the in-editor pager. */
  pageCount: number;
  hasPreviousPart: boolean;
  hasNextPart: boolean;
  onPreviousPart: () => void;
  onNextPart: () => void;
  lines: LineResponse[];
  pairingProgress: {
    paired_lines: number;
    total_lines: number;
    percent: number;
  };
  partId: string;
  selectedSegmentId: string | null;
  textLines: { order: number; text: string; paired_line_id: string | null }[];
  onPairTextLine: (order: number) => void;
  onDocumentWorkflowChange: (
    workflow: DocumentWithPartsResponse["workflow"],
  ) => void;
  actionsOpen: boolean;
  onActionsOpenChange: (open: boolean) => void;
  segmenting: boolean;
  ocrRunning: boolean;
  ocrScope?: "segment" | "page" | null;
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  segmentModels: InferenceModelResponse[];
  selectedSegmentModelId: string | null;
  onSelectedSegmentModelIdChange: (modelId: string | null) => void;
  onRunAutoSegment: () => void;
  onRunSegmentOcr: () => void;
  onRunPageOcr: () => void;
  transcriptionPdfOpen: boolean;
  onOpenTranscriptionPdf: () => void;
  onCloseTranscriptionPdf: () => void;
  settingsOpen: boolean;
  onSettingsOpenChange: (open: boolean) => void;
  segmentHealthOpen: boolean;
  onSegmentHealthOpenChange: (open: boolean) => void;
  /** Passed through from ``useSegmentHealth`` as one object: the toolbar owns
   *  none of this state, it only decides where the panel hangs. */
  segmentHealth: {
    report: SegmentHealthResponse | null;
    loading: boolean;
    error: string | null;
    pending: string | null;
    apply: (action: SegmentHealthAction) => void;
    refresh: () => void;
  };
  canvasSettings: PageEditorCanvasSettings;
  onCanvasSettingsChange: (settings: PageEditorCanvasSettings) => void;
  preferLocalInference: boolean;
  onPreferLocalInferenceChange: (preferLocal: boolean) => void;
  preferenceSaving: boolean;
  /** **Capacity** for this account's own computer, as the platform reports it. */
  hasLocalCapacity: boolean;
  hostPreferenceLoading: boolean;
};

function toolbarButton(iconOnly: boolean): string {
  return iconOnly ? "pe-tb-btn pe-tb-btn--icon" : "pe-tb-btn";
}

export function PageEditorToolbar({
  projectId,
  documentId,
  partId,
  document,
  partIndex,
  pageCount,
  hasPreviousPart,
  hasNextPart,
  onPreviousPart,
  onNextPart,
  lines,
  pairingProgress,
  selectedSegmentId,
  textLines,
  onPairTextLine,
  onDocumentWorkflowChange,
  actionsOpen,
  onActionsOpenChange,
  segmenting,
  ocrRunning,
  ocrScope = null,
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  segmentModels,
  selectedSegmentModelId,
  onSelectedSegmentModelIdChange,
  onRunAutoSegment,
  onRunSegmentOcr,
  onRunPageOcr,
  transcriptionPdfOpen,
  onOpenTranscriptionPdf,
  onCloseTranscriptionPdf,
  settingsOpen,
  onSettingsOpenChange,
  segmentHealthOpen,
  onSegmentHealthOpenChange,
  segmentHealth,
  canvasSettings,
  onCanvasSettingsChange,
  preferLocalInference,
  onPreferLocalInferenceChange,
  preferenceSaving,
  hasLocalCapacity,
  hostPreferenceLoading,
}: PageEditorToolbarProps) {
  const pathname = usePathname();
  const { dialogOpen, thanks, openDialog, closeDialog, handleSent } =
    useFeedbackButton();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  const segmentHealthRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLElement>(null);
  const moreRef = useCallback((element: HTMLDivElement | null) => {
    dropdownRef.current = element;
    segmentHealthRef.current = element;
  }, []);
  // The bar never wraps: as room runs out the labelled buttons go icon-only,
  // then fold into one "More" menu. What fits is measured, not guessed.
  const collapse = useToolbarCollapse(headerRef);
  const iconOnly = collapse >= 1;
  const folded = collapse >= 2;

  useEffect(() => {
    if (!actionsOpen && !settingsOpen && !segmentHealthOpen) return;
    function handleClick(event: MouseEvent) {
      const target = event.target as Node;
      if (
        actionsOpen &&
        dropdownRef.current &&
        !dropdownRef.current.contains(target)
      ) {
        onActionsOpenChange(false);
      }
      if (
        settingsOpen &&
        settingsRef.current &&
        !settingsRef.current.contains(target)
      ) {
        onSettingsOpenChange(false);
      }
      if (
        segmentHealthOpen &&
        segmentHealthRef.current &&
        !segmentHealthRef.current.contains(target)
      ) {
        onSegmentHealthOpenChange(false);
      }
    }
    globalThis.document.addEventListener("click", handleClick);
    return () => globalThis.document.removeEventListener("click", handleClick);
  }, [
    actionsOpen,
    settingsOpen,
    segmentHealthOpen,
    onActionsOpenChange,
    onSettingsOpenChange,
    onSegmentHealthOpenChange,
  ]);

  const segmentLabel = lines.length === 1 ? "seg" : "segs";
  const pairingPercent =
    pairingProgress.total_lines > 0
      ? Math.round(
          (pairingProgress.paired_lines / pairingProgress.total_lines) * 100,
        )
      : 0;
  const processing = segmenting || ocrRunning;
  const processingLabel = segmenting
    ? "Segmenting"
    : ocrRunning
      ? ocrScope === "page"
        ? "Transcribing page"
        : "Transcribing"
      : null;

  const selectedModelName =
    transcribeModels.find((model) => model.id === selectedTranscribeModelId)
      ?.name ?? "not selected";
  const selectedSegmentModelName =
    segmentModels.find((model) => model.id === selectedSegmentModelId)?.name ??
    "Default";

  // The quick button transcribes what the researcher is looking at: the
  // selected segment if there is one, otherwise the page. Naming the scope on
  // the button is what keeps a one-click run from being a guess.
  const transcribeScope = selectedSegmentId ? "segment" : "page";
  const canTranscribe =
    !processing &&
    Boolean(selectedTranscribeModelId) &&
    (Boolean(selectedSegmentId) || lines.length > 0);

  const segmentText = segmenting ? "Segmenting…" : "Segment";
  const segmentTitle = "Segment this page with blla-segment";
  const transcribeText = ocrRunning ? "Transcribing…" : "Transcribe";
  const transcribeTitle = selectedTranscribeModelId
    ? `Transcribe the ${transcribeScope} with ${selectedModelName}`
    : "Choose a model before transcribing";
  // The count only appears once a report has been read, so an unopened panel
  // never implies the page is clean.
  const findingCount =
    segmentHealth.report && segmentHealth.report.finding_count > 0
      ? segmentHealth.report.finding_count
      : null;
  const checkPageText =
    findingCount !== null ? `Check page · ${findingCount}` : "Check page";
  const healthPanel = segmentHealthOpen && (
    <PageEditorSegmentHealthPanel
      report={segmentHealth.report}
      loading={segmentHealth.loading}
      error={segmentHealth.error}
      pending={segmentHealth.pending}
      onApply={segmentHealth.apply}
      onRefresh={segmentHealth.refresh}
    />
  );
  const workflowItems = (
    <>
      <div className="pe-dd-section">Segment</div>
      <p className="pe-dd-model">
        Model <strong>{selectedSegmentModelName}</strong>
      </p>
      <button
        type="button"
        role="menuitem"
        disabled={processing}
        onClick={() => {
          onActionsOpenChange(false);
          void onRunAutoSegment();
        }}
        className="pe-dd-item"
      >
        {segmenting ? "Segmenting…" : "Auto segment page"}
      </button>
      <div className="pe-dd-divider" />
      <div className="pe-dd-section">Transcribe</div>
      <p className="pe-dd-model">
        Model <strong>{selectedModelName}</strong>
      </p>
      <button
        type="button"
        role="menuitem"
        disabled={
          !selectedSegmentId || processing || !selectedTranscribeModelId
        }
        onClick={() => {
          onActionsOpenChange(false);
          void onRunSegmentOcr();
        }}
        className="pe-dd-item"
      >
        {ocrRunning ? "Transcribing…" : "Selected segment"}
      </button>
      <button
        type="button"
        role="menuitem"
        disabled={
          processing || lines.length === 0 || !selectedTranscribeModelId
        }
        onClick={() => {
          onActionsOpenChange(false);
          void onRunPageOcr();
        }}
        className="pe-dd-item"
      >
        {ocrRunning ? "Transcribing…" : "Whole page"}
      </button>
      {/*
        Export is a workflow step, not a permanent fixture of the bar.
        PDF and XML were two of the least-pressed controls in the
        editor holding two of its most valuable slots, next to the
        settings gear. In the menu they sit beside Sharing, which is
        the same act at a different fidelity.
      */}
      <div className="pe-dd-divider" />
      <div className="pe-dd-section">Export</div>
      <button
        type="button"
        role="menuitemcheckbox"
        className="pe-dd-item"
        aria-checked={transcriptionPdfOpen}
        onClick={() => {
          onActionsOpenChange(false);
          if (transcriptionPdfOpen) onCloseTranscriptionPdf();
          else onOpenTranscriptionPdf();
        }}
      >
        Transcription PDF
        <span className="pe-dd-meta">
          {transcriptionPdfOpen ? "open" : "preview"}
        </span>
      </button>
      {projectId && documentId && (
        <PageEditorDocxButton
          projectId={projectId}
          documentId={documentId}
          partId={partId}
          className="pe-dd-item"
          role="menuitem"
          onActivate={() => onActionsOpenChange(false)}
          downloadFilename={`${exportFileStem(document.name, partIndex)}.docx`}
        />
      )}
      {projectId && documentId && (
        <PageEditorPageXmlButton
          projectId={projectId}
          documentId={documentId}
          partId={partId}
          className="pe-dd-item"
          role="menuitem"
          onActivate={() => onActionsOpenChange(false)}
          downloadFilename={`${exportFileStem(document.name, partIndex)}.zip`}
        />
      )}
      {projectId && documentId && (
        <PageEditorSharingMenu
          projectId={projectId}
          documentId={documentId}
          workflow={document.workflow}
          publicShareToken={document.public_share_token ?? null}
          onWorkflowChange={onDocumentWorkflowChange}
          disabled={processing}
        />
      )}
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
    </>
  );

  return (
    <header
      ref={headerRef}
      className="pe-toolbar"
      role="banner"
      data-collapse={collapse}
    >
      <span className="visually-hidden">ANNOTE PAGE WORKSPACE</span>
      <span className="visually-hidden">
        Pairing progress: {pairingProgress.paired_lines}/
        {pairingProgress.total_lines} Lines paired
      </span>
      <span className="visually-hidden">
        {lines.length} {lines.length === 1 ? "Segment" : "Segments"}
      </span>

      <div className="visually-hidden">
        {selectedSegmentId &&
          textLines.map((textLine) => (
            <button
              key={textLine.order}
              type="button"
              disabled={!selectedSegmentId}
              onClick={() => void onPairTextLine(textLine.order)}
            >
              Pair Text line {textLine.order + 1}
            </button>
          ))}
      </div>

      <span className="visually-hidden">
        {document.name} · Page {partIndex}
      </span>

      {/* Announced rather than only drawn, so a page turn made from the
          keyboard says where it landed instead of silently redrawing. */}
      <span className="visually-hidden" role="status" aria-live="polite">
        Page {partIndex} of {pageCount}
      </span>

      <Link
        href="/projects"
        className="pe-toolbar__logo"
        aria-label="nomikos home"
      >
        <img src="/nomos.svg" alt="" />
        <span>nomikos</span>
      </Link>

      <div className="pe-toolbar__title">
        {projectId && documentId && (
          <PageEditorBackLink
            to={`/projects/${projectId}/documents/${documentId}`}
          />
        )}
        <div className="pe-toolbar__sep" aria-hidden="true" />
        <h1
          className="pe-toolbar__doc"
          title={`${document.name} · Page ${partIndex}`}
        >
          {document.name}
          <span className="pe-toolbar__doc-page"> · p.{partIndex}</span>
        </h1>
        <PageEditorPager
          pageNumber={partIndex}
          pageCount={pageCount}
          hasPreviousPart={hasPreviousPart}
          hasNextPart={hasNextPart}
          onPreviousPart={onPreviousPart}
          onNextPart={onNextPart}
        />
      </div>

      <div className="pe-toolbar__center" aria-label="Page statistics">
        {processingLabel && (
          <div
            className="pe-toolbar__processing"
            role="status"
            aria-live="polite"
          >
            <span className="pe-toolbar__processing-dot" aria-hidden="true" />
            {processingLabel}
          </div>
        )}
        {/* One line, one group: "23 segs · 23/23" and the pairing track. */}
        <div
          className="pe-toolbar__progress"
          title={`${lines.length} ${segmentLabel}, ${pairingProgress.paired_lines} of ${pairingProgress.total_lines} lines paired`}
        >
          <span className="pe-toolbar__stat">
            <strong>{lines.length}</strong> {segmentLabel}
            <span className="pe-toolbar__stat-sep" aria-hidden="true">
              {" · "}
            </span>
            <strong>{pairingProgress.paired_lines}</strong>/
            {pairingProgress.total_lines}
          </span>
          <div
            className="pe-toolbar__progress-track"
            role="progressbar"
            aria-valuenow={pairingPercent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Pairing progress"
          >
            <div
              className="pe-toolbar__progress-fill"
              style={{ width: `${pairingPercent}%` }}
            />
          </div>
        </div>
      </div>

      <div className="pe-toolbar__actions">
        <div className="pe-toolbar__cluster">
          <PageEditorModelSelect
            label="HTR"
            ariaLabel="HTR transcription model"
            models={transcribeModels}
            selectedModelId={selectedTranscribeModelId}
            onSelectedModelIdChange={onSelectedTranscribeModelIdChange}
            disabled={processing}
          />
        </div>

        {/*
          The two runs a researcher makes all day get their own buttons. Behind
          a menu they taxed the most repeated action in the editor; the menu
          keeps the variants that need a choice made first. Only when the bar
          runs out of room do they fold into "More", where the Workflow menu
          already offers both.
        */}
        <div
          className="pe-toolbar__cluster"
          role="group"
          aria-label="Run inference"
        >
          <PageEditorModelSelect
            label="Seg"
            ariaLabel="Segmentation model"
            models={segmentModels}
            selectedModelId={selectedSegmentModelId}
            onSelectedModelIdChange={onSelectedSegmentModelIdChange}
            disabled={processing}
            includeDefaultOption
          />
          {!folded && (
            <>
              <button
                type="button"
                className={toolbarButton(iconOnly)}
                disabled={processing}
                onClick={() => {
                  onActionsOpenChange(false);
                  void onRunAutoSegment();
                }}
                title={segmentTitle}
                aria-label={iconOnly ? segmentText : undefined}
              >
                <SegmentIcon />
                {!iconOnly && segmentText}
              </button>
              <button
                type="button"
                className={toolbarButton(iconOnly)}
                disabled={!canTranscribe}
                onClick={() => {
                  onActionsOpenChange(false);
                  if (selectedSegmentId) void onRunSegmentOcr();
                  else void onRunPageOcr();
                }}
                title={transcribeTitle}
                aria-label={
                  iconOnly ? `${transcribeText} ${transcribeScope}` : undefined
                }
              >
                <TranscribeIcon />
                {!iconOnly && (
                  <>
                    {transcribeText}
                    <span className="pe-tb-btn__scope">{transcribeScope}</span>
                  </>
                )}
              </button>
            </>
          )}
        </div>

        {!folded && (
          <div
            className="pe-toolbar__cluster pe-dropdown-wrap"
            ref={dropdownRef}
          >
            <button
              type="button"
              aria-haspopup="menu"
              aria-expanded={actionsOpen}
              aria-label={iconOnly ? "Workflow" : undefined}
              title={iconOnly ? "Workflow" : undefined}
              onClick={() => onActionsOpenChange(!actionsOpen)}
              className={`${toolbarButton(iconOnly)}${actionsOpen ? " pe-tb-btn--on" : ""}`}
            >
              {iconOnly ? <WorkflowIcon /> : "Workflow ▾"}
            </button>
            {actionsOpen && (
              <div className="pe-dropdown" role="menu" aria-label="Workflow">
                {workflowItems}
              </div>
            )}
          </div>
        )}

        <div className="pe-toolbar__cluster">
          {!folded && (
            <div className="pe-dropdown-wrap" ref={segmentHealthRef}>
              <button
                type="button"
                className={`${toolbarButton(iconOnly)}${segmentHealthOpen ? " pe-tb-btn--on" : ""}`}
                aria-haspopup="dialog"
                aria-expanded={segmentHealthOpen}
                aria-label={iconOnly ? checkPageText : undefined}
                title="Look for segmentation mistakes on this page"
                disabled={lines.length === 0}
                onClick={() => onSegmentHealthOpenChange(!segmentHealthOpen)}
              >
                {iconOnly ? (
                  <>
                    <CheckPageIcon />
                    {findingCount !== null && (
                      <span className="pe-tb-btn__count">{findingCount}</span>
                    )}
                  </>
                ) : (
                  checkPageText
                )}
              </button>
              {healthPanel}
            </div>
          )}
          {folded && (
            // One anchor for both popovers: the menu and, once Check page is
            // picked from it, the health panel.
            <div className="pe-dropdown-wrap" ref={moreRef}>
              <button
                type="button"
                className={`pe-tb-btn pe-tb-btn--icon${actionsOpen ? " pe-tb-btn--on" : ""}`}
                aria-haspopup="menu"
                aria-expanded={actionsOpen}
                aria-label="More"
                title="More: Segment, Transcribe, Workflow, Check page"
                onClick={() => {
                  onSegmentHealthOpenChange(false);
                  onActionsOpenChange(!actionsOpen);
                }}
              >
                <MoreIcon />
              </button>
              {actionsOpen && (
                <div className="pe-dropdown" role="menu" aria-label="More">
                  <button
                    type="button"
                    role="menuitem"
                    className="pe-dd-item"
                    disabled={lines.length === 0}
                    onClick={() => {
                      onActionsOpenChange(false);
                      onSegmentHealthOpenChange(true);
                    }}
                  >
                    {checkPageText}
                  </button>
                  <div className="pe-dd-divider" />
                  {workflowItems}
                </div>
              )}
              {healthPanel}
            </div>
          )}
          {/* Icon-only: the text label cost a wrap in the counter at 1440px. */}
          <button
            type="button"
            className="pe-tb-btn pe-tb-btn--icon"
            title={thanks ? "Thanks" : "Send feedback"}
            aria-label={thanks ? "Thanks" : "Send feedback"}
            onClick={openDialog}
          >
            {thanks ? <CheckIcon /> : <FeedbackIcon />}
          </button>
          <div className="pe-dropdown-wrap" ref={settingsRef}>
            <button
              type="button"
              className={`pe-tb-btn pe-tb-btn--icon${settingsOpen ? " pe-tb-btn--on" : ""}`}
              aria-haspopup="dialog"
              aria-expanded={settingsOpen}
              aria-label="Editor settings"
              title="Editor settings"
              onClick={() => onSettingsOpenChange(!settingsOpen)}
            >
              <SettingsIcon />
            </button>
            {settingsOpen && (
              <PageEditorSettingsPanel
                settings={canvasSettings}
                onSettingsChange={onCanvasSettingsChange}
                preferLocalInference={preferLocalInference}
                onPreferLocalInferenceChange={onPreferLocalInferenceChange}
                preferenceSaving={preferenceSaving}
                hasLocalCapacity={hasLocalCapacity}
              />
            )}
          </div>
        </div>
      </div>
      <FeedbackDialog
        open={dialogOpen}
        pagePath={pathname ?? "/"}
        onClose={closeDialog}
        onSent={handleSent}
      />
    </header>
  );
}
