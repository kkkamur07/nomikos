import Link from "next/link";
import type {
  DocumentWithPartsResponse,
  InferenceModelResponse,
  LineResponse,
  PageTranscriptionTextLineResponse,
  PairingProgressResponse,
} from "../../api/client";
import type { PageEditorProcessingKind } from "./hooks/usePageEditorRunState";
import { PageEditorBackLink } from "./PageEditorNavHeader";
import { PageEditorModelSelect } from "./PageEditorModelSelect";
import { PageEditorPager } from "./PageEditorPager";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";
import { ToolbarRunButtons } from "./ToolbarRunButtons";
import {
  SegmentHealthPopover,
  SettingsPopover,
  type ToolbarSegmentHealth,
} from "./ToolbarPanels";
import { ToolbarWorkflowMenu } from "./ToolbarWorkflowMenu";

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
  pairingProgress: PairingProgressResponse;
  partId: string;
  selectedSegmentId: string | null;
  textLines: PageTranscriptionTextLineResponse[];
  onPairTextLine: (order: number) => void;
  onDocumentWorkflowChange: (
    workflow: DocumentWithPartsResponse["workflow"],
  ) => void;
  actionsOpen: boolean;
  onActionsOpenChange: (open: boolean) => void;
  segmenting: boolean;
  ocrRunning: boolean;
  /** What the page is busy with, from `usePageEditorRunState`. */
  processingKind: PageEditorProcessingKind | null;
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
  segmentHealth: ToolbarSegmentHealth;
  canvasSettings: PageEditorCanvasSettings;
  onCanvasSettingsChange: (settings: PageEditorCanvasSettings) => void;
  preferLocalInference: boolean;
  onPreferLocalInferenceChange: (preferLocal: boolean) => void;
  preferenceSaving: boolean;
  /** **Capacity** for this account's own computer, as the platform reports it. */
  hasLocalCapacity: boolean;
  hostPreferenceLoading: boolean;
};

/** The toolbar's shorter wording for what the page is busy with. */
const BUSY_LABELS: Record<PageEditorProcessingKind, string> = {
  segmentation: "Segmenting",
  "transcription-page": "Transcribing page",
  "transcription-segment": "Transcribing",
};

function modelName(
  models: InferenceModelResponse[],
  modelId: string | null,
): string | undefined {
  return models.find((model) => model.id === modelId)?.name;
}

/** Screen-reader copies of what the bar shows, and the pairing buttons. */
function ToolbarAnnouncements({
  document,
  partIndex,
  pageCount,
  lineCount,
  pairingProgress,
  selectedSegmentId,
  textLines,
  onPairTextLine,
}: {
  document: DocumentWithPartsResponse;
  partIndex: number;
  pageCount: number;
  lineCount: number;
  pairingProgress: PairingProgressResponse;
  selectedSegmentId: string | null;
  textLines: PageTranscriptionTextLineResponse[];
  onPairTextLine: (order: number) => void;
}) {
  return (
    <>
      <span className="visually-hidden">ANNOTE PAGE WORKSPACE</span>
      <span className="visually-hidden">
        Pairing progress: {pairingProgress.paired_lines}/
        {pairingProgress.total_lines} Lines paired
      </span>
      <span className="visually-hidden">
        {lineCount} {lineCount === 1 ? "Segment" : "Segments"}
      </span>

      <div className="visually-hidden">
        {selectedSegmentId &&
          textLines.map((textLine) => (
            <button
              key={textLine.order}
              type="button"
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
    </>
  );
}

/** The way back to the document, the document's name, and the pager. */
function ToolbarTitle({
  projectId,
  documentId,
  documentName,
  partIndex,
  pageCount,
  ...pager
}: {
  projectId: string | undefined;
  documentId: string | undefined;
  documentName: string;
  partIndex: number;
  pageCount: number;
  hasPreviousPart: boolean;
  hasNextPart: boolean;
  onPreviousPart: () => void;
  onNextPart: () => void;
}) {
  return (
    <div className="pe-toolbar__title">
      {projectId && documentId && (
        <PageEditorBackLink
          to={`/projects/${projectId}/documents/${documentId}`}
        />
      )}
      <div className="pe-toolbar__sep" aria-hidden="true" />
      <h1
        className="pe-toolbar__doc"
        title={`${documentName} · Page ${partIndex}`}
      >
        {documentName}
        <span className="pe-toolbar__doc-page"> · p.{partIndex}</span>
      </h1>
      <PageEditorPager
        pageNumber={partIndex}
        pageCount={pageCount}
        {...pager}
      />
    </div>
  );
}

/** What the page is busy with, its Segment count, and pairing progress. */
function ToolbarStats({
  busyLabel,
  lineCount,
  pairingProgress,
}: {
  busyLabel: string | null;
  lineCount: number;
  pairingProgress: PairingProgressResponse;
}) {
  const { paired_lines: paired, total_lines: total } = pairingProgress;
  const pairingPercent = total > 0 ? Math.round((paired / total) * 100) : 0;
  return (
    <div className="pe-toolbar__center" aria-label="Page statistics">
      {busyLabel && (
        <div
          className="pe-toolbar__processing"
          role="status"
          aria-live="polite"
        >
          <span className="pe-toolbar__processing-dot" aria-hidden="true" />
          {busyLabel}
        </div>
      )}
      <div className="pe-toolbar__stat">
        <strong>{lineCount}</strong> {lineCount === 1 ? "seg" : "segs"}
      </div>
      <div
        className="pe-toolbar__progress"
        title={`${paired} of ${total} lines paired`}
      >
        <span className="pe-toolbar__stat">
          <strong>{paired}</strong>/{total}
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
  );
}

export function PageEditorToolbar(props: PageEditorToolbarProps) {
  const {
    projectId,
    documentId,
    document,
    partIndex,
    lines,
    selectedSegmentId,
    selectedTranscribeModelId,
    onActionsOpenChange,
  } = props;
  const processing = props.processingKind !== null;
  const hasModel = Boolean(selectedTranscribeModelId);
  const transcribeModelName =
    modelName(props.transcribeModels, selectedTranscribeModelId) ??
    "not selected";

  // The quick button transcribes what the researcher is looking at: the
  // selected segment if there is one, otherwise the page. Naming the scope on
  // the button is what keeps a one-click run from being a guess.
  const transcribeScope = selectedSegmentId ? "segment" : "page";
  const canTranscribe =
    !processing && hasModel && (Boolean(selectedSegmentId) || lines.length > 0);

  function runAutoSegment() {
    onActionsOpenChange(false);
    void props.onRunAutoSegment();
  }

  function runTranscribe() {
    onActionsOpenChange(false);
    if (selectedSegmentId) void props.onRunSegmentOcr();
    else void props.onRunPageOcr();
  }

  return (
    <header className="pe-toolbar" role="banner">
      <ToolbarAnnouncements
        document={document}
        partIndex={partIndex}
        pageCount={props.pageCount}
        lineCount={lines.length}
        pairingProgress={props.pairingProgress}
        selectedSegmentId={selectedSegmentId}
        textLines={props.textLines}
        onPairTextLine={props.onPairTextLine}
      />

      <Link
        href="/projects"
        className="pe-toolbar__logo"
        aria-label="nomikos home"
      >
        <img src="/nomos.svg" alt="" />
        <span>nomikos</span>
      </Link>

      <ToolbarTitle
        projectId={projectId}
        documentId={documentId}
        documentName={document.name}
        partIndex={partIndex}
        pageCount={props.pageCount}
        hasPreviousPart={props.hasPreviousPart}
        hasNextPart={props.hasNextPart}
        onPreviousPart={props.onPreviousPart}
        onNextPart={props.onNextPart}
      />

      <ToolbarStats
        busyLabel={props.processingKind && BUSY_LABELS[props.processingKind]}
        lineCount={lines.length}
        pairingProgress={props.pairingProgress}
      />

      <div className="pe-toolbar__actions">
        <div className="pe-toolbar__cluster">
          <PageEditorModelSelect
            label="HTR"
            ariaLabel="HTR transcription model"
            models={props.transcribeModels}
            selectedModelId={selectedTranscribeModelId}
            onSelectedModelIdChange={props.onSelectedTranscribeModelIdChange}
            disabled={processing}
          />
        </div>

        <div
          className="pe-toolbar__cluster"
          role="group"
          aria-label="Run inference"
        >
          <PageEditorModelSelect
            label="Seg"
            ariaLabel="Segmentation model"
            models={props.segmentModels}
            selectedModelId={props.selectedSegmentModelId}
            onSelectedModelIdChange={props.onSelectedSegmentModelIdChange}
            disabled={processing}
            includeDefaultOption
          />
          <ToolbarRunButtons
            processing={processing}
            segmenting={props.segmenting}
            ocrRunning={props.ocrRunning}
            transcribeScope={transcribeScope}
            canTranscribe={canTranscribe}
            transcribeModelName={hasModel ? transcribeModelName : null}
            onRunAutoSegment={runAutoSegment}
            onRunTranscribe={runTranscribe}
          />
        </div>

        <ToolbarWorkflowMenu
          open={props.actionsOpen}
          onOpenChange={onActionsOpenChange}
          processing={processing}
          segmenting={props.segmenting}
          ocrRunning={props.ocrRunning}
          segmentModelName={
            modelName(props.segmentModels, props.selectedSegmentModelId) ??
            "Default"
          }
          transcribeModelName={transcribeModelName}
          hasTranscribeModel={hasModel}
          hasSelectedSegment={selectedSegmentId !== null}
          hasSegments={lines.length > 0}
          onRunAutoSegment={props.onRunAutoSegment}
          onRunSegmentOcr={props.onRunSegmentOcr}
          onRunPageOcr={props.onRunPageOcr}
          projectId={projectId}
          documentId={documentId}
          partId={props.partId}
          document={document}
          partIndex={partIndex}
          onDocumentWorkflowChange={props.onDocumentWorkflowChange}
          transcriptionPdfOpen={props.transcriptionPdfOpen}
          onOpenTranscriptionPdf={props.onOpenTranscriptionPdf}
          onCloseTranscriptionPdf={props.onCloseTranscriptionPdf}
          hostPreferenceLoading={props.hostPreferenceLoading}
          hasLocalCapacity={props.hasLocalCapacity}
          preferLocalInference={props.preferLocalInference}
        />

        <div className="pe-toolbar__cluster">
          <SegmentHealthPopover
            open={props.segmentHealthOpen}
            onOpenChange={props.onSegmentHealthOpenChange}
            segmentHealth={props.segmentHealth}
            hasSegments={lines.length > 0}
          />
          <SettingsPopover
            open={props.settingsOpen}
            onOpenChange={props.onSettingsOpenChange}
            canvasSettings={props.canvasSettings}
            onCanvasSettingsChange={props.onCanvasSettingsChange}
            preferLocalInference={props.preferLocalInference}
            onPreferLocalInferenceChange={props.onPreferLocalInferenceChange}
            preferenceSaving={props.preferenceSaving}
            hasLocalCapacity={props.hasLocalCapacity}
          />
        </div>
      </div>
    </header>
  );
}
