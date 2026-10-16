import type { ChangeEventHandler } from "react";
import type {
  InferenceModelResponse,
  LineResponse,
  TranscriptionLayerResponse,
} from "../../api/client";
import { ModelOutputBlock } from "./ModelOutputBlock";
import { PageEditorModelSelect } from "./PageEditorModelSelect";
import { PageEditorSegmentNav } from "./PageEditorSegmentNav";
import { useStripResize } from "./hooks/useStripResize";
import {
  transcriptionStripView,
  type StripAccept,
} from "./transcriptionStripView";

type PageEditorTranscriptionStripProps = {
  visible: boolean;
  transcriptionLayers: TranscriptionLayerResponse[];
  selectedTranscriptionLayerId: string | null;
  onSelectTranscriptionLayer: ChangeEventHandler<HTMLSelectElement>;
  selectedSegmentNumber: number | null;
  selectedSegment: LineResponse | null;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  approvedTextDraft: string;
  onApprovedTextDraftChange: (value: string) => void;
  onSaveGroundTruthText: () => void;
  onSaveApprovedText: () => void;
  onPromoteSelectedSegmentToGroundTruth: () => void;
  onRunSegmentOcr: () => void;
  onNavigateSegment: (direction: -1 | 1) => void;
  onDismiss: () => void;
  lines: LineResponse[];
  selectedSegmentId: string | null;
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  ocrRunning: boolean;
  ocrScope?: "segment" | "page" | null;
  backgroundJobsActive?: boolean;
};

type StripHeaderProps = {
  segmentNumber: number | null;
  totalSegments: number;
  ocrRunning: boolean;
  transcribeModels: InferenceModelResponse[];
  selectedTranscribeModelId: string | null;
  onSelectedTranscribeModelIdChange: (modelId: string | null) => void;
  onNavigateSegment: (direction: -1 | 1) => void;
  onDismiss: () => void;
};

function StripHeader({
  segmentNumber,
  totalSegments,
  ocrRunning,
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  onNavigateSegment,
  onDismiss,
}: StripHeaderProps) {
  return (
    <header className="pe-strip__bar">
      <h2 className="pe-strip__heading" id="strip-heading">
        <span className="pe-strip__badge" aria-hidden="true">
          {segmentNumber ?? "-"}
        </span>
        <span id="strip-heading-text" aria-live="polite">
          {segmentNumber ? `Segment ${segmentNumber}` : "Transcription"}
        </span>
      </h2>
      <PageEditorSegmentNav
        segmentNumber={segmentNumber}
        totalSegments={totalSegments}
        onPrevious={() => onNavigateSegment(-1)}
        onNext={() => onNavigateSegment(1)}
        disabled={ocrRunning}
      />
      <PageEditorModelSelect
        label="HTR"
        ariaLabel="HTR transcription model"
        models={transcribeModels}
        selectedModelId={selectedTranscribeModelId}
        onSelectedModelIdChange={onSelectedTranscribeModelIdChange}
        disabled={ocrRunning}
      />
      <span className="pe-strip__bar-spacer" />
      <button
        type="button"
        className="pe-tb-btn"
        aria-label="Hide transcription panel"
        title="Hide panel"
        onClick={onDismiss}
      >
        ✕
      </button>
    </header>
  );
}

/** Why the segment's OCR cannot run, or undefined when it can. */
function rerunDisabledReason(
  selectedTranscribeModelId: string | null,
  selectedSegmentId: string | null,
): string | undefined {
  if (!selectedTranscribeModelId) return "Select an HTR model above";
  if (!selectedSegmentId) return "Select a segment first";
  return undefined;
}

type RerunOcrButtonProps = {
  segmentNumber: number | null;
  disabled: boolean;
  running: boolean;
  hasOutput: boolean;
  /** The chosen model's display name, or null when none is chosen. */
  modelName: string | null;
  onRun: () => void;
};

function RerunOcrButton({
  segmentNumber,
  disabled,
  running,
  hasOutput,
  modelName,
  onRun,
}: RerunOcrButtonProps) {
  const idleLabel = hasOutput ? "Re-run OCR on segment" : "Run OCR on segment";
  return (
    <button
      type="button"
      className="btn btn-outline btn-xs"
      disabled={disabled}
      aria-label={
        segmentNumber
          ? `Re-run OCR on segment ${segmentNumber}`
          : "Re-run OCR on selected segment"
      }
      title={
        modelName === null
          ? "Select an OCR model first"
          : `Run ${modelName} on this segment`
      }
      onClick={() => void onRun()}
    >
      {running ? "Running OCR…" : idleLabel}
    </button>
  );
}

function ConfidenceLegend() {
  return (
    <div
      className="pe-strip__legend"
      id="confidence-legend"
      aria-label="Confidence thresholds"
    >
      <span>
        <i className="dot dot--high" aria-hidden="true" /> high &gt;90%
      </span>
      <span>
        <i className="dot dot--mid" aria-hidden="true" /> mid 51-90%
      </span>
      <span>
        <i className="dot dot--low" aria-hidden="true" /> low ≤50%
      </span>
    </div>
  );
}

function acceptHandler(
  accept: StripAccept,
  props: PageEditorTranscriptionStripProps,
): () => void {
  switch (accept.action) {
    case "save-ground-truth":
      return props.onSaveGroundTruthText;
    case "promote":
      return props.onPromoteSelectedSegmentToGroundTruth;
    case "save-approved":
      return props.onSaveApprovedText;
  }
}

export function PageEditorTranscriptionStrip(
  props: PageEditorTranscriptionStripProps,
) {
  const {
    selectedSegmentNumber,
    selectedSegmentId,
    selectedTranscribeModelId,
    approvedTextDraft,
    ocrRunning,
    onRunSegmentOcr,
  } = props;
  const { height, onPointerDown, onPointerMove, onPointerUp } =
    useStripResize(240);

  if (!props.visible) {
    return null;
  }

  const view = transcriptionStripView(props);
  const output = view.modelOutputTranscription;
  const segmentOcrRunning =
    ocrRunning && props.ocrScope === "segment" && !props.backgroundJobsActive;
  const rerunReason = rerunDisabledReason(
    selectedTranscribeModelId,
    selectedSegmentId,
  );
  const rerunDisabled = rerunReason !== undefined || ocrRunning;
  const modelName = selectedTranscribeModelId
    ? (props.transcribeModels.find(
        (model) => model.id === selectedTranscribeModelId,
      )?.name ?? "HTR model")
    : null;
  const accept = view.accept;

  return (
    <section
      className="pe-strip"
      style={{ height }}
      aria-labelledby="strip-heading"
    >
      <div
        className="pe-strip__resizer"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize transcription panel"
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      <StripHeader
        segmentNumber={selectedSegmentNumber}
        totalSegments={props.lines.length}
        ocrRunning={ocrRunning}
        transcribeModels={props.transcribeModels}
        selectedTranscribeModelId={selectedTranscribeModelId}
        onSelectedTranscribeModelIdChange={
          props.onSelectedTranscribeModelIdChange
        }
        onNavigateSegment={props.onNavigateSegment}
        onDismiss={props.onDismiss}
      />

      <div className="pe-strip__content">
        <ModelOutputBlock
          transcription={output}
          segmentNumber={selectedSegmentNumber}
          onRerunOcr={onRunSegmentOcr}
          ocrRunning={segmentOcrRunning}
          ocrDisabled={rerunDisabled}
          ocrDisabledReason={rerunReason}
        />

        <div className="pe-tx-block pe-tx-block--edit pe-tx-block--compact">
          <div className="pe-tx-block__head">
            <label className="pe-tx-block__label" htmlFor="strip-edit">
              Ground truth
            </label>
          </div>
          <textarea
            className="pe-strip__edit"
            id="strip-edit"
            aria-label={view.editLabel}
            spellCheck={false}
            placeholder="Accept model output or edit…"
            value={approvedTextDraft}
            disabled={!selectedSegmentNumber}
            onChange={(e) => props.onApprovedTextDraftChange(e.target.value)}
          />
        </div>

        <footer className="pe-strip__footer">
          <ConfidenceLegend />
          <div className="pe-strip__footer-actions">
            <RerunOcrButton
              segmentNumber={selectedSegmentNumber}
              disabled={rerunDisabled}
              running={segmentOcrRunning}
              hasOutput={Boolean(output?.text)}
              modelName={modelName}
              onRun={onRunSegmentOcr}
            />
            {accept && (
              <button
                type="button"
                className="btn btn-accent btn-xs"
                disabled={!selectedSegmentNumber}
                aria-label={accept.label}
                onClick={() => void acceptHandler(accept, props)()}
              >
                {accept.label}
              </button>
            )}
          </div>
        </footer>
      </div>
    </section>
  );
}
