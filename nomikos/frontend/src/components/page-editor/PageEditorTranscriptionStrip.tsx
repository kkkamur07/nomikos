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
  lineTranscriptionForLayer,
  modelTranscriptionForLine,
  showsModelSourceReview,
  transcriptionForOcrReview,
} from "./hooks/utils";
import { groundTruthLayer } from "../../utils/transcriptionLayerLabel";

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

export function PageEditorTranscriptionStrip({
  visible,
  transcriptionLayers,
  selectedSegmentNumber,
  selectedSegment,
  selectedTranscriptionLayer,
  approvedTextDraft,
  onApprovedTextDraftChange,
  onSaveGroundTruthText,
  onSaveApprovedText,
  onPromoteSelectedSegmentToGroundTruth,
  onRunSegmentOcr,
  onNavigateSegment,
  onDismiss,
  lines,
  selectedSegmentId,
  transcribeModels,
  selectedTranscribeModelId,
  onSelectedTranscribeModelIdChange,
  ocrRunning,
  ocrScope = null,
  backgroundJobsActive = false,
}: PageEditorTranscriptionStripProps) {
  const { height, onPointerDown, onPointerMove, onPointerUp } = useStripResize(
    240,
    visible,
  );

  const activeGroundTruthLayer =
    groundTruthLayer(transcriptionLayers) ?? selectedTranscriptionLayer;
  const groundTruthTranscription =
    selectedSegment && activeGroundTruthLayer?.kind === "ground_truth"
      ? lineTranscriptionForLayer(selectedSegment, activeGroundTruthLayer.id)
      : null;
  const modelName =
    transcribeModels.find((m) => m.id === selectedTranscribeModelId)?.name ??
    "HTR model";
  const segmentOcrRunning =
    ocrRunning && ocrScope === "segment" && !backgroundJobsActive;
  const rerunDisabled =
    !selectedSegmentId || !selectedTranscribeModelId || ocrRunning;
  const showGroundTruthEditor =
    Boolean(groundTruthTranscription) &&
    !showsModelSourceReview(groundTruthTranscription);
  const modelOutputTranscription = selectedSegment
    ? modelTranscriptionForLine(
        selectedSegment,
        selectedTranscriptionLayer?.kind === "model"
          ? selectedTranscriptionLayer.id
          : null,
      )
    : null;
  const ocrReviewTranscription =
    selectedSegment && selectedTranscriptionLayer
      ? transcriptionForOcrReview(selectedSegment, selectedTranscriptionLayer)
      : null;
  const showOcrReview = Boolean(ocrReviewTranscription);
  const showApprovedEditor =
    Boolean(selectedSegmentNumber) && !showGroundTruthEditor && !showOcrReview;
  /**
   * Whether the footer offers to accept the model output as Ground truth.
   *
   * The output is a suggestion wherever it is shown, so the offer no longer
   * waits for the model layer to be the selected one: a segment with no Ground
   * truth yet gets "Accept" while Ground truth is the open layer. Once the
   * researcher has typed into the box the button saves what they typed
   * instead, because accepting there would throw that typing away.
   */
  const canPromoteModelOutput = Boolean(
    modelOutputTranscription?.text &&
    !showGroundTruthEditor &&
    (selectedTranscriptionLayer?.kind === "model" ||
      showOcrReview ||
      approvedTextDraft.trim() === ""),
  );

  const onAccept = showGroundTruthEditor
    ? onSaveGroundTruthText
    : canPromoteModelOutput
      ? onPromoteSelectedSegmentToGroundTruth
      : showApprovedEditor
        ? onSaveApprovedText
        : undefined;

  const acceptLabel = showGroundTruthEditor
    ? "Save"
    : canPromoteModelOutput
      ? "Accept"
      : "Save";

  if (!visible) {
    return null;
  }

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
      <header className="pe-strip__bar">
        <h2 className="pe-strip__heading" id="strip-heading">
          <span className="pe-strip__badge" aria-hidden="true">
            {selectedSegmentNumber ?? "-"}
          </span>
          <span id="strip-heading-text" aria-live="polite">
            {selectedSegmentNumber
              ? `Segment ${selectedSegmentNumber}`
              : "Transcription"}
          </span>
        </h2>
        <PageEditorSegmentNav
          segmentNumber={selectedSegmentNumber}
          totalSegments={lines.length}
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

      <div className="pe-strip__content">
        <ModelOutputBlock
          layout="inline"
          transcription={modelOutputTranscription}
          segmentNumber={selectedSegmentNumber}
          onRerunOcr={onRunSegmentOcr}
          ocrRunning={segmentOcrRunning}
          ocrDisabled={rerunDisabled}
          ocrDisabledReason={
            !selectedTranscribeModelId
              ? "Select an HTR model above"
              : !selectedSegmentId
                ? "Select a segment first"
                : undefined
          }
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
            aria-label={
              showGroundTruthEditor
                ? "Ground truth text for selected Segment"
                : showApprovedEditor
                  ? "Approved text for selected Segment"
                  : "Edit transcription"
            }
            spellCheck={false}
            placeholder="Accept model output or edit…"
            value={approvedTextDraft}
            disabled={!selectedSegmentNumber}
            onChange={(e) => onApprovedTextDraftChange(e.target.value)}
          />
        </div>

        <footer className="pe-strip__footer">
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
          <div className="pe-strip__footer-actions">
            <button
              type="button"
              className="btn btn-outline btn-xs"
              disabled={rerunDisabled}
              aria-label={
                selectedSegmentNumber
                  ? `Re-run OCR on segment ${selectedSegmentNumber}`
                  : "Re-run OCR on selected segment"
              }
              title={
                selectedTranscribeModelId
                  ? `Run ${modelName} on this segment`
                  : "Select an OCR model first"
              }
              onClick={() => void onRunSegmentOcr()}
            >
              {segmentOcrRunning
                ? "Running OCR…"
                : modelOutputTranscription?.text
                  ? "Re-run OCR on segment"
                  : "Run OCR on segment"}
            </button>
            {onAccept &&
              (showGroundTruthEditor ||
                canPromoteModelOutput ||
                showApprovedEditor) && (
                <button
                  type="button"
                  className="btn btn-accent btn-xs"
                  disabled={!selectedSegmentNumber}
                  aria-label={acceptLabel}
                  onClick={() => void onAccept()}
                >
                  {acceptLabel}
                </button>
              )}
          </div>
        </footer>
      </div>
    </section>
  );
}
