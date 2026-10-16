import { CharacterConfidenceText } from "./CharacterConfidenceText";
import {
  characterConfidencesForTranscription,
  type LineTranscriptionWithCharacterConfidence,
} from "./characterConfidence";

type ModelOutputBlockProps = {
  transcription: LineTranscriptionWithCharacterConfidence | null;
  segmentNumber: number | null;
  onRerunOcr?: () => void;
  ocrRunning?: boolean;
  ocrDisabled?: boolean;
  ocrDisabledReason?: string;
};

function emptyMessageFor(
  segmentNumber: number | null,
  transcription: LineTranscriptionWithCharacterConfidence | null,
): string {
  if (!segmentNumber) return "Select a segment to view model output.";
  if (transcription !== null) {
    return "OCR finished with no text for this segment.";
  }
  return "No OCR yet. Run OCR on this segment.";
}

function ocrButtonLabel(hasText: boolean, ocrRunning: boolean): string {
  if (ocrRunning) return "Running…";
  return hasText ? "Re-run OCR" : "Run OCR";
}

function ModelOutputText({
  transcription,
  segmentNumber,
  hasText,
}: {
  transcription: LineTranscriptionWithCharacterConfidence | null;
  segmentNumber: number | null;
  hasText: boolean;
}) {
  if (!transcription || !hasText) {
    return (
      <span className="pe-model-inline__empty">
        {emptyMessageFor(segmentNumber, transcription)}
      </span>
    );
  }
  return (
    <CharacterConfidenceText
      characterConfidences={characterConfidencesForTranscription(transcription)}
      ariaLabel={
        segmentNumber
          ? `OCR model output for segment ${segmentNumber}`
          : "OCR model output"
      }
    />
  );
}

/** The model's text for the selected Segment, coloured by confidence. */
export function ModelOutputBlock({
  transcription,
  segmentNumber,
  onRerunOcr,
  ocrRunning = false,
  ocrDisabled = false,
  ocrDisabledReason,
}: ModelOutputBlockProps) {
  const text = transcription?.text ?? "";
  const hasText = text.trim() !== "";

  return (
    <div className="pe-model-inline">
      <span className="pe-model-inline__label" id="model-output-label">
        Model output:
      </span>
      <div
        className="pe-model-inline__body"
        role="text"
        aria-labelledby="model-output-label"
        aria-describedby="confidence-legend"
      >
        <ModelOutputText
          transcription={transcription}
          segmentNumber={segmentNumber}
          hasText={hasText}
        />
      </div>
      <div className="pe-model-inline__actions">
        {hasText && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={() => void navigator.clipboard.writeText(text)}
          >
            Copy
          </button>
        )}
        {onRerunOcr && (
          <button
            type="button"
            className="btn btn-outline btn-xs"
            disabled={ocrDisabled || ocrRunning}
            title={ocrDisabled ? ocrDisabledReason : undefined}
            onClick={() => void onRerunOcr()}
          >
            {ocrButtonLabel(hasText, ocrRunning)}
          </button>
        )}
      </div>
    </div>
  );
}
