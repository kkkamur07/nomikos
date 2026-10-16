type ToolbarRunButtonsProps = {
  processing: boolean;
  segmenting: boolean;
  ocrRunning: boolean;
  /** What the quick Transcribe button runs on: the selection, or the page. */
  transcribeScope: "segment" | "page";
  canTranscribe: boolean;
  /** The chosen transcription model's name, or null when none is chosen. */
  transcribeModelName: string | null;
  onRunAutoSegment: () => void;
  onRunTranscribe: () => void;
};

/**
 * The two runs a researcher makes all day. Behind a menu they taxed the most
 * repeated action in the editor; the menu keeps the variants that need a
 * choice made first.
 */
export function ToolbarRunButtons({
  processing,
  segmenting,
  ocrRunning,
  transcribeScope,
  canTranscribe,
  transcribeModelName,
  onRunAutoSegment,
  onRunTranscribe,
}: ToolbarRunButtonsProps) {
  return (
    <>
      <button
        type="button"
        className="pe-tb-btn"
        disabled={processing}
        onClick={onRunAutoSegment}
        title="Segment this page with blla-segment"
      >
        <svg
          className="pe-tb-btn__icon"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <rect x="2.2" y="2.6" width="11.6" height="3" rx="0.8" />
          <rect x="2.2" y="6.9" width="11.6" height="3" rx="0.8" />
          <path d="M2.2 12.6h7.4" />
        </svg>
        {segmenting ? "Segmenting…" : "Segment"}
      </button>
      <button
        type="button"
        className="pe-tb-btn"
        disabled={!canTranscribe}
        onClick={onRunTranscribe}
        title={
          transcribeModelName === null
            ? "Choose a model before transcribing"
            : `Transcribe the ${transcribeScope} with ${transcribeModelName}`
        }
      >
        <svg
          className="pe-tb-btn__icon"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M2.6 4.2h10.8M8 4.2v8.2M5.6 12.4h4.8" />
        </svg>
        {ocrRunning ? "Transcribing…" : "Transcribe"}
        <span className="pe-tb-btn__scope">{transcribeScope}</span>
      </button>
    </>
  );
}
