import { SegmentIcon, TranscribeIcon } from "./EditorIcons";

type ToolbarRunButtonsProps = {
  processing: boolean;
  segmenting: boolean;
  ocrRunning: boolean;
  /** The bar is short of room: the buttons keep their icons and lose their text. */
  iconOnly: boolean;
  /** What the quick Transcribe button runs on: the selection, or the page. */
  transcribeScope: "segment" | "page";
  canTranscribe: boolean;
  /** The chosen segmentation model's name, or null when none is chosen. */
  segmentModelName: string | null;
  /** The chosen transcription model's name, or null when none is chosen. */
  transcribeModelName: string | null;
  onRunAutoSegment: () => void;
  onRunTranscribe: () => void;
};

export function toolbarButton(iconOnly: boolean): string {
  return iconOnly ? "pe-tb-btn pe-tb-btn--icon" : "pe-tb-btn";
}

/**
 * The two runs a researcher makes all day. Behind a menu they taxed the most
 * repeated action in the editor; the menu keeps the variants that need a
 * choice made first. Only when the bar runs out of room do they fold into
 * "More", where the Workflow menu already offers both.
 */
export function ToolbarRunButtons({
  processing,
  segmenting,
  ocrRunning,
  iconOnly,
  transcribeScope,
  canTranscribe,
  segmentModelName,
  transcribeModelName,
  onRunAutoSegment,
  onRunTranscribe,
}: ToolbarRunButtonsProps) {
  const segmentText = segmenting ? "Segmenting…" : "Segment";
  const transcribeText = ocrRunning ? "Transcribing…" : "Transcribe";
  return (
    <>
      <button
        type="button"
        className={toolbarButton(iconOnly)}
        disabled={processing}
        onClick={onRunAutoSegment}
        // The tooltip names the model the button will actually use. Null only
        // happens while the catalog is empty or failed to load, when the
        // select is disabled and the request omits the model id.
        title={
          segmentModelName === null
            ? "Segment this page"
            : `Segment this page with ${segmentModelName}`
        }
        aria-label={iconOnly ? segmentText : undefined}
      >
        <SegmentIcon />
        {!iconOnly && segmentText}
      </button>
      <button
        type="button"
        className={toolbarButton(iconOnly)}
        disabled={!canTranscribe}
        onClick={onRunTranscribe}
        title={
          transcribeModelName === null
            ? "Choose a model before transcribing"
            : `Transcribe the ${transcribeScope} with ${transcribeModelName}`
        }
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
  );
}
