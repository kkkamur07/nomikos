import type {
  LineResponse,
  LineTranscriptionResponse,
  TranscriptionLayerResponse,
} from "../../api/client";
import { approvedText, modelReadingFor } from "./hooks/utils";

type StripViewInput = {
  selectedSegment: LineResponse | null;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  approvedTextDraft: string;
};

/**
 * The strip's one primary button. "Approve" copies the model's reading into
 * Ground truth unchanged, "Save" writes the draft, and "Approved" (disabled)
 * says the draft is what Ground truth already holds.
 */
export type StripAccept =
  | { action: "approve"; label: "Approve"; disabled: false }
  | { action: "save"; label: "Save"; disabled: boolean }
  | { action: "approved"; label: "Approved"; disabled: true };

type StripView = {
  modelOutputTranscription: LineTranscriptionResponse | null;
  /** The accessible name of the text box, which says what a save writes. */
  editLabel: string;
  /** Null when no Segment is selected. */
  accept: StripAccept | null;
  /** The box holds the model's reading untouched, shown like a suggestion. */
  showsModelReading: boolean;
};

/**
 * The strip for one selected Segment: which text is the model's suggestion,
 * and what its one primary button does with the draft.
 */
export function transcriptionStripView(input: StripViewInput): StripView {
  const { selectedSegment, approvedTextDraft } = input;
  const modelOutputTranscription = selectedSegment
    ? modelReadingFor(selectedSegment, input.selectedTranscriptionLayer)
    : null;
  if (!selectedSegment) {
    return {
      modelOutputTranscription,
      editLabel: "Edit transcription",
      accept: null,
      showsModelReading: false,
    };
  }
  const saved = approvedText(selectedSegment) ?? "";
  const hasGroundTruth = saved.trim() !== "";
  const editLabel = hasGroundTruth
    ? "Ground truth text for selected Segment"
    : "Approved text for selected Segment";
  const modelText = modelOutputTranscription?.text ?? "";
  const showsModelReading =
    !hasGroundTruth && modelText !== "" && approvedTextDraft === modelText;
  let accept: StripAccept;
  if (showsModelReading) {
    accept = { action: "approve", label: "Approve", disabled: false };
  } else if (hasGroundTruth && approvedTextDraft === saved) {
    accept = { action: "approved", label: "Approved", disabled: true };
  } else {
    accept = {
      action: "save",
      label: "Save",
      disabled: !hasGroundTruth && approvedTextDraft.trim() === "",
    };
  }
  return { modelOutputTranscription, editLabel, accept, showsModelReading };
}
