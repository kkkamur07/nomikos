import type {
  LineResponse,
  LineTranscriptionResponse,
  TranscriptionLayerResponse,
} from "../../api/client";
import { groundTruthLayer } from "../../utils/transcriptionLayerLabel";
import {
  lineTranscriptionForLayer,
  modelTranscriptionForLine,
  showsModelSourceReview,
  transcriptionForOcrReview,
} from "./hooks/utils";

type StripViewInput = {
  transcriptionLayers: TranscriptionLayerResponse[];
  selectedSegment: LineResponse | null;
  selectedSegmentNumber: number | null;
  selectedTranscriptionLayer: TranscriptionLayerResponse | null;
  approvedTextDraft: string;
};

/** What the strip's primary button does, and what it says. */
export type StripAccept =
  | { action: "save-ground-truth"; label: "Save" }
  | { action: "promote"; label: "Accept" }
  | { action: "save-approved"; label: "Save" };

export type StripView = {
  modelOutputTranscription: LineTranscriptionResponse | null;
  /** The accessible name of the text box, which says what a save writes. */
  editLabel: string;
  /** Null when the strip offers no save at all. */
  accept: StripAccept | null;
};

function groundTruthTranscriptionFor({
  transcriptionLayers,
  selectedSegment,
  selectedTranscriptionLayer,
}: StripViewInput): LineTranscriptionResponse | null {
  const layer =
    groundTruthLayer(transcriptionLayers) ?? selectedTranscriptionLayer;
  if (!selectedSegment || layer?.kind !== "ground_truth") return null;
  return lineTranscriptionForLayer(selectedSegment, layer.id);
}

function modelOutputFor({
  selectedSegment,
  selectedTranscriptionLayer,
}: StripViewInput): LineTranscriptionResponse | null {
  if (!selectedSegment) return null;
  const modelLayerId =
    selectedTranscriptionLayer?.kind === "model"
      ? selectedTranscriptionLayer.id
      : null;
  return modelTranscriptionForLine(selectedSegment, modelLayerId);
}

function showsOcrReview({
  selectedSegment,
  selectedTranscriptionLayer,
}: StripViewInput): boolean {
  if (!selectedSegment || !selectedTranscriptionLayer) return false;
  return Boolean(
    transcriptionForOcrReview(selectedSegment, selectedTranscriptionLayer),
  );
}

/**
 * Whether the footer offers to accept the model output as Ground truth.
 *
 * The output is a suggestion wherever it is shown, so the offer does not wait
 * for the model layer to be the selected one: a segment with no Ground truth
 * yet gets "Accept" while Ground truth is the open layer. Once the researcher
 * has typed into the box the button saves what they typed instead, because
 * accepting there would throw that typing away.
 */
function canPromoteModelOutput(
  input: StripViewInput,
  modelOutput: LineTranscriptionResponse | null,
  ocrReview: boolean,
): boolean {
  if (!modelOutput?.text) return false;
  return (
    input.selectedTranscriptionLayer?.kind === "model" ||
    ocrReview ||
    input.approvedTextDraft.trim() === ""
  );
}

/**
 * The strip for one selected Segment: which text is the model's suggestion,
 * and whether its primary button saves Ground truth, accepts the suggestion,
 * or saves approved text.
 */
export function transcriptionStripView(input: StripViewInput): StripView {
  const modelOutputTranscription = modelOutputFor(input);
  const groundTruth = groundTruthTranscriptionFor(input);
  if (groundTruth && !showsModelSourceReview(groundTruth)) {
    return {
      modelOutputTranscription,
      editLabel: "Ground truth text for selected Segment",
      accept: { action: "save-ground-truth", label: "Save" },
    };
  }
  const ocrReview = showsOcrReview(input);
  const approvedEditor = Boolean(input.selectedSegmentNumber) && !ocrReview;
  const editLabel = approvedEditor
    ? "Approved text for selected Segment"
    : "Edit transcription";
  if (canPromoteModelOutput(input, modelOutputTranscription, ocrReview)) {
    return {
      modelOutputTranscription,
      editLabel,
      accept: { action: "promote", label: "Accept" },
    };
  }
  return {
    modelOutputTranscription,
    editLabel,
    accept: approvedEditor ? { action: "save-approved", label: "Save" } : null,
  };
}
