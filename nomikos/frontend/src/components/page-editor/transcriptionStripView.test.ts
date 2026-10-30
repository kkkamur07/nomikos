import { describe, expect, it } from "vitest";

import type {
  LineResponse,
  LineTranscriptionResponse,
  TranscriptionLayerResponse,
} from "../../api/client";
import { transcriptionStripView } from "./transcriptionStripView";

const GROUND_TRUTH: TranscriptionLayerResponse = {
  id: "gt",
  document_id: "doc-1",
  name: "Ground truth",
  kind: "ground_truth",
  created_by_job_id: null,
  created_at: "2026-06-16T10:00:00Z",
};

const MODEL: TranscriptionLayerResponse = {
  ...GROUND_TRUTH,
  id: "model",
  name: "Model",
  kind: "model",
};

function text(
  layer: TranscriptionLayerResponse,
  value: string,
  kind = layer.kind,
): LineTranscriptionResponse {
  return {
    id: `${layer.id}-line`,
    transcription_id: layer.id,
    transcription_kind: kind,
    text: value,
    confidence: 0.8,
  };
}

function segment(...transcriptions: LineTranscriptionResponse[]): LineResponse {
  return {
    id: "S",
    part_id: "part-1",
    block_id: null,
    order: 0,
    kind: "polygon",
    points: [],
    baseline: null,
    mask: null,
    source: "machine",
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: false,
    created_at: "2026-06-16T10:00:00Z",
    line_transcriptions: transcriptions,
  };
}

function view(
  selectedSegment: LineResponse | null,
  selectedTranscriptionLayer: TranscriptionLayerResponse | null,
  approvedTextDraft = "",
) {
  const result = transcriptionStripView({
    transcriptionLayers: [GROUND_TRUTH, MODEL],
    selectedSegment,
    selectedSegmentNumber: selectedSegment ? 1 : null,
    selectedTranscriptionLayer,
    approvedTextDraft,
  });
  return {
    output: result.modelOutputTranscription?.text ?? null,
    editLabel: result.editLabel,
    accept: result.accept?.action ?? null,
    label: result.accept?.label ?? null,
  };
}

describe("transcriptionStripView", () => {
  it("saves Ground truth when the segment has its own", () => {
    expect(
      view(
        segment(text(GROUND_TRUTH, "typed"), text(MODEL, "read")),
        GROUND_TRUTH,
      ),
    ).toEqual({
      output: "read",
      editLabel: "Ground truth text for selected Segment",
      accept: "save-ground-truth",
      label: "Save",
    });
  });

  it("offers the model output while the Ground truth box is empty", () => {
    expect(view(segment(text(MODEL, "read")), GROUND_TRUTH)).toEqual({
      output: "read",
      editLabel: "Approved text for selected Segment",
      accept: "promote",
      label: "Accept",
    });
  });

  it("saves what the researcher typed instead of accepting over it", () => {
    expect(view(segment(text(MODEL, "read")), GROUND_TRUTH, "mine")).toEqual({
      output: "read",
      editLabel: "Approved text for selected Segment",
      accept: "save-approved",
      label: "Save",
    });
  });

  it("accepts from the model layer even with a typed draft", () => {
    expect(view(segment(text(MODEL, "read")), MODEL, "mine")).toEqual({
      output: "read",
      editLabel: "Edit transcription",
      accept: "promote",
      label: "Accept",
    });
  });

  it("reviews model text that was copied into Ground truth", () => {
    expect(
      view(
        segment(text(GROUND_TRUTH, "read", "model"), text(MODEL, "read")),
        GROUND_TRUTH,
        "mine",
      ),
    ).toEqual({
      output: "read",
      editLabel: "Edit transcription",
      accept: "promote",
      label: "Accept",
    });
  });

  it("offers no save with an empty model layer open", () => {
    expect(view(segment(text(MODEL, "")), MODEL, "mine")).toEqual({
      output: "",
      editLabel: "Edit transcription",
      accept: null,
      label: null,
    });
  });

  it("offers no save with nothing selected", () => {
    expect(view(null, GROUND_TRUTH)).toEqual({
      output: null,
      editLabel: "Edit transcription",
      accept: null,
      label: null,
    });
  });
});
