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

const NEWER_MODEL: TranscriptionLayerResponse = {
  ...MODEL,
  id: "model-2",
  name: "Newer model",
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
    selectedSegment,
    selectedTranscriptionLayer,
    approvedTextDraft,
  });
  return {
    output: result.modelOutputTranscription?.text ?? null,
    editLabel: result.editLabel,
    accept: result.accept,
    showsModelReading: result.showsModelReading,
  };
}

describe("transcriptionStripView", () => {
  it("offers no accept with no segment selected", () => {
    expect(view(null, GROUND_TRUTH)).toEqual({
      output: null,
      editLabel: "Edit transcription",
      accept: null,
      showsModelReading: false,
    });
  });

  it("approves an untouched model reading when there is no Ground truth", () => {
    expect(view(segment(text(MODEL, "read")), null, "read")).toEqual({
      output: "read",
      editLabel: "Approved text for selected Segment",
      accept: { action: "approve", label: "Approve", disabled: false },
      showsModelReading: true,
    });
  });

  it("saves an edited draft of a model-only line", () => {
    expect(view(segment(text(MODEL, "read")), null, "mine")).toEqual({
      output: "read",
      editLabel: "Approved text for selected Segment",
      accept: { action: "save", label: "Save", disabled: false },
      showsModelReading: false,
    });
  });

  it("shows Approved and disabled when the draft is the saved Ground truth", () => {
    expect(
      view(
        segment(text(GROUND_TRUTH, "typed"), text(MODEL, "read")),
        null,
        "typed",
      ),
    ).toEqual({
      output: "read",
      editLabel: "Ground truth text for selected Segment",
      accept: { action: "approved", label: "Approved", disabled: true },
      showsModelReading: false,
    });
  });

  it("does not offer Approve when Ground truth equals the model text", () => {
    const result = view(
      segment(text(GROUND_TRUTH, "read", "ground_truth"), text(MODEL, "read")),
      null,
      "read",
    );
    expect(result.accept).toEqual({
      action: "approved",
      label: "Approved",
      disabled: true,
    });
    expect(result.showsModelReading).toBe(false);
  });

  it("enables Save when the draft differs from saved Ground truth", () => {
    const line = segment(text(GROUND_TRUTH, "typed"), text(MODEL, "read"));
    expect(view(line, null, "changed").accept).toEqual({
      action: "save",
      label: "Save",
      disabled: false,
    });
    expect(view(line, null, "").accept).toEqual({
      action: "save",
      label: "Save",
      disabled: false,
    });
  });

  it("disables Save with no Ground truth, no model text and a blank draft", () => {
    const line = segment(text(MODEL, ""));
    for (const draft of ["", "   "]) {
      expect(view(line, null, draft).accept).toEqual({
        action: "save",
        label: "Save",
        disabled: true,
      });
    }
    expect(view(segment(), null, "").accept?.disabled).toBe(true);
  });

  it("enables Save for typed text with no Ground truth and no model text", () => {
    expect(view(segment(), null, "typed").accept).toEqual({
      action: "save",
      label: "Save",
      disabled: false,
    });
  });

  it("treats blank Ground truth as no Ground truth", () => {
    const result = view(
      segment(text(GROUND_TRUTH, "  "), text(MODEL, "read")),
      null,
      "read",
    );
    expect(result.accept?.action).toBe("approve");
    expect(result.editLabel).toBe("Approved text for selected Segment");
  });

  it("uses the selected model layer's row, else the newest model row", () => {
    const line = segment(text(MODEL, "older"), text(NEWER_MODEL, "newer"));
    expect(view(line, MODEL).output).toBe("older");
    expect(view(line, NEWER_MODEL).output).toBe("newer");
    expect(view(line, null).output).toBe("newer");
    expect(view(line, GROUND_TRUTH).output).toBe("newer");
  });

  it("approves the selected layer's reading, not the newest", () => {
    const line = segment(text(MODEL, "older"), text(NEWER_MODEL, "newer"));
    expect(view(line, MODEL, "older").accept?.action).toBe("approve");
    expect(view(line, MODEL, "newer").accept?.action).toBe("save");
  });
});
