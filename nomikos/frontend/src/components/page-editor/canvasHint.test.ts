import { describe, expect, it } from "vitest";

import type { LineResponse } from "../../api/client";
import { canvasHintFor, type CanvasHintState } from "./canvasHint";

function segment(groundTruth: string | null): LineResponse {
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
    line_transcriptions: groundTruth
      ? [
          {
            id: "lt-1",
            transcription_id: "gt",
            transcription_kind: "ground_truth",
            text: groundTruth,
            confidence: null,
          },
        ]
      : [],
  };
}

const IDLE: CanvasHintState = {
  partLoading: false,
  partIndex: 3,
  processingLabel: null,
  drawMode: "none",
  draftPointCount: 0,
  selectedSegment: null,
  selectedSegmentNumber: null,
  selectedVertexIndex: null,
};

describe("canvasHintFor", () => {
  it.each<[string, Partial<CanvasHintState>, string]>([
    ["a page turn", { partLoading: true }, "Loading page 3…"],
    [
      "a page turn before the index is known",
      { partLoading: true, partIndex: null },
      "Loading page…",
    ],
    [
      "a running job, over any tool",
      { processingLabel: "Segmentation in progress", drawMode: "polygon" },
      "Segmentation in progress…",
    ],
    [
      "an empty polygon",
      { drawMode: "polygon" },
      "Polygon: click to place the first corner · hold Space to pan",
    ],
    [
      "a one-point polygon",
      { drawMode: "polygon", draftPointCount: 1 },
      "Polygon: 1 point · click to add · double-click or Enter to finish",
    ],
    [
      "a three-point polygon",
      { drawMode: "polygon", draftPointCount: 3 },
      "Polygon: 3 points · click to add · double-click or Enter to finish",
    ],
    [
      "the rectangle tool",
      { drawMode: "rectangle" },
      "Rectangle: drag to draw a segment · hold Space to pan",
    ],
    [
      "nothing selected",
      {},
      "Drag to pan · scroll to zoom · Page Up/Down for pages · select a segment to edit it",
    ],
    [
      "a selected vertex",
      {
        selectedSegment: segment(null),
        selectedSegmentNumber: 4,
        selectedVertexIndex: 1,
      },
      "Segment 4 · vertex 2 selected · Delete removes point · Esc deselects",
    ],
    [
      "an unpaired segment",
      { selectedSegment: segment(null), selectedSegmentNumber: 4 },
      "Segment 4 unpaired · click edge to add · click handle to select · Esc deselects",
    ],
    [
      "a paired segment",
      { selectedSegment: segment("λόγος"), selectedSegmentNumber: 4 },
      "Segment 4 paired · click edge to add · click handle to select · Esc deselects",
    ],
  ])("describes %s", (_case, overrides, hint) => {
    expect(canvasHintFor({ ...IDLE, ...overrides })).toBe(hint);
  });
});
