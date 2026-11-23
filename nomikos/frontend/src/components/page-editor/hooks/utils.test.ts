import { describe, expect, it } from "vitest";

import type {
  LineResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";
import {
  initialDraftFor,
  mergeSavedLine,
  modelLayerIdForPromotion,
  modelReadingFor,
  segmentHasGroundTruth,
  segmentIdsWithGroundTruth,
  syncLayoutLinesFromSegments,
} from "./utils";

const MODEL_LAYER: TranscriptionLayerResponse = {
  id: "model-1",
  document_id: "doc-1",
  name: "Kraken run",
  kind: "model",
  created_by_job_id: "job-1",
  created_at: "2026-06-16T10:01:00Z",
};

const GROUND_TRUTH_LAYER: TranscriptionLayerResponse = {
  id: "ground-truth-1",
  document_id: "doc-1",
  name: "Ground truth",
  kind: "ground_truth",
  created_by_job_id: null,
  created_at: "2026-06-16T10:00:00Z",
};

const LINE = {
  id: "line-1",
  part_id: "part-1",
  block_id: null,
  order: 0,
  kind: "polygon",
  points: [
    [10, 10],
    [50, 10],
    [50, 30],
    [10, 30],
  ],
  baseline: {
    points: [
      [10, 30],
      [50, 30],
    ],
  },
  mask: {
    points: [
      [10, 10],
      [50, 10],
      [50, 30],
      [10, 30],
    ],
  },
  source: "manual",
  source_metadata: null,
  kraken_ceiling: null,
  manual_geometry: true,
  line_transcriptions: [
    {
      id: "line-tx-ground-1",
      transcription_id: "ground-truth-1",
      transcription_kind: "ground_truth",
      text: "model suggestion",
      confidence: null,
    },
    {
      id: "line-tx-model-1",
      transcription_id: "model-1",
      transcription_kind: "model",
      text: "model suggestion",
      confidence: 0.91,
    },
  ],
  created_at: "2026-06-16T10:00:00Z",
} satisfies LineResponse;

describe("page editor transcription utils", () => {
  it("resolves the model layer id used for ground truth promotion", () => {
    expect(modelLayerIdForPromotion(LINE, MODEL_LAYER)).toBe("model-1");
    expect(modelLayerIdForPromotion(LINE, GROUND_TRUTH_LAYER)).toBe("model-1");
  });

  it("treats segments with saved ground truth text as paired on the canvas", () => {
    expect(segmentHasGroundTruth(LINE)).toBe(true);
    expect(segmentIdsWithGroundTruth([LINE])).toEqual(new Set(["line-1"]));
    expect(
      segmentHasGroundTruth({
        ...LINE,
        id: "line-2",
        line_transcriptions: [],
      }),
    ).toBe(false);
  });

  it("syncs layout line geometry from segment state", () => {
    const layout = syncLayoutLinesFromSegments(
      {
        blocks: [],
        lines: [
          {
            id: "line-1",
            baseline: {
              points: [
                [0, 0],
                [10, 0],
              ],
            },
            manual_geometry: false,
          },
        ],
      },
      [
        {
          ...LINE,
          baseline: {
            points: [
              [60, 140],
              [300, 150],
            ],
          },
          manual_geometry: true,
        },
      ],
    );

    expect(layout.lines[0]?.baseline).toEqual({
      points: [
        [60, 140],
        [300, 150],
      ],
    });
    expect(layout.lines[0]?.manual_geometry).toBe(true);
  });

  it("merges a saved line into segment state by id or appends in order", () => {
    const krakenLine = {
      ...LINE,
      id: "line-2",
      order: 1,
      block_id: "block-1",
      source: "kraken",
      source_metadata: { model: "kraken:blla" },
      kraken_ceiling: [
        [-1, 9],
        [11, 9],
        [11, 16],
        [-1, 16],
      ],
    } as LineResponse;

    expect(
      mergeSavedLine([LINE], {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      }),
    ).toEqual([
      {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      },
    ]);
    expect(mergeSavedLine([LINE], krakenLine)).toEqual([LINE, krakenLine]);
    expect(
      mergeSavedLine([krakenLine, LINE], {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      }),
    ).toEqual([
      krakenLine,
      {
        ...LINE,
        points: [
          [12, 12],
          [52, 12],
          [52, 32],
          [12, 32],
        ],
      },
    ]);
  });
});

function lineWith(
  ...rows: [kind: "ground_truth" | "model", layerId: string, text: string][]
): LineResponse {
  return {
    ...LINE,
    line_transcriptions: rows.map(([kind, layerId, text]) => ({
      id: `${layerId}-line`,
      transcription_id: layerId,
      transcription_kind: kind,
      text,
      confidence: null,
    })),
  };
}

describe("initialDraftFor", () => {
  it("returns Ground truth even when model rows exist", () => {
    const line = lineWith(
      ["ground_truth", "ground-truth-1", "typed"],
      ["model", "model-1", "read"],
    );
    expect(initialDraftFor(line, "model-1")).toBe("typed");
  });

  it("returns the model text for a model-only line", () => {
    expect(initialDraftFor(lineWith(["model", "model-1", "read"]), null)).toBe(
      "read",
    );
  });

  it("prefers the selected layer's model row, else the newest", () => {
    const line = lineWith(
      ["model", "model-1", "older"],
      ["model", "model-2", "newer"],
    );
    expect(initialDraftFor(line, "model-1")).toBe("older");
    expect(initialDraftFor(line, null)).toBe("newer");
  });

  it("falls through to the model when Ground truth is blank", () => {
    const line = lineWith(
      ["ground_truth", "ground-truth-1", "   "],
      ["model", "model-1", "read"],
    );
    expect(initialDraftFor(line, null)).toBe("read");
  });

  it("returns an empty string when there is nothing", () => {
    expect(initialDraftFor(lineWith(), null)).toBe("");
  });
});

describe("modelReadingFor", () => {
  it("picks the selected model layer's row, else the newest", () => {
    const line = lineWith(
      ["model", "model-1", "older"],
      ["model", "model-2", "newer"],
    );
    expect(modelReadingFor(line, MODEL_LAYER)?.text).toBe("older");
    expect(modelReadingFor(line, GROUND_TRUTH_LAYER)?.text).toBe("newer");
    expect(modelReadingFor(line, null)?.text).toBe("newer");
    expect(modelReadingFor(lineWith(), null)).toBeNull();
  });
});
