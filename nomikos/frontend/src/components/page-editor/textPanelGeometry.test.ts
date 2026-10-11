import { describe, expect, it } from "vitest";

import type { LineResponse } from "../../api/client";
import {
  baselinePath,
  baselinePoints,
  displayText,
  lineFontSize,
  polygonArea,
  polygonBounds,
  polylineLength,
} from "./textPanelGeometry";

function makeLine(overrides: Record<string, unknown> = {}): LineResponse {
  return {
    id: "line-1",
    order: 0,
    baseline: [],
    mask: null,
    points: [],
    line_transcriptions: [],
    ...overrides,
  } as unknown as LineResponse;
}

function transcription(
  kind: "ground_truth" | "model",
  text: string,
  id = `${kind}-1`,
) {
  return {
    id: `${id}-row`,
    transcription_id: id,
    transcription_kind: kind,
    text,
    confidence: null,
  };
}

describe("textPanelGeometry", () => {
  it("derives a fake baseline at the mask lower quarter", () => {
    const line = makeLine({
      baseline: [],
      mask: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
    });
    expect(baselinePoints(line)).toEqual([
      [10, 25],
      [50, 25],
    ]);
  });

  it("falls back to points when the mask is null", () => {
    const line = makeLine({
      baseline: [],
      mask: null,
      points: [
        [0, 0],
        [20, 0],
        [20, 40],
        [0, 40],
      ],
    });
    expect(baselinePoints(line)).toEqual([
      [0, 30],
      [20, 30],
    ]);
  });

  it("returns no baseline when no geometry exists at all", () => {
    expect(baselinePoints(makeLine())).toEqual([]);
  });

  it("normalizes {points} and {type, coordinates} baselines", () => {
    const fromPoints = makeLine({
      baseline: {
        points: [
          [0, 0],
          [10, 0],
        ],
      },
    });
    const fromCoordinates = makeLine({
      baseline: {
        type: "LineString",
        coordinates: [
          [0, 0],
          [3, 4],
        ],
      },
    });
    expect(baselinePoints(fromPoints)).toEqual([
      [0, 0],
      [10, 0],
    ]);
    expect(baselinePoints(fromCoordinates)).toEqual([
      [0, 0],
      [3, 4],
    ]);
    expect(polylineLength(baselinePoints(fromCoordinates))).toBe(5);
  });

  it("formats a baseline path with integers", () => {
    expect(
      baselinePath([
        [10.4, 25.6],
        [50, 25],
      ]),
    ).toBe("M 10 26 L 50 25");
    expect(baselinePath([])).toBe("");
  });

  it("measures polyline length, area and bounds", () => {
    const rect = [
      [10, 10],
      [50, 10],
      [50, 30],
      [10, 30],
    ] as [number, number][];
    expect(
      polylineLength([
        [0, 0],
        [3, 4],
      ]),
    ).toBe(5);
    expect(polylineLength([])).toBe(0);
    expect(polygonArea(rect)).toBe(800);
    expect(polygonBounds(rect)).toEqual({
      x: 10,
      y: 10,
      width: 40,
      height: 20,
    });
    expect(polygonBounds([])).toEqual({ x: 0, y: 0, width: 0, height: 0 });
  });

  it("sizes the font from the mask strip height", () => {
    const line = makeLine({
      baseline: [
        [10, 25],
        [50, 25],
      ],
      mask: [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ],
    });
    expect(lineFontSize(line, 1)).toBe(13);
    expect(lineFontSize(line, 2)).toBe(26);
  });

  it("clamps the font size and defaults without a polygon", () => {
    const tall = makeLine({
      baseline: [
        [0, 0],
        [10, 0],
      ],
      mask: [
        [0, 0],
        [10, 0],
        [10, 10000],
        [0, 10000],
      ],
    });
    expect(lineFontSize(tall, 1)).toBe(400);
    const thin = makeLine({
      baseline: [
        [0, 0],
        [100, 0],
      ],
      mask: [
        [0, 0],
        [100, 0],
        [100, 1],
        [0, 1],
      ],
    });
    expect(lineFontSize(thin, 1)).toBe(8);
    expect(lineFontSize(makeLine(), 1)).toBe(20);
  });

  it("prefers ground truth over model text", () => {
    const line = makeLine({
      line_transcriptions: [
        transcription("model", "model words", "model-1"),
        transcription("ground_truth", "true words", "gt-1"),
      ],
    });
    expect(displayText(line, null)).toEqual({
      text: "true words",
      source: "ground_truth",
    });
  });

  it("falls back to the preferred model layer, then to none", () => {
    const line = makeLine({
      line_transcriptions: [
        transcription("model", "first", "model-1"),
        transcription("model", "second", "model-2"),
      ],
    });
    expect(displayText(line, "model-1")).toEqual({
      text: "first",
      source: "model",
    });
    expect(displayText(line, null)).toEqual({
      text: "second",
      source: "model",
    });
    expect(displayText(makeLine(), null)).toEqual({ text: "", source: "none" });
  });

  it("falls back to the bbox baseline for an identical-point baseline", () => {
    const line = makeLine({
      baseline: [
        [650, 158],
        [650, 158],
      ],
      mask: [
        [500, 100],
        [800, 100],
        [800, 160],
        [500, 160],
      ],
    });
    expect(baselinePoints(line)).toEqual([
      [500, 145],
      [800, 145],
    ]);
  });

  it("falls back for a short stub under a wide polygon", () => {
    const line = makeLine({
      baseline: [
        [600, 150],
        [633, 150],
      ],
      mask: [
        [500, 100],
        [800, 100],
        [800, 160],
        [500, 160],
      ],
    });
    expect(baselinePoints(line)).toEqual([
      [500, 145],
      [800, 145],
    ]);
  });

  it("keeps a proper baseline about the polygon width", () => {
    const stored = [
      [500, 140],
      [800, 140],
    ] as [number, number][];
    const line = makeLine({
      baseline: stored,
      mask: [
        [500, 100],
        [800, 100],
        [800, 160],
        [500, 160],
      ],
    });
    expect(baselinePoints(line)).toEqual(stored);
  });

  it("keeps a positive-length baseline when there is no polygon", () => {
    const stored = [
      [0, 0],
      [3, 4],
    ] as [number, number][];
    expect(baselinePoints(makeLine({ baseline: stored }))).toEqual(stored);
  });

  it("caps the font size by polygon height", () => {
    const line = makeLine({
      baseline: [
        [0, 15],
        [50, 15],
      ],
      mask: [
        [0, 0],
        [100, 0],
        [100, 20],
        [0, 20],
      ],
    });
    expect(baselinePoints(line)).toEqual([
      [0, 15],
      [50, 15],
    ]);
    expect(lineFontSize(line, 1)).toBe(18);
  });

  it("ignores blank ground truth in favour of model text", () => {
    const line = makeLine({
      line_transcriptions: [
        transcription("ground_truth", "   ", "gt-1"),
        transcription("model", "model words", "model-1"),
      ],
    });
    expect(displayText(line, null)).toEqual({
      text: "model words",
      source: "model",
    });
  });
});
