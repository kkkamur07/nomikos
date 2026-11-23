/**
 * What the strip's edit box opens with, and how its one save keeps the link
 * to the model output it corrected.
 *
 * A model-only line opens holding the model's reading, so the researcher fixes
 * a few characters instead of retyping the line. Saving that correction
 * copies the reading into Ground truth first (which records the model layer it
 * came from) and writes the correction over it.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  api as Api,
  LineResponse,
  LineTranscriptionResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";
import { usePairingState } from "./usePairingState";

const api = vi.hoisted(() => ({
  listPartLines: vi.fn<typeof Api.listPartLines>(),
  getPagePairing: vi.fn<typeof Api.getPagePairing>(),
  copyToGroundTruth: vi.fn<typeof Api.copyToGroundTruth>(),
  updateGroundTruthLineText: vi.fn<typeof Api.updateGroundTruthLineText>(),
}));

vi.mock("../../../api/client", () => ({ api }));

const CREATED_AT = "2026-06-16T10:00:00Z";

const GROUND_TRUTH: TranscriptionLayerResponse = {
  id: "gt",
  document_id: "document-1",
  name: "Ground truth",
  kind: "ground_truth",
  created_by_job_id: null,
  created_at: CREATED_AT,
};

function row(
  layerId: string,
  kind: "ground_truth" | "model",
  text: string,
): LineTranscriptionResponse {
  return {
    id: `${layerId}-row`,
    transcription_id: layerId,
    transcription_kind: kind,
    text,
    confidence: null,
  };
}

function segment(
  id: string,
  order: number,
  ...rows: LineTranscriptionResponse[]
): LineResponse {
  return {
    id,
    part_id: "part-1",
    block_id: null,
    order,
    kind: "rectangle",
    points: [
      [0, 0],
      [10, 0],
      [10, 10],
    ],
    baseline: null,
    mask: null,
    source: "machine",
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: false,
    created_at: CREATED_AT,
    line_transcriptions: rows,
  };
}

const LINES = [
  segment(
    "A",
    0,
    row("gt", "ground_truth", "checked A"),
    row("m1", "model", "read A"),
  ),
  segment("B", 1, row("m1", "model", "old B"), row("m2", "model", "read B")),
  segment("C", 2),
];

function setup() {
  const setPairingError = vi.fn();
  const view = renderHook(() =>
    usePairingState({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      lines: LINES,
      setLines: vi.fn(),
      transcriptionLayers: [GROUND_TRUTH],
      setTranscriptionLayers: vi.fn(),
      selectedTranscriptionLayerId: "gt",
      setSelectedTranscriptionLayerId: vi.fn(),
      groundTruthTranscriptionId: "gt",
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError,
      selectedTranscribeModelId: null,
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait: vi.fn(),
    }),
  );
  return { view, setPairingError };
}

beforeEach(() => {
  vi.clearAllMocks();
  api.listPartLines.mockResolvedValue(LINES);
  api.getPagePairing.mockResolvedValue({
    text_lines: [],
    pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
  });
  api.copyToGroundTruth.mockResolvedValue({ line_ids: ["B"] } as never);
  api.updateGroundTruthLineText.mockImplementation(
    async (_p, _d, _t, lineId, body) => row("gt", "ground_truth", body.text),
  );
});

describe("usePairingState draft seeding", () => {
  it("opens a Ground truth line with its Ground truth", () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("A"));
    expect(view.result.current.approvedTextDraft).toBe("checked A");
  });

  it("opens a model-only line with the newest model reading", () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("B"));
    expect(view.result.current.approvedTextDraft).toBe("read B");
  });

  it("opens a line with no text at all empty", () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("B"));
    act(() => view.result.current.selectSegment("C"));
    expect(view.result.current.approvedTextDraft).toBe("");
  });

  it("seeds the next line on keyboard navigation", () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("A"));
    act(() => view.result.current.navigateSegment(1));
    expect(view.result.current.selectedSegmentId).toBe("B");
    expect(view.result.current.approvedTextDraft).toBe("read B");
  });

  it("seeds the line a merge kept", () => {
    const { view } = setup();
    act(() => view.result.current.selectMergedSurvivor(LINES[1]));
    expect(view.result.current.approvedTextDraft).toBe("read B");
  });
});

describe("usePairingState strip save", () => {
  it("copies the model reading before writing a correction over it", async () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("B"));
    act(() => view.result.current.setApprovedTextDraft("read B, fixed"));

    let saved = false;
    await act(async () => {
      saved = await view.result.current.saveSelectedSegmentText();
    });

    expect(saved).toBe(true);
    expect(api.copyToGroundTruth).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "m2",
      { line_ids: ["B"] },
    );
    expect(api.updateGroundTruthLineText).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "gt",
      "B",
      { text: "read B, fixed", previous_text: "read B" },
    );
    expect(api.copyToGroundTruth.mock.invocationCallOrder[0]).toBeLessThan(
      api.updateGroundTruthLineText.mock.invocationCallOrder[0],
    );
  });

  it("writes an edit of existing Ground truth directly", async () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("A"));
    act(() => view.result.current.setApprovedTextDraft("checked A again"));

    await act(async () => {
      await view.result.current.saveSelectedSegmentText();
    });

    expect(api.copyToGroundTruth).not.toHaveBeenCalled();
    expect(api.updateGroundTruthLineText).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "gt",
      "A",
      { text: "checked A again" },
    );
  });

  it("approves the model reading the box was seeded with", async () => {
    const { view } = setup();
    act(() => view.result.current.selectSegment("B"));

    let approved = false;
    await act(async () => {
      approved =
        await view.result.current.promoteSelectedSegmentToGroundTruth();
    });

    expect(approved).toBe(true);
    expect(api.copyToGroundTruth).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "m2",
      { line_ids: ["B"] },
    );
    expect(api.updateGroundTruthLineText).not.toHaveBeenCalled();
  });
});
