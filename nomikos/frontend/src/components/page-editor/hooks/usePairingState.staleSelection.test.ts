/**
 * A write that resolves after the researcher opened another segment.
 *
 * OCR can run for half an hour and a pairing request takes a round trip, and
 * the researcher keeps working meanwhile. Whatever lands late must not put the
 * old segment's text into the draft box of the segment open now: "Save" would
 * then write it over that segment's Ground truth.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  api as Api,
  JobResponse,
  LineResponse,
  PagePairingResponse,
  TranscriptionLayerResponse,
} from "../../../api/client";
import type { TrackJobAndWait } from "./editorHookTypes";
import { usePairingState } from "./usePairingState";

const api = vi.hoisted(() => ({
  enqueueTranscribePart: vi.fn<typeof Api.enqueueTranscribePart>(),
  listPartLines: vi.fn<typeof Api.listPartLines>(),
  listTranscriptions: vi.fn<typeof Api.listTranscriptions>(),
  getPagePairing: vi.fn<typeof Api.getPagePairing>(),
  pairTextLine: vi.fn<typeof Api.pairTextLine>(),
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

function segment(id: string, order: number, text: string): LineResponse {
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
    line_transcriptions: [
      {
        id: `t-${id}`,
        transcription_id: "gt",
        transcription_kind: "ground_truth",
        text,
        confidence: null,
      },
    ],
  };
}

const LINES = [segment("A", 0, "text of A"), segment("B", 1, "text of B")];

function job(overrides: Partial<JobResponse>): JobResponse {
  return {
    id: "job-1",
    type: "transcribe",
    status: "done",
    payload: {},
    result: null,
    error: null,
    document_id: "document-1",
    document_part_id: "part-1",
    user_id: null,
    execution_target: "cloud",
    execution_target_substituted: false,
    preferred_execution_target: "cloud",
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

function pairing(
  textLines: PagePairingResponse["text_lines"],
): PagePairingResponse {
  return {
    text_lines: textLines,
    pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
  };
}

/** A promise the test resolves by hand, so the selection can move first. */
function deferred<T>() {
  let settle!: (value: T) => void;
  const promise = new Promise<T>((resolve) => {
    settle = resolve;
  });
  return { promise, resolve: settle };
}

function setup(trackJobAndWait: TrackJobAndWait = vi.fn<TrackJobAndWait>()) {
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
      selectedTranscribeModelId: "model-1",
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait,
    }),
  );
  return { view, setPairingError };
}

beforeEach(() => {
  vi.clearAllMocks();
  api.enqueueTranscribePart.mockResolvedValue({
    job_id: "job-1",
    execution_target: "cloud",
    execution_target_substituted: false,
    preferred_execution_target: "cloud",
  });
  api.listPartLines.mockResolvedValue(LINES);
  api.listTranscriptions.mockResolvedValue([GROUND_TRUTH]);
  api.getPagePairing.mockResolvedValue(pairing([]));
});

describe("usePairingState after a selection change", () => {
  it("keeps the draft of the segment opened while OCR was running", async () => {
    const finished = deferred<JobResponse>();
    const { view } = setup(vi.fn<TrackJobAndWait>(() => finished.promise));
    api.updateGroundTruthLineText.mockImplementation(
      async (_p, _d, _t, _lineId, body) => ({
        id: "t-B",
        transcription_id: "gt",
        transcription_kind: "ground_truth",
        text: body.text,
        confidence: null,
      }),
    );

    act(() => view.result.current.selectSegment("A"));
    let run!: Promise<void>;
    act(() => {
      run = view.result.current.runPageOcr();
    });
    await act(async () => {
      await Promise.resolve();
    });
    act(() => view.result.current.selectSegment("B"));
    await act(async () => {
      finished.resolve(
        job({ result: { transcription_id: "model-layer", lines: [] } }),
      );
      await run;
    });

    expect(view.result.current.selectedSegmentId).toBe("B");
    expect(view.result.current.approvedTextDraft).toBe("text of B");
    await act(async () => {
      await view.result.current.saveGroundTruthText();
    });
    expect(api.updateGroundTruthLineText).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "gt",
      "B",
      { text: "text of B" },
    );
  });

  it("still refreshes the draft when the same segment stayed open", async () => {
    const { view } = setup(
      vi.fn<TrackJobAndWait>(async () =>
        job({ result: { transcription_id: "model-layer", lines: [] } }),
      ),
    );
    api.listPartLines.mockResolvedValue([
      segment("A", 0, "text of A, reloaded"),
      LINES[1],
    ]);

    act(() => view.result.current.selectSegment("A"));
    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(view.result.current.approvedTextDraft).toBe("text of A, reloaded");
  });

  it("does not put a late pairing into the draft of another segment", async () => {
    const paired = deferred<PagePairingResponse>();
    api.pairTextLine.mockReturnValue(paired.promise);
    const { view } = setup();

    act(() => view.result.current.selectSegment("A"));
    let pending!: Promise<void>;
    act(() => {
      pending = view.result.current.pairTextLine(3);
    });
    act(() => view.result.current.selectSegment("B"));
    await act(async () => {
      paired.resolve(
        pairing([{ order: 3, text: "paired for A", paired_line_id: "A" }]),
      );
      await pending;
    });

    expect(view.result.current.selectedSegmentId).toBe("B");
    expect(view.result.current.approvedTextDraft).toBe("text of B");
    expect(api.pairTextLine).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      { line_id: "A", text_line_order: 3 },
    );
  });

  it("puts the pairing into the draft when the segment is still open", async () => {
    api.pairTextLine.mockResolvedValue(
      pairing([{ order: 3, text: "paired for A", paired_line_id: "A" }]),
    );
    const { view } = setup();

    act(() => view.result.current.selectSegment("A"));
    await act(async () => {
      await view.result.current.pairTextLine(3);
    });

    expect(view.result.current.approvedTextDraft).toBe("paired for A");
  });
});

describe("usePairingState when the job is cancelled", () => {
  it("says it was cancelled and reports no completion", async () => {
    const { view, setPairingError } = setup(
      vi.fn<TrackJobAndWait>(async () => job({ status: "cancelled" })),
    );

    await act(async () => {
      await view.result.current.runPageOcr();
    });

    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
    expect(view.result.current.ocrMessage?.text).toBe("Job cancelled");
    expect(view.result.current.ocrRunning).toBe(false);
    expect(api.listPartLines).not.toHaveBeenCalled();
  });
});

describe("usePairingState overlapping OCR runs", () => {
  it("stays busy until the last run finishes", async () => {
    const first = deferred<JobResponse>();
    const second = deferred<JobResponse>();
    const track = vi
      .fn<TrackJobAndWait>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const { view } = setup(track);

    act(() => view.result.current.selectSegment("A"));
    let runs!: Promise<void>[];
    act(() => {
      runs = [
        view.result.current.runPageOcr(),
        view.result.current.runSegmentOcr(),
      ];
    });
    await act(async () => {
      first.resolve(job({ result: { transcription_id: "m", lines: [] } }));
      await runs[0];
    });

    expect(view.result.current.ocrRunning).toBe(true);
    expect(view.result.current.ocrScope).toBe("segment");

    await act(async () => {
      second.resolve(job({ result: { transcription_id: "m", lines: [] } }));
      await runs[1];
    });

    expect(view.result.current.ocrRunning).toBe(false);
    expect(view.result.current.ocrScope).toBeNull();
  });
});
