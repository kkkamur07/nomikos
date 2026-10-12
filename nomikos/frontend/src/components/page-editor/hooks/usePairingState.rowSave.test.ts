/**
 * A transcript row's save: it names the text it was edited from, reports a
 * failure only in the row (no page banner), and the page picks up text saved
 * elsewhere when the tab comes back into view.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../../api/client";
import { usePairingState } from "./usePairingState";

const updateGroundTruthLineText = vi.fn();
const getPagePairing = vi.fn();
const listPartLines = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    updateGroundTruthLineText: (...args: unknown[]) =>
      updateGroundTruthLineText(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
  },
}));

function lineWith(text: string, points = [[0, 0]]): LineResponse {
  return {
    id: "line-1",
    order: 0,
    points,
    line_transcriptions: [
      {
        id: "line-1-gt",
        transcription_id: "ground-truth-1",
        transcription_kind: "ground_truth",
        text,
        confidence: null,
      },
    ],
  } as unknown as LineResponse;
}

function setup() {
  const setPairingError = vi.fn();
  const setLines = vi.fn();
  const view = renderHook(() =>
    usePairingState({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      lines: [lineWith("old")],
      setLines,
      transcriptionLayers: [],
      setTranscriptionLayers: vi.fn(),
      selectedTranscriptionLayerId: null,
      setSelectedTranscriptionLayerId: vi.fn(),
      groundTruthTranscriptionId: "ground-truth-1",
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError,
      selectedTranscribeModelId: null,
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait: vi.fn(),
    }),
  );
  return { view, setPairingError, setLines };
}

/** Applies the updater the hook passed to setLines to ``current``. */
function applied(setLines: ReturnType<typeof vi.fn>, current: LineResponse[]) {
  const update = setLines.mock.calls.at(-1)?.[0] as (
    lines: LineResponse[],
  ) => LineResponse[];
  return update(current);
}

beforeEach(() => {
  vi.clearAllMocks();
  getPagePairing.mockResolvedValue({
    text_lines: [],
    pairing_progress: { paired_lines: 0, total_lines: 1, percent: 0 },
  });
});

afterEach(() => {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: "visible",
  });
});

describe("usePairingState row saves", () => {
  it("sends previous_text when the row names the text it started from", async () => {
    updateGroundTruthLineText.mockResolvedValue({ text: "new" });
    const { view } = setup();

    await act(() => view.result.current.commitLineText("line-1", "new", "old"));

    expect(updateGroundTruthLineText).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "ground-truth-1",
      "line-1",
      { text: "new", previous_text: "old" },
    );
  });

  it("leaves a failed row save to the row: no page banner, no toast", async () => {
    updateGroundTruthLineText.mockRejectedValue(new Error("boom"));
    const { view, setPairingError } = setup();

    await act(async () => {
      await expect(
        view.result.current.commitLineText("line-1", "new", "old"),
      ).rejects.toThrow("boom");
    });

    expect(setPairingError).not.toHaveBeenCalledWith("boom");
  });

  it("still reports a failed strip save on the page", async () => {
    updateGroundTruthLineText.mockRejectedValue(new Error("boom"));
    const { view, setPairingError } = setup();
    act(() => view.result.current.setSelectedSegmentId("line-1"));

    let saved = true;
    await act(async () => {
      saved = await view.result.current.saveApprovedText();
    });

    expect(saved).toBe(false);
    expect(setPairingError).toHaveBeenCalledWith("boom");
  });

  it("refetches line texts when the tab becomes visible, keeping geometry", async () => {
    listPartLines.mockResolvedValue([lineWith("FROM_A", [[9, 9]])]);
    const { setLines } = setup();

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(listPartLines).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
    );
    const [merged] = applied(setLines, [lineWith("old")]);
    expect(merged.line_transcriptions[0].text).toBe("FROM_A");
    expect(merged.points).toEqual([[0, 0]]);
  });

  it("does not refetch when the tab is hidden", async () => {
    setup();
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(listPartLines).not.toHaveBeenCalled();
  });

  it("reloads one row's text, or answers null when the line is gone", async () => {
    listPartLines.mockResolvedValueOnce([lineWith("FROM_A")]);
    const { view } = setup();

    let fresh: LineResponse | null = null;
    await act(async () => {
      fresh = await view.result.current.reloadLineText("line-1");
    });
    expect(fresh).not.toBeNull();

    listPartLines.mockResolvedValueOnce([]);
    await act(async () => {
      fresh = await view.result.current.reloadLineText("line-1");
    });
    expect(fresh).toBeNull();
  });
});
