/**
 * The pairing read that follows a committed write is not part of the write.
 *
 * Undo, redo and delete used to read the pairing inside the write's own
 * `try`, so a failed read rolled back a write the server had already taken:
 * a deleted segment came back on screen as a ghost, the edit sat on both
 * stacks, and a second undo created the segment on the server twice.
 */
import { act, renderHook } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  api as Api,
  LineResponse,
  PartLayoutResponse,
} from "../../../api/client";
import { useLayoutMutations } from "./useLayoutMutations";

const api = vi.hoisted(() => ({
  createPartLine: vi.fn<typeof Api.createPartLine>(),
  deletePartLine: vi.fn<typeof Api.deletePartLine>(),
  patchPartLine: vi.fn<typeof Api.patchPartLine>(),
  getPagePairing: vi.fn<typeof Api.getPagePairing>(),
}));

vi.mock("../../../api/client", () => ({ api }));

const CREATED_AT = "2026-06-16T10:00:00Z";

function segment(id: string, x: number): LineResponse {
  return {
    id,
    part_id: "part-1",
    block_id: null,
    order: 0,
    kind: "polygon",
    points: [
      [x, 0],
      [x + 9, 0],
      [x + 9, 9],
    ],
    baseline: null,
    mask: null,
    source: "machine",
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: false,
    created_at: CREATED_AT,
    line_transcriptions: [],
  };
}

/** Renders the hook over real `lines` state, so the screen copy is visible. */
function setup(initialLines: LineResponse[], selectedSegmentId: string | null) {
  const setLineError = vi.fn();
  const setPairingError = vi.fn();
  const view = renderHook(() => {
    const [lines, setLines] = useState(initialLines);
    const [layout, setLayout] = useState<PartLayoutResponse>({
      blocks: [],
      lines: [],
    });
    const mutations = useLayoutMutations({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      layout,
      setLayout,
      lines,
      setLines,
      setLineError,
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError,
      selectedSegmentId,
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait: vi.fn(),
    });
    return { lines, ...mutations };
  });
  return { view, setLineError, setPairingError };
}

const ids = (lines: LineResponse[]) => lines.map((line) => line.id);

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(window, "confirm").mockReturnValue(true);
  api.deletePartLine.mockResolvedValue(undefined);
  api.createPartLine.mockImplementation(async () => segment("S2", 0));
  api.getPagePairing.mockRejectedValue(new Error("pairing down"));
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useLayoutMutations when only the pairing read fails", () => {
  it("keeps a delete the server took, and reports the pairing failure", async () => {
    const { view, setLineError, setPairingError } = setup(
      [segment("S", 0)],
      "S",
    );

    await act(async () => {
      await view.result.current.deleteSelectedSegment();
    });

    expect(api.deletePartLine).toHaveBeenCalledTimes(1);
    expect(ids(view.result.current.lines)).toEqual([]);
    expect(setLineError).toHaveBeenLastCalledWith(null);
    expect(setPairingError).toHaveBeenLastCalledWith("pairing down");
    expect(view.result.current.canUndo).toBe(true);
    expect(view.result.current.canRedo).toBe(false);
  });

  it("moves an undo to the redo stack once, and creates the segment once", async () => {
    const { view } = setup([segment("S", 0)], "S");
    await act(async () => {
      await view.result.current.deleteSelectedSegment();
    });

    await act(async () => {
      await view.result.current.undoEdit();
    });
    await act(async () => {
      await view.result.current.undoEdit();
    });

    expect(api.createPartLine).toHaveBeenCalledTimes(1);
    expect(ids(view.result.current.lines)).toEqual(["S2"]);
    expect(view.result.current.canUndo).toBe(false);
    expect(view.result.current.canRedo).toBe(true);
  });

  it("keeps a redo the server took", async () => {
    const { view } = setup([segment("S", 0)], "S");
    await act(async () => {
      await view.result.current.deleteSelectedSegment();
    });
    await act(async () => {
      await view.result.current.undoEdit();
    });

    await act(async () => {
      await view.result.current.redoEdit();
    });

    expect(api.deletePartLine).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-1",
      "S2",
    );
    expect(ids(view.result.current.lines)).toEqual([]);
    expect(view.result.current.canUndo).toBe(true);
    expect(view.result.current.canRedo).toBe(false);
  });

  it("still rolls back a delete the server refused", async () => {
    api.deletePartLine.mockRejectedValueOnce(new Error("delete refused"));
    const { view, setLineError } = setup([segment("S", 0)], "S");

    await act(async () => {
      await view.result.current.deleteSelectedSegment();
    });

    expect(ids(view.result.current.lines)).toEqual(["S"]);
    expect(setLineError).toHaveBeenLastCalledWith("delete refused");
    expect(view.result.current.canUndo).toBe(false);
  });
});

describe("useLayoutMutations two commits in one render", () => {
  it("builds the second optimistic step on the first", async () => {
    api.patchPartLine.mockReturnValue(new Promise(() => undefined));
    const { view } = setup([segment("A", 0), segment("B", 100)], null);
    const movedA: [number, number][] = [
      [1, 1],
      [20, 1],
      [20, 20],
    ];
    const movedB: [number, number][] = [
      [101, 1],
      [120, 1],
      [120, 20],
    ];

    act(() => {
      void view.result.current.updateSegmentPoints("A", movedA);
      void view.result.current.updateSegmentPoints("B", movedB);
    });

    const [a, b] = view.result.current.lines;
    expect(a.points).toEqual(movedA);
    expect(b.points).toEqual(movedB);
  });
});
