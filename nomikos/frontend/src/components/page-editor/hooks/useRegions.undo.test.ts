/**
 * Region writes share the Segment undo stack (ADR 0011): Ctrl+Z after
 * "Delete region" brings the region back instead of undoing an older Segment
 * edit, and a failed region write is reported where it can be seen.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLayoutMutations } from "./useLayoutMutations";
import { useRegions } from "./useRegions";

const createPartBlock = vi.fn();
const deletePartBlock = vi.fn();
const patchPartLine = vi.fn();
const getPartLayout = vi.fn();
const listPartLines = vi.fn();
const toastError = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    createPartBlock: (...args: unknown[]) => createPartBlock(...args),
    deletePartBlock: (...args: unknown[]) => deletePartBlock(...args),
    patchPartLine: (...args: unknown[]) => patchPartLine(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
  },
}));

vi.mock("../../ui/toast", () => ({
  toast: { error: (message: string) => toastError(message), success: vi.fn() },
}));

const BOX = {
  points: [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
  ],
};

function segment(id: string, blockId: string | null) {
  return { id, order: 0, block_id: blockId, points: [], kind: "rectangle" };
}

function render(blocks: object[], lines: object[]) {
  const ids = {
    projectId: "project-1",
    documentId: "document-1",
    partId: "part-1",
  };
  const { result } = renderHook(() => {
    const layout = { blocks, lines: [] } as never;
    const mutations = useLayoutMutations({
      ...ids,
      layout,
      setLayout: vi.fn(),
      lines: lines as never,
      setLines: vi.fn(),
      setLineError: vi.fn(),
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError: vi.fn(),
      selectedSegmentId: null,
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait: vi.fn(),
    } as never);
    const regions = useRegions({
      ...ids,
      layout,
      setLayout: vi.fn(),
      lines: lines as never,
      setLines: vi.fn(),
      onRecordEdit: mutations.recordRegionEdit,
    });
    return { mutations, regions };
  });
  return () => result.current;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
  listPartLines.mockResolvedValue([segment("a", null), segment("b", null)]);
  deletePartBlock.mockResolvedValue(undefined);
  patchPartLine.mockResolvedValue({});
});

describe("undo of a region create", () => {
  it("deletes the region, and redo draws it again", async () => {
    createPartBlock
      .mockResolvedValueOnce({ id: "region-1", box: BOX })
      .mockResolvedValueOnce({ id: "region-2", box: BOX });
    const hook = render([], [segment("a", null)]);

    await act(async () => {
      await hook().regions.createRegion(BOX.points as never);
    });
    await act(async () => {
      await hook().mutations.undoEdit();
    });
    expect(deletePartBlock).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      "region-1",
    );

    await act(async () => {
      await hook().mutations.redoEdit();
    });
    expect(createPartBlock).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-1",
      { box: BOX, label: null, assign_contained: true },
    );
  });
});

describe("undo of a region delete", () => {
  it("recreates the region with its box and label and moves its Segments back", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    createPartBlock.mockResolvedValue({ id: "region-9", box: BOX });
    const hook = render(
      [{ id: "region-1", order: 0, box: BOX, label: "4v" }],
      [segment("a", "region-1"), segment("b", "region-1"), segment("c", null)],
    );

    await act(async () => {
      await hook().regions.deleteRegion("region-1");
    });
    expect(deletePartBlock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await hook().mutations.undoEdit();
    });
    expect(createPartBlock).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      { box: BOX, label: "4v", assign_contained: false },
    );
    expect(patchPartLine.mock.calls.map((call) => [call[3], call[4]])).toEqual([
      ["a", { block_id: "region-9" }],
      ["b", { block_id: "region-9" }],
    ]);

    await act(async () => {
      await hook().mutations.redoEdit();
    });
    expect(deletePartBlock).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-1",
      "region-9",
    );
  });

  it("asks first when the region holds Segments, and keeps it on No", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const hook = render(
      [{ id: "region-1", order: 0, box: BOX, label: null }],
      [segment("a", "region-1"), segment("b", "region-1")],
    );

    await act(async () => {
      await hook().regions.deleteRegion("region-1");
    });
    expect(confirm.mock.calls[0][0]).toMatch(
      /2 segments stay on the page and lose their region/,
    );
    expect(deletePartBlock).not.toHaveBeenCalled();
  });
});

describe("a failed region create", () => {
  it("shows the server's message as a toast", async () => {
    const { ApiError } = await import("../../../api/errors");
    createPartBlock.mockRejectedValue(new ApiError("region boom", 500));
    const hook = render([], []);

    await act(async () => {
      await hook().regions.createRegion(BOX.points as never);
    });
    expect(toastError).toHaveBeenCalledWith("region boom");
    expect(hook().regions.regionError).toBe("region boom");
  });
});
