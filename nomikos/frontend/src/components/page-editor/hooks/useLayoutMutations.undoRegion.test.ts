/**
 * Undoing a delete restores the Segment into the region it was in (ADR 0011).
 * The server numbers a Page region by region, so a Segment restored without
 * its region would be numbered after every region despite its explicit order.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLayoutMutations } from "./useLayoutMutations";

const createPartLine = vi.fn();
const deletePartLine = vi.fn();
const listPartLines = vi.fn();
const getPagePairing = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    createPartLine: (...args: unknown[]) => createPartLine(...args),
    deletePartLine: (...args: unknown[]) => deletePartLine(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
  },
}));

function segment(id: string, order: number, blockId: string | null) {
  return {
    id,
    order,
    block_id: blockId,
    kind: "rectangle",
    points: [
      [0, order * 10],
      [10, order * 10],
      [10, order * 10 + 5],
    ],
    source: "manual",
    manual_geometry: true,
    line_transcriptions: [],
  };
}

function render(blocks: { id: string }[]) {
  const lines = [segment("a", 0, "region-1"), segment("b", 1, "region-1")];
  return renderHook(() =>
    useLayoutMutations({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      layout: { blocks, lines: [] },
      setLayout: vi.fn(),
      lines,
      setLines: vi.fn(),
      setLineError: vi.fn(),
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError: vi.fn(),
      selectedSegmentId: "b",
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      setSubmissionRefusal: vi.fn(),
      trackJobAndWait: vi.fn(),
    } as never),
  );
}

async function deleteThenUndo(blocks: { id: string }[]) {
  const { result } = render(blocks);
  const hook = () => result.current as ReturnType<typeof useLayoutMutations>;
  await act(async () => {
    await hook().deleteSelectedSegment();
  });
  await act(async () => {
    await hook().undoEdit();
  });
  return createPartLine.mock.calls[0][3];
}

describe("undo of a deleted Segment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deletePartLine.mockResolvedValue(undefined);
    createPartLine.mockResolvedValue(segment("b2", 1, "region-1"));
    listPartLines.mockResolvedValue([
      segment("a", 0, "region-1"),
      segment("b2", 1, "region-1"),
    ]);
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
  });

  it("resends the region the Segment was in", async () => {
    const body = await deleteThenUndo([{ id: "region-1" }]);
    expect(body).toMatchObject({ order: 1, block_id: "region-1" });
  });

  it("leaves the region to the server when that region is gone", async () => {
    const body = await deleteThenUndo([]);
    expect(body).not.toHaveProperty("block_id");
  });
});
