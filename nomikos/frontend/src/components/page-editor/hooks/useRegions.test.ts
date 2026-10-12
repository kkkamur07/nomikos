/**
 * A drawn region is ranked by the server from where it sits on the Page
 * (ADR 0011), not numbered after the regions drawn before it.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useRegions } from "./useRegions";

const createPartBlock = vi.fn();
const getPartLayout = vi.fn();
const listPartLines = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    createPartBlock: (...args: unknown[]) => createPartBlock(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
  },
}));

describe("drawing a region", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createPartBlock.mockResolvedValue({ id: "region-2" });
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
  });

  it("sends no order, so the server ranks it by position", async () => {
    const { result } = renderHook(() =>
      useRegions({
        projectId: "project-1",
        documentId: "document-1",
        partId: "part-1",
        layout: {
          blocks: [{ id: "region-1", order: 0, box: { points: [] } }],
          lines: [],
        },
        setLayout: vi.fn(),
        setLines: vi.fn(),
      } as never),
    );

    await act(async () => {
      await result.current.createRegion([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
      ]);
    });

    const body = createPartBlock.mock.calls[0][3];
    expect(body).not.toHaveProperty("order");
    expect(body).toMatchObject({ assign_contained: true });
  });
});
