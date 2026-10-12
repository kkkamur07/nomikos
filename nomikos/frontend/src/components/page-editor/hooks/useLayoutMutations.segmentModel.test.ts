import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useLayoutMutations } from "./useLayoutMutations";

const segmentPart = vi.fn();
const listPartLines = vi.fn();
const getPartLayout = vi.fn();
const getPagePairing = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    segmentPart: (...args: unknown[]) => segmentPart(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
  },
}));

function setup(selectedSegmentModelId: string | null) {
  return renderHook(() =>
    useLayoutMutations({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      layout: { blocks: [], lines: [] },
      setLayout: vi.fn(),
      lines: [],
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
      selectedSegmentModelId,
      trackJobAndWait: vi.fn().mockResolvedValue({ status: "done" }),
    }),
  );
}

describe("useLayoutMutations runAutoSegment model", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    segmentPart.mockResolvedValue({ job_id: "job-1" });
    listPartLines.mockResolvedValue([]);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
  });

  it("sends an empty body when the catalog is empty and no segment model is chosen", async () => {
    const view = setup(null);

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      {},
    );
  });

  it("sends model_id when a segment model is chosen", async () => {
    const view = setup("seg-b");

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(segmentPart).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      { model_id: "seg-b" },
    );
  });
});
