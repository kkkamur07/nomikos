import { act, renderHook } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  LineResponse,
  PartLayoutResponse,
  api,
} from "../../../api/client";
import { useLayoutMutations } from "./useLayoutMutations";

const createPartLine = vi.fn();
const patchPartLine = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    createPartLine: (...args: Parameters<typeof api.createPartLine>) =>
      createPartLine(...args),
    patchPartLine: (...args: Parameters<typeof api.patchPartLine>) =>
      patchPartLine(...args),
  },
}));

vi.mock("../../../api/imageCache", () => ({
  fetchPartImage: async () => new Blob(["page-image"], { type: "image/png" }),
}));

function line(id: string, order: number, x: number): LineResponse {
  return {
    id,
    order,
    kind: "rectangle",
    points: [
      [x, 0],
      [x + 10, 0],
      [x + 10, 10],
    ],
    source: "machine",
    part_id: "part-1",
    block_id: null,
    baseline: null,
    mask: null,
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: false,
    created_at: "2026-06-16T10:00:00Z",
    line_transcriptions: [],
  };
}

/** A promise the test resolves by hand, so an edit can land mid-request. */
function deferred<T>() {
  let settle: (value: T) => void;
  const promise = new Promise<T>((resolve) => {
    settle = resolve;
  });
  return { promise, resolve: (value: T) => settle(value) };
}

/**
 * Renders the hook against real `lines` state, so a write-back that was
 * computed from a stale render is visible in the state the editor would show.
 */
function setup(initialLines: LineResponse[]) {
  return renderHook(() => {
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
      setLineError: vi.fn(),
      setTextLines: vi.fn(),
      setPairingProgress: vi.fn(),
      setPairingError: vi.fn(),
      selectedSegmentId: null,
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      partImageUrl: null,
      shouldUseLocalPath: () => false,
      setSubmissionRefusal: vi.fn(),
      segmentRegistryModelId: "blla-segment",
      trackJobAndWait: vi.fn(),
    });
    return { lines, setLines, ...mutations };
  });
}

describe("useLayoutMutations concurrent edits", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps a concurrent edit when a segment update resolves", async () => {
    const saved = deferred<LineResponse>();
    patchPartLine.mockReturnValue(saved.promise);
    const view = setup([line("line-1", 0, 0)]);

    let pending: Promise<void>;
    await act(async () => {
      pending = view.result.current.updateSegmentPoints("line-1", [
        [5, 5],
        [15, 5],
        [15, 15],
      ]);
    });

    // A second segment arrives while the first request is still in flight.
    await act(async () => {
      view.result.current.setLines((current) => [
        ...current,
        line("line-2", 1, 100),
      ]);
    });

    await act(async () => {
      saved.resolve(line("line-1", 0, 5));
      await pending;
    });

    expect(view.result.current.lines.map((entry) => entry.id)).toEqual([
      "line-1",
      "line-2",
    ]);
  });

  it("keeps a concurrent edit when a drawn segment is saved", async () => {
    const saved = deferred<LineResponse>();
    createPartLine.mockReturnValue(saved.promise);
    const view = setup([line("line-1", 0, 0)]);

    let pending: Promise<void>;
    await act(async () => {
      pending = view.result.current.replaceWithManualLine("rectangle", [
        [20, 20],
        [30, 20],
        [30, 30],
      ]);
    });

    await act(async () => {
      view.result.current.setLines((current) => [
        ...current,
        line("line-2", 1, 100),
      ]);
    });

    await act(async () => {
      saved.resolve(line("line-3", 2, 20));
      await pending;
    });

    expect(view.result.current.lines.map((entry) => entry.id)).toEqual([
      "line-1",
      "line-2",
      "line-3",
    ]);
  });
});
