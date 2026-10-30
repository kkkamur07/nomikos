/**
 * What the panel's data layer refuses to keep.
 *
 * The report is a measurement of the page as it stands, so almost every
 * assertion here is about it being *discarded*: after an apply, after a failed
 * apply, and when the page or its lines change. Closing the panel alone keeps
 * it, so the Workflow menu can show the finding count. A stale report is worse than none, because
 * every row in it is a button offering a fix the server will refuse.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../../api/client";
import { useSegmentHealth } from "./useSegmentHealth";
import type { api } from "../../../api/client";

const getSegmentHealth = vi.fn();
const splitSpanningSegment = vi.fn();
const deleteSegmentSuspect = vi.fn();
const mergeSegmentFragment = vi.fn();
const trimSegmentOverlap = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    getSegmentHealth: (...args: Parameters<typeof api.getSegmentHealth>) =>
      getSegmentHealth(...args),
    splitSpanningSegment: (
      ...args: Parameters<typeof api.splitSpanningSegment>
    ) => splitSpanningSegment(...args),
    mergeSegmentFragment: (
      ...args: Parameters<typeof api.mergeSegmentFragment>
    ) => mergeSegmentFragment(...args),
    trimSegmentOverlap: (...args: Parameters<typeof api.trimSegmentOverlap>) =>
      trimSegmentOverlap(...args),
    deleteSegmentSuspect: (
      ...args: Parameters<typeof api.deleteSegmentSuspect>
    ) => deleteSegmentSuspect(...args),
  },
}));

const SPECK = "33333333-3333-4333-8333-333333333333";

function report(findings = 0) {
  return {
    part_id: "part-1",
    page_width: 2479,
    page_height: 3508,
    measured_page: true,
    line_count: 40,
    considered_count: 40,
    finding_count: findings,
    suspects: [],
    spanning: [],
    fragments: [],
    overlaps: [],
  };
}

const LINES: LineResponse[] = [];

type Props = { isOpen: boolean; partId?: string; lines?: LineResponse[] };

function setup(open = true) {
  const setLines = vi.fn();
  const view = renderHook(
    ({ isOpen, partId = "part-1", lines = LINES }: Props) =>
      useSegmentHealth({
        projectId: "project-1",
        documentId: "document-1",
        partId,
        open: isOpen,
        lines,
        setLines,
      }),
    { initialProps: { isOpen: open } as Props },
  );
  return { ...view, setLines };
}

describe("useSegmentHealth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getSegmentHealth.mockResolvedValue(report());
    splitSpanningSegment.mockResolvedValue([]);
    deleteSegmentSuspect.mockResolvedValue([]);
    mergeSegmentFragment.mockResolvedValue([]);
    trimSegmentOverlap.mockResolvedValue([]);
  });

  it("reads nothing until the panel is opened", async () => {
    const { rerender } = setup(false);
    expect(getSegmentHealth).not.toHaveBeenCalled();

    rerender({ isOpen: true });

    await waitFor(() => expect(getSegmentHealth).toHaveBeenCalledTimes(1));
  });

  it("re-reads the page after a fix is applied", async () => {
    splitSpanningSegment.mockResolvedValue([{ id: "line-1" }]);
    const { result, setLines } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());

    act(() => result.current.apply({ kind: "split", lineId: "line-1" }));

    await waitFor(() =>
      expect(setLines).toHaveBeenCalledWith([{ id: "line-1" }]),
    );
    await waitFor(() => expect(getSegmentHealth).toHaveBeenCalledTimes(2));
  });

  it("re-reads the page after a fix is refused, not just after one that works", async () => {
    // A refusal usually means the page moved under the panel, so the findings
    // on screen are the ones least worth keeping.
    deleteSegmentSuspect.mockRejectedValue(new Error("no longer offered"));
    const { result } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());

    act(() => result.current.apply({ kind: "delete", lineId: SPECK }));

    await waitFor(() => expect(result.current.error).toBe("no longer offered"));
    await waitFor(() => expect(getSegmentHealth).toHaveBeenCalledTimes(2));
    expect(result.current.pending).toBeNull();
  });

  it("keeps the report after the panel closes, for the menu's count", async () => {
    getSegmentHealth.mockResolvedValue(report(7));
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());

    rerender({ isOpen: false });

    expect(result.current.report?.finding_count).toBe(7);
  });

  it("forgets a kept report once the lines change", async () => {
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());
    rerender({ isOpen: false });
    expect(result.current.report).not.toBeNull();

    rerender({ isOpen: false, lines: [{ id: "line-9" } as LineResponse] });

    expect(result.current.report).toBeNull();
  });

  it("forgets a kept report on a page turn", async () => {
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());
    rerender({ isOpen: false });

    rerender({ isOpen: false, partId: "part-2" });

    expect(result.current.report).toBeNull();
  });

  it("reads the page again when the panel reopens", async () => {
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());
    rerender({ isOpen: false });

    rerender({ isOpen: true });

    await waitFor(() => expect(getSegmentHealth).toHaveBeenCalledTimes(2));
  });

  it("spins the row belonging to the finding, not the page", async () => {
    let release: (
      value: Awaited<ReturnType<typeof api.splitSpanningSegment>>,
    ) => void = () => {};
    splitSpanningSegment.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    const { result } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());

    act(() => result.current.apply({ kind: "split", lineId: "line-7" }));
    await waitFor(() => expect(result.current.pending).toBe("line-7"));

    await act(async () => {
      release([]);
    });
    await waitFor(() => expect(result.current.pending).toBeNull());
  });

  it("keys a merge's spinner to the fragment, which is the row with the button", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.report).not.toBeNull());

    act(() =>
      result.current.apply({
        kind: "merge",
        primaryId: "primary-1",
        fragmentId: "fragment-1",
      }),
    );

    expect(result.current.pending).toBe("fragment-1");
  });
});
