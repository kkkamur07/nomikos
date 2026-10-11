/**
 * What survives the deletion of the local-first write path (#60).
 *
 * Most of this file used to be about the loopback transport: a local run, its
 * three abort causes, and the fallback from the helper to the cloud. There is
 * one path now, so those cases are gone rather than rewritten - a cloud-only
 * "fallback" test would be asserting a decision nothing makes any more.
 *
 * What is left is the pair of rules that were never about the transport: a
 * refused submission is an explanation the researcher can act on, and a failure
 * of the reload *after* a stored segmentation must not be mistaken for a
 * failure of the segmentation.
 */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../../api/errors";
import { queryClient, taggedMeta } from "../../../api/queryClient";
import { resourceTags } from "../../../api/resources";
import { platformNoCapacityMessage } from "../../../inference/platformMessages";
import { useLayoutMutations } from "./useLayoutMutations";

const segmentPart = vi.fn();
const listPartLines = vi.fn();
const getPartLayout = vi.fn();
const getPagePairing = vi.fn();
const createPartLine = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    segmentPart: (...args: unknown[]) => segmentPart(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
    createPartLine: (...args: unknown[]) => createPartLine(...args),
  },
}));

function setup() {
  const setPairingError = vi.fn();
  const setSubmissionRefusal = vi.fn();
  const trackJobAndWait = vi.fn().mockResolvedValue({ status: "done" });
  // Captures the listener so a test can fire a completion that this hook did
  // not start, which is the case the background refresh introduced.
  const jobCompletionListeners: ((event: unknown) => void)[] = [];
  const subscribeToJobCompletion = vi.fn(
    (listener: (event: unknown) => void) => {
      jobCompletionListeners.push(listener);
      return () => {
        const at = jobCompletionListeners.indexOf(listener);
        if (at >= 0) jobCompletionListeners.splice(at, 1);
      };
    },
  );

  const view = renderHook(() =>
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
      setPairingError,
      selectedSegmentId: null,
      setSelectedSegmentId: vi.fn(),
      setApprovedTextDraft: vi.fn(),
      onDrawComplete: vi.fn(),
      setSubmissionRefusal,
      trackJobAndWait,
      subscribeToJobCompletion,
    }),
  );

  return {
    view,
    setPairingError,
    setSubmissionRefusal,
    trackJobAndWait,
    jobCompletionListeners,
  };
}

/**
 * A cached read of the published page, as `PublicDocumentPage` holds it. The
 * editor never renders it, which is why nothing here used to notice it had gone
 * stale.
 */
const PUBLIC_DOCUMENT_KEY = ["public-document", "project-1", "document-1"];

async function seedPublishedPageRead() {
  await queryClient.fetchQuery({
    queryKey: PUBLIC_DOCUMENT_KEY,
    queryFn: () => Promise.resolve({ name: "before the edit" }),
    meta: taggedMeta([resourceTags.publicDocument("project-1", "document-1")]),
  });
}

function publishedPageReadIsStale(): boolean {
  return queryClient.getQueryState(PUBLIC_DOCUMENT_KEY)?.isInvalidated ?? false;
}

describe("useLayoutMutations auto segment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    segmentPart.mockResolvedValue({ job_id: "cloud-job-1" });
    listPartLines.mockResolvedValue([]);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
  });

  it("keeps a finished segmentation when the reload that follows it fails", async () => {
    // A blip on the cosmetic reload, after the segmentation is already stored.
    listPartLines.mockRejectedValue(new Error("network hiccup"));
    const { view, setPairingError } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // Segmenting again would replace Segments that are already saved.
    expect(segmentPart).toHaveBeenCalledTimes(1);
    expect(setPairingError).toHaveBeenCalledWith("network hiccup");
  });

  it("does not name a host in the message it reports", async () => {
    const { view } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // The job announces its **execution target**; a second sentence here, from
    // a second source, is how the two come to disagree.
    expect(view.result.current.segmentMessage?.text).not.toMatch(
      /locally|in the cloud|on your computer/i,
    );
  });

  it("explains a refused submission instead of reporting a generic failure", async () => {
    segmentPart.mockRejectedValueOnce(
      new ApiError(platformNoCapacityMessage(), 409),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(setSubmissionRefusal).toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
    expect(setPairingError).not.toHaveBeenCalledWith(
      platformNoCapacityMessage(),
    );
  });

  it("makes the published page stale once the segmentation is stored", async () => {
    await seedPublishedPageRead();
    const { view } = setup();
    expect(publishedPageReadIsStale()).toBe(false);

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // The reader's copy of this document now has different lines in it.
    expect(publishedPageReadIsStale()).toBe(true);
  });

  it("leaves the published page alone when the submission is refused", async () => {
    segmentPart.mockRejectedValueOnce(
      new ApiError(platformNoCapacityMessage(), 409),
    );
    await seedPublishedPageRead();
    const { view } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // Nothing was written, so nothing is stale.
    expect(publishedPageReadIsStale()).toBe(false);
  });

  it("stays quiet when the researcher cancels the job", async () => {
    segmentPart.mockRejectedValueOnce(
      new DOMException("The operation was aborted.", "AbortError"),
    );
    const { view, setPairingError, setSubmissionRefusal } = setup();

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    // The jobs panel already reported it.
    expect(setPairingError).not.toHaveBeenCalledWith(expect.any(String));
    expect(setSubmissionRefusal).not.toHaveBeenCalledWith(expect.any(String));
  });

  it("clears the undo and redo stacks once the segmentation reload replaces the Segments", async () => {
    // The segmentation already replaced every Segment on the server by the
    // time the reload runs. A stack entry from before it names a line id the
    // reload just made up: an undo would pop it, applyCanvasEditInverse would
    // silently no-op against the new lines, and the paired patchPartLine or
    // deletePartLine call would 404.
    createPartLine.mockResolvedValue({
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "manual",
      manual_geometry: false,
      line_transcriptions: [],
    });
    const { view } = setup();

    await act(async () => {
      await view.result.current.replaceWithManualLine("rectangle", [
        [0, 0],
        [10, 0],
        [10, 10],
      ]);
    });
    expect(view.result.current.canUndo).toBe(true);
    const revisionBeforeSegmentation = view.result.current.editUndoRevision;

    await act(async () => {
      await view.result.current.runAutoSegment();
    });

    expect(view.result.current.canUndo).toBe(false);
    expect(view.result.current.canRedo).toBe(false);
    expect(view.result.current.editUndoRevision).toBeGreaterThan(
      revisionBeforeSegmentation,
    );
  });

  it("clears the undo stack when a job this hook did not start replaces the Segments", async () => {
    // The background refresh in usePageEditorData reloads the same lines from a
    // sibling hook, so a job that finished while the tab was in the background
    // leaves this stack naming line ids that refresh has already replaced.
    createPartLine.mockResolvedValue({
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "manual",
      manual_geometry: false,
      line_transcriptions: [],
    });
    const { view, jobCompletionListeners } = setup();

    await act(async () => {
      await view.result.current.replaceWithManualLine("rectangle", [
        [0, 0],
        [10, 0],
        [10, 10],
      ]);
    });
    expect(view.result.current.canUndo).toBe(true);
    const revisionBefore = view.result.current.editUndoRevision;

    act(() => {
      for (const listener of jobCompletionListeners) {
        listener({
          jobId: "job-9",
          kind: "segmentation",
          documentPartId: "part-1",
          status: "done",
        });
      }
    });

    expect(view.result.current.canUndo).toBe(false);
    expect(view.result.current.canRedo).toBe(false);
    expect(view.result.current.editUndoRevision).toBeGreaterThan(
      revisionBefore,
    );
  });

  it("leaves the undo stack alone for a job on another page", async () => {
    createPartLine.mockResolvedValue({
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "manual",
      manual_geometry: false,
      line_transcriptions: [],
    });
    const { view, jobCompletionListeners } = setup();

    await act(async () => {
      await view.result.current.replaceWithManualLine("rectangle", [
        [0, 0],
        [10, 0],
        [10, 10],
      ]);
    });

    act(() => {
      for (const listener of jobCompletionListeners) {
        listener({
          jobId: "job-9",
          kind: "segmentation",
          documentPartId: "part-2",
          status: "done",
        });
      }
    });

    expect(view.result.current.canUndo).toBe(true);
  });

  it("leaves the undo stack alone for a job that does not touch geometry", async () => {
    createPartLine.mockResolvedValue({
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "manual",
      manual_geometry: false,
      line_transcriptions: [],
    });
    const { view, jobCompletionListeners } = setup();

    await act(async () => {
      await view.result.current.replaceWithManualLine("rectangle", [
        [0, 0],
        [10, 0],
        [10, 10],
      ]);
    });

    // Transcription writes text onto lines that are already there and keeps
    // their ids, and a segmentation that failed wrote nothing at all. Neither
    // invalidates the entry above, so running OCR must not silently cost the
    // researcher the geometry edit they just made.
    act(() => {
      for (const listener of jobCompletionListeners) {
        listener({
          jobId: "job-10",
          kind: "transcription-page",
          documentPartId: "part-1",
          status: "done",
        });
        listener({
          jobId: "job-11",
          kind: "transcription-segment",
          documentPartId: "part-1",
          status: "done",
        });
        listener({
          jobId: "job-12",
          kind: "segmentation",
          documentPartId: "part-1",
          status: "failed",
        });
      }
    });

    expect(view.result.current.canUndo).toBe(true);
  });
});

describe("useLayoutMutations drawn Segment numbering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  function segment(id: string, order: number) {
    return {
      id,
      order,
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

  it("lets the server place a drawn Segment and takes the orders it shifted", async () => {
    // Drawn between the first and second Segment: the server numbers it 1 and
    // moves the two below it down, returning only the new row.
    createPartLine.mockResolvedValue(segment("new", 1));
    listPartLines.mockResolvedValue([
      segment("a", 0),
      segment("new", 1),
      segment("b", 2),
      segment("c", 3),
    ]);
    getPagePairing.mockResolvedValue({
      text_lines: [],
      pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
    });
    const setLines = vi.fn();
    const { result } = renderHook(
      () =>
        useLayoutMutations({
          projectId: "project-1",
          documentId: "document-1",
          partId: "part-1",
          layout: { blocks: [], lines: [] },
          setLayout: vi.fn(),
          lines: [segment("a", 0), segment("b", 1), segment("c", 2)],
          setLines,
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
        }) as never,
    );

    await act(async () => {
      await (
        result.current as ReturnType<typeof useLayoutMutations>
      ).replaceWithManualLine("rectangle", [
        [0, 6],
        [10, 6],
        [10, 8],
      ]);
    });

    const body = createPartLine.mock.calls[0][3];
    expect(body).not.toHaveProperty("order");
    const finalLines = setLines.mock.calls.at(-1)?.[0] as {
      id: string;
      order: number;
    }[];
    expect(
      Object.fromEntries(finalLines.map((line) => [line.id, line.order])),
    ).toEqual({ a: 0, new: 1, b: 2, c: 3 });
  });
});
