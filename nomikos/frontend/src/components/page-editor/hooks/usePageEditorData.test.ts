/**
 * The gap this closes: `runAutoSegment` and `runSegmentOcr`/`runPageOcr`
 * already reload the page after their job finishes, but that reload is the
 * continuation of one promise held by the one component instance whose
 * button was clicked. If that continuation never runs against a live
 * instance - the tab was backgrounded and its timers throttled, the
 * component remounted, or the researcher navigated away and back mid-job -
 * nothing else re-syncs, and only a hard reload recovers.
 *
 * `usePageEditorData` closes it from the other side: it listens for any job
 * finishing for the part it currently has open, independent of who started
 * that job or whether their own promise continuation is still around.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
import { ApiError } from "../../../api/errors";
import { clearAccessToken, setAccessToken } from "../../../auth/storage";
import { usePageEditorData } from "./usePageEditorData";
import type { api } from "../../../api/client";

const getDocument = vi.fn();
const getPartLayout = vi.fn();
const listPartLines = vi.fn();
const listTranscriptions = vi.fn();
const getPagePairing = vi.fn();
const listInferenceModels = vi.fn();
const resolvePartModelBinding = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    getDocument: (...args: Parameters<typeof api.getDocument>) =>
      getDocument(...args),
    getPartLayout: (...args: Parameters<typeof api.getPartLayout>) =>
      getPartLayout(...args),
    listPartLines: (...args: Parameters<typeof api.listPartLines>) =>
      listPartLines(...args),
    listTranscriptions: (...args: Parameters<typeof api.listTranscriptions>) =>
      listTranscriptions(...args),
    getPagePairing: (...args: Parameters<typeof api.getPagePairing>) =>
      getPagePairing(...args),
    listInferenceModels: (
      ...args: Parameters<typeof api.listInferenceModels>
    ) => listInferenceModels(...args),
    resolvePartModelBinding: (
      ...args: Parameters<typeof api.resolvePartModelBinding>
    ) => resolvePartModelBinding(...args),
  },
}));

let completionListener: ((event: JobCompletionEvent) => void) | null = null;
const subscribeToJobCompletion = vi.fn(
  (listener: (event: JobCompletionEvent) => void) => {
    completionListener = listener;
    return () => {
      if (completionListener === listener) completionListener = null;
    };
  },
);

vi.mock("../../../context/BackgroundJobsContext", () => ({
  useBackgroundJobs: () => ({ subscribeToJobCompletion }),
}));

const DOCUMENT = {
  id: "document-1",
  project_id: "project-1",
  parts: [
    { id: "part-1", order: 0 },
    { id: "part-2", order: 1 },
  ],
};

const EMPTY_PAIRING = {
  text_lines: [],
  pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
};

function announce(event: JobCompletionEvent) {
  completionListener?.(event);
}

describe("usePageEditorData job-completion refresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    completionListener = null;
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([]);
    resolvePartModelBinding.mockRejectedValue(new Error("no binding"));
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("re-fetches the open part's content when a job finishes for it, and ignores a job for a different part", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(listPartLines).toHaveBeenCalledTimes(1);
    expect(listPartLines).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-1",
    );
    expect(subscribeToJobCompletion).toHaveBeenCalledTimes(1);

    // A segmentation that finished for the *other* part on this document is
    // not this mounted instance's business.
    await act(async () => {
      announce({
        jobId: "job-other-part",
        kind: "segmentation",
        documentPartId: "part-2",
        status: "done",
      });
    });
    expect(listPartLines).toHaveBeenCalledTimes(1);

    // A job for the part actually open re-syncs it - this is the whole point:
    // no promise continuation from whoever started the job had to run.
    const newLine = {
      id: "line-1",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "machine",
      manual_geometry: false,
      line_transcriptions: [],
    };
    listPartLines.mockResolvedValueOnce([newLine]);

    await act(async () => {
      announce({
        jobId: "job-this-part",
        kind: "segmentation",
        documentPartId: "part-1",
        status: "done",
      });
    });

    await waitFor(() => expect(result.current.lines).toEqual([newLine]));
    expect(listPartLines).toHaveBeenCalledTimes(2);
  });

  it("ignores a job announcement for a failed or cancelled run", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(listPartLines).toHaveBeenCalledTimes(1);

    await act(async () => {
      announce({
        jobId: "job-failed",
        kind: "transcription-page",
        documentPartId: "part-1",
        status: "failed",
      });
    });

    expect(listPartLines).toHaveBeenCalledTimes(1);
  });

  it("does not let a slow first load overwrite a refresh that already landed", async () => {
    // Opening a page whose job is about to finish runs two reads of the same
    // part at once. The refresh is the newer of the two by definition, so the
    // first load losing the race has to mean it stays quiet: landing its own
    // response afterwards would put the page back on pre-job state, with no
    // second announcement coming to correct it.
    const staleLine = {
      id: "line-stale",
      order: 0,
      kind: "rectangle",
      points: [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      source: "machine",
      manual_geometry: false,
      line_transcriptions: [],
    };
    const freshLine = { ...staleLine, id: "line-fresh" };

    let releaseFirstLoad = () => {};
    listPartLines.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          releaseFirstLoad = () => resolve([staleLine]);
        }),
    );

    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(listPartLines).toHaveBeenCalledTimes(1));

    listPartLines.mockResolvedValueOnce([freshLine]);
    await act(async () => {
      announce({
        jobId: "job-mid-load",
        kind: "segmentation",
        documentPartId: "part-1",
        status: "done",
      });
    });
    await waitFor(() => expect(result.current.lines).toEqual([freshLine]));

    await act(async () => {
      releaseFirstLoad();
      await Promise.resolve();
    });

    expect(result.current.lines).toEqual([freshLine]);
  });
});

describe("usePageEditorData page turn", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    completionListener = null;
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([
      { id: "layer-gt", kind: "ground_truth" },
    ]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([
      { id: "model-1", task: "transcribe" },
    ]);
    resolvePartModelBinding.mockRejectedValue(new Error("no binding"));
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("fetches only the page's own content on a page turn and keeps the document's", async () => {
    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(listTranscriptions).toHaveBeenCalledTimes(1);
    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.selectedTranscriptionLayerId).toBe("layer-gt");
    expect(result.current.selectedTranscribeModelId).toBe("model-1");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    // The page's own content came down again for the new page.
    expect(listPartLines).toHaveBeenCalledTimes(2);
    expect(listPartLines).toHaveBeenLastCalledWith(
      "project-1",
      "document-1",
      "part-2",
    );
    expect(getPagePairing).toHaveBeenCalledTimes(2);
    // One resolve per task per page: transcribe and segment bindings are
    // each bound per part, so a page turn resolves both.
    expect(resolvePartModelBinding).toHaveBeenCalledTimes(4);
    // The document's did not, and what was on screen is still there.
    expect(listTranscriptions).toHaveBeenCalledTimes(1);
    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.transcriptionLayers).toHaveLength(1);
    expect(result.current.selectedTranscriptionLayerId).toBe("layer-gt");
    expect(result.current.transcribeModels).toHaveLength(1);
    expect(result.current.selectedTranscribeModelId).toBe("model-1");
  });
});

describe("usePageEditorData transcriptions read", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([]);
    resolvePartModelBinding.mockRejectedValue(new Error("no binding"));
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("names the transcriptions, not the pairing, when the layers are refused", async () => {
    listTranscriptions.mockRejectedValue(new ApiError("Forbidden", 403));
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.pairingError).toBe(
      "Transcriptions are not available for this page.",
    );
  });

  it("names the transcriptions when the layers fail without a message", async () => {
    listTranscriptions.mockRejectedValue("offline");
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.pairingError).toBe("Failed to load transcriptions.");
  });
});
