import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

vi.mock("../../../context/BackgroundJobsContext", () => ({
  useBackgroundJobs: () => ({ subscribeToJobCompletion: () => () => {} }),
}));

const DOCUMENT = {
  id: "document-1",
  project_id: "project-1",
  parts: [
    { id: "part-1", order: 0 },
    { id: "part-2", order: 1 },
  ],
};

const DOCUMENT_TWO = {
  id: "document-2",
  project_id: "project-1",
  parts: [{ id: "part-3", order: 0 }],
};

const EMPTY_PAIRING = {
  text_lines: [],
  pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
};

const TRANSCRIBE_MODEL = { id: "htr-1", task: "transcribe", name: "htr" };
const SEGMENT_A = {
  id: "seg-a",
  task: "segment",
  name: "seg-v1",
  artifact_ref: "registry://blla-segment?tag=stable",
};
const SEGMENT_B = {
  id: "seg-b",
  task: "segment",
  name: "seg-v2",
  artifact_ref: "registry://ppocr-segment?tag=stable",
};

describe("usePageEditorData segment models", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([
      TRANSCRIBE_MODEL,
      SEGMENT_A,
      SEGMENT_B,
      { id: "bin-1", task: "binarize", name: "bin" },
    ]);
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        return Promise.reject(new Error("no binding"));
      },
    );
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("filters segment models from the single catalog call", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.segmentModels.map((model) => model.id)).toEqual([
      "seg-a",
      "seg-b",
    ]);
    expect(result.current.transcribeModels.map((model) => model.id)).toEqual([
      "htr-1",
    ]);
    expect(resolvePartModelBinding).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      "segment",
    );
  });

  it("preselects the canonical row when the segment binding rejects, without an error", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-a");
    expect(result.current.error).toBeNull();
    // The HTR picker still falls back to the first catalog row.
    expect(result.current.selectedTranscribeModelId).toBe("htr-1");
  });

  it("preselects by artifact_ref when the canonical row is not first", async () => {
    listInferenceModels.mockResolvedValue([
      TRANSCRIBE_MODEL,
      SEGMENT_B,
      SEGMENT_A,
    ]);
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.segmentModels.map((model) => model.id)).toEqual([
      "seg-b",
      "seg-a",
    ]);
    expect(result.current.selectedSegmentModelId).toBe("seg-a");
  });

  it("falls back to the first model when no row carries the canonical registry id", async () => {
    listInferenceModels.mockResolvedValue([TRANSCRIBE_MODEL, SEGMENT_B]);
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
  });

  it("leaves no model selected when the segment catalog is empty", async () => {
    listInferenceModels.mockResolvedValue([TRANSCRIBE_MODEL]);
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.segmentModels).toEqual([]);
    expect(result.current.selectedSegmentModelId).toBeNull();
  });

  it("preselects the resolved segment binding", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "segment") return Promise.resolve({ model: SEGMENT_B });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
  });

  it("keeps the explicit choice across a page turn without refetching the catalog", async () => {
    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    act(() => {
      result.current.setSelectedSegmentModelId("seg-b");
    });
    expect(result.current.selectedSegmentModelId).toBe("seg-b");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
    expect(listInferenceModels).toHaveBeenCalledTimes(1);
    expect(result.current.segmentModels.map((model) => model.id)).toEqual([
      "seg-a",
      "seg-b",
    ]);
  });

  it("keeps an explicit choice when the next page is bound to another model", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        if (task === "segment" && partId === "part-2")
          return Promise.resolve({ model: SEGMENT_B });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    act(() => {
      result.current.setSelectedSegmentModelId("seg-a");
    });
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-a");
  });

  it("follows the binding on a page turn without an explicit choice", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        if (task === "segment" && partId === "part-2")
          return Promise.resolve({ model: SEGMENT_B });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
  });

  it("resets an explicit choice on a document-level load to the new binding", async () => {
    getDocument.mockImplementation((projectId: string, documentId: string) =>
      Promise.resolve(documentId === "document-2" ? DOCUMENT_TWO : DOCUMENT),
    );
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        if (task === "segment" && partId === "part-3")
          return Promise.resolve({ model: SEGMENT_B });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result, rerender } = renderHook(
      ({ documentId, partId }: { documentId: string; partId: string }) =>
        usePageEditorData("project-1", documentId, partId),
      { initialProps: { documentId: "document-1", partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.setSelectedSegmentModelId("seg-a");
    });
    expect(result.current.selectedSegmentModelId).toBe("seg-a");

    rerender({ documentId: "document-2", partId: "part-3" });
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-b");
  });

  it("resets an explicit choice on a document-level load to the canonical row when unbound", async () => {
    getDocument.mockImplementation((projectId: string, documentId: string) =>
      Promise.resolve(documentId === "document-2" ? DOCUMENT_TWO : DOCUMENT),
    );
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: TRANSCRIBE_MODEL });
        return Promise.reject(new Error("no binding"));
      },
    );

    const { result, rerender } = renderHook(
      ({ documentId, partId }: { documentId: string; partId: string }) =>
        usePageEditorData("project-1", documentId, partId),
      { initialProps: { documentId: "document-1", partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.setSelectedSegmentModelId("seg-b");
    });
    expect(result.current.selectedSegmentModelId).toBe("seg-b");

    rerender({ documentId: "document-2", partId: "part-3" });
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedSegmentModelId).toBe("seg-a");
  });
});
