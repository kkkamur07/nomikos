import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { clearAccessToken, setAccessToken } from "../../../auth/storage";
import { usePageEditorData } from "./usePageEditorData";

const getDocument = vi.fn();
const getPartLayout = vi.fn();
const listPartLines = vi.fn();
const listTranscriptions = vi.fn();
const getPagePairing = vi.fn();
const listInferenceModels = vi.fn();
const resolvePartModelBinding = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    getDocument: (...args: unknown[]) => getDocument(...args),
    getPartLayout: (...args: unknown[]) => getPartLayout(...args),
    listPartLines: (...args: unknown[]) => listPartLines(...args),
    listTranscriptions: (...args: unknown[]) => listTranscriptions(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    resolvePartModelBinding: (...args: unknown[]) =>
      resolvePartModelBinding(...args),
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

const EMPTY_PAIRING = {
  text_lines: [],
  pairing_progress: { paired_lines: 0, total_lines: 0, percent: 0 },
};

const HTR_FIRST = { id: "htr-1", task: "transcribe", name: "htr" };
const HTR_SYRIAC = { id: "htr-2", task: "transcribe", name: "syriac" };
const SEGMENT_A = {
  id: "seg-a",
  task: "segment",
  name: "kraken",
  artifact_ref: "registry://blla-segment?tag=stable",
};

describe("usePageEditorData transcribe models", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setAccessToken("test-token");
    getDocument.mockResolvedValue(DOCUMENT);
    getPartLayout.mockResolvedValue({ blocks: [], lines: [] });
    listPartLines.mockResolvedValue([]);
    listTranscriptions.mockResolvedValue([]);
    getPagePairing.mockResolvedValue(EMPTY_PAIRING);
    listInferenceModels.mockResolvedValue([HTR_FIRST, HTR_SYRIAC, SEGMENT_A]);
    resolvePartModelBinding.mockRejectedValue(new Error("no binding"));
  });

  afterEach(() => {
    clearAccessToken();
  });

  it("falls back to the first catalog row when nothing is bound", async () => {
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedTranscribeModelId).toBe("htr-1");
  });

  it("prefers the resolved binding over the first catalog row", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "transcribe")
          return Promise.resolve({ model: HTR_SYRIAC });
        return Promise.reject(new Error("no binding"));
      },
    );
    const { result } = renderHook(() =>
      usePageEditorData("project-1", "document-1", "part-1"),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.selectedTranscribeModelId).toBe("htr-2");
  });

  it("keeps an explicit choice across a page turn bound to another model", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        _partId: string,
        task: string,
      ) => {
        if (task === "transcribe") return Promise.resolve({ model: HTR_FIRST });
        return Promise.reject(new Error("no binding"));
      },
    );
    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedTranscribeModelId).toBe("htr-1");

    act(() => {
      result.current.setSelectedTranscribeModelId("htr-2");
    });
    expect(result.current.selectedTranscribeModelId).toBe("htr-2");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedTranscribeModelId).toBe("htr-2");
    expect(listInferenceModels).toHaveBeenCalledTimes(1);
  });

  it("follows the binding on a page turn without an explicit choice", async () => {
    resolvePartModelBinding.mockImplementation(
      (
        _projectId: string,
        _documentId: string,
        partId: string,
        task: string,
      ) => {
        if (task === "transcribe" && partId === "part-2")
          return Promise.resolve({ model: HTR_SYRIAC });
        return Promise.reject(new Error("no binding"));
      },
    );
    const { result, rerender } = renderHook(
      ({ partId }: { partId: string }) =>
        usePageEditorData("project-1", "document-1", partId),
      { initialProps: { partId: "part-1" } },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.selectedTranscribeModelId).toBe("htr-1");

    rerender({ partId: "part-2" });
    await waitFor(() => expect(result.current.partLoading).toBe(false));

    expect(result.current.selectedTranscribeModelId).toBe("htr-2");
  });
});
