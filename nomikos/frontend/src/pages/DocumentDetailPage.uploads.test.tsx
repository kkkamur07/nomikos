import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import { toast } from "../components/ui/toast";
import { prepareDirectUpload } from "../utils/encodePartImage";
import {
  renderDocumentPage,
  seedActionMocks,
} from "./DocumentDetailPageActionsSupport";

vi.mock("../components/ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}));
vi.mock("../components/AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));
vi.mock("../components/document/JobsNotice", () => ({
  JobsNotice: () => <div>Jobs panel</div>,
}));
vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getDocument: vi.fn(),
      me: vi.fn(),
      getProject: vi.fn(),
      updateDocument: vi.fn(),
      deleteDocument: vi.fn(),
      updatePartReviewStatus: vi.fn(),
      updatePartsPublished: vi.fn(),
      getDocumentWorkflowCounts: vi.fn(),
      listInferenceModels: vi.fn(),
      enqueueDocumentSegment: vi.fn(),
      enqueueDocumentTranscribe: vi.fn(),
      exportDocumentPageXml: vi.fn(),
      exportDocumentTranscriptionPdf: vi.fn(),
      exportDocumentText: vi.fn(),
      rotateDocumentShareToken: vi.fn(),
      beginPartUpload: vi.fn(),
      finalizePartUpload: vi.fn(),
      uploadPart: vi.fn(),
      deletePart: vi.fn(),
    },
  };
});
vi.mock("../utils/encodePartImage", () => ({
  prepareDirectUpload: vi.fn(),
}));

function seedDirectUpload() {
  vi.mocked(prepareDirectUpload).mockResolvedValue({
    blob: new Blob(["bytes"], { type: "image/png" }),
    contentType: "image/png",
    filename: "page.png",
  });
  vi.mocked(api.beginPartUpload).mockResolvedValue({
    part_id: "part-9",
    image_key: "minted-key",
    upload_url: "https://storage.example/upload",
    token: "token",
  });
  vi.mocked(api.uploadPart).mockResolvedValue({} as never);
}

async function pickFile(): Promise<void> {
  const { container } = renderDocumentPage();
  await screen.findByRole("heading", { name: "Chapter 4" });
  const input = container.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  fireEvent.change(input, {
    target: {
      files: [new File(["bytes"], "page.png", { type: "image/png" })],
    },
  });
}

describe("DocumentDetailPage direct-upload fallback", () => {
  beforeEach(seedActionMocks);
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("deletes the pending part before falling back to multipart", async () => {
    seedDirectUpload();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("blocked")));
    await pickFile();

    // Without the delete, `begin`'s pending row would sit in the document
    // as a broken page until the media GC sweeps it after the presign
    // window. The file still uploads through the multipart route.
    await waitFor(() => {
      expect(api.deletePart).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-9",
      );
    });
    expect(api.uploadPart).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.success)).toHaveBeenCalledWith("Part uploaded");
  });

  it("does not retry finalize on a 422", async () => {
    seedDirectUpload();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    vi.mocked(api.finalizePartUpload).mockRejectedValue(
      new ApiError("bad seal", 422),
    );
    await pickFile();

    // A 4xx answers the same way twice; the error stands after one attempt.
    await waitFor(() => {
      expect(api.finalizePartUpload).toHaveBeenCalledTimes(1);
    });
    expect(vi.mocked(toast.error)).toHaveBeenCalledWith("bad seal");
  });

  it("retries finalize once on a 500", async () => {
    seedDirectUpload();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    vi.mocked(api.finalizePartUpload)
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValue({} as never);
    await pickFile();

    await waitFor(() => {
      expect(api.finalizePartUpload).toHaveBeenCalledTimes(2);
    });
    expect(vi.mocked(toast.success)).toHaveBeenCalledWith("Part uploaded");
  });
});
