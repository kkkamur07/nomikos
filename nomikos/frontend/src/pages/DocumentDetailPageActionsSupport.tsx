import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { api, type DocumentWithPartsResponse } from "../api/client";
import * as session from "../auth/session";
import { DocumentDetailPage } from "./DocumentDetailPage";

/** Three pages, one of them checked by a human: the numbers the menus quote. */
export const DOCUMENT: DocumentWithPartsResponse = {
  id: "doc-1",
  project_id: "project-1",
  name: "Chapter 4",
  workflow: "draft",
  created_at: "2026-06-16T10:00:00Z",
  updated_at: "2026-06-16T10:00:00Z",
  part_count: 3,
  parts: [0, 1, 2].map((order) => ({
    id: `part-${order + 1}`,
    document_id: "doc-1",
    order,
    image_url: `/media/parts/part-${order + 1}`,
    width: 800,
    height: 1000,
    reviewed: order === 0,
    published: true,
    created_at: "2026-06-16T10:00:00Z",
  })),
};

const COUNTS = { total: 3, reviewed: 1, unsegmented: 2, unpaired: 3 };

/** The anchor `saveBlobAsFile` builds, captured instead of navigated to. */
export const lastDownloadRef: {
  current: { href: string; download: string } | null;
} = { current: null };

export function renderDocumentPage() {
  window.history.replaceState({}, "", "/projects/project-1/documents/doc-1");
  return render(<DocumentDetailPage />);
}

export async function openMenu(triggerName: string, menuName: string) {
  fireEvent.click(screen.getByRole("button", { name: triggerName }));
  return await screen.findByRole("menu", { name: menuName });
}

export function seedProject(ownerId: string) {
  vi.mocked(api.getProject).mockResolvedValue({
    id: "project-1",
    name: "East Syriac",
    slug: "east-syriac",
    owner_id: ownerId,
    guidelines: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    document_count: 1,
  });
}

/** One job as a batch route announces it: id plus the host fixed for it. */
function queuedJob(jobId: string) {
  return {
    job_id: jobId,
    execution_target: "cloud" as const,
    preferred_execution_target: "cloud" as const,
    execution_target_substituted: false,
  };
}

export function seedActionMocks() {
  vi.clearAllMocks();
  lastDownloadRef.current = null;
  vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
  vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
    this: HTMLAnchorElement,
  ) {
    lastDownloadRef.current = { href: this.href, download: this.download };
  });
  Object.assign(URL, {
    createObjectURL: vi.fn(() => "blob:document-export"),
    revokeObjectURL: vi.fn(),
  });

  vi.mocked(api.me).mockResolvedValue({
    id: "user-1",
    email: "dev@example.com",
    username: "dev",
    is_admin: false,
    prefer_local_inference: false,
    created_at: "2026-01-01T00:00:00Z",
  });
  seedProject("user-1");
  vi.mocked(api.getDocument).mockResolvedValue(DOCUMENT);
  vi.mocked(api.getDocumentWorkflowCounts).mockResolvedValue(COUNTS);
  vi.mocked(api.listInferenceModels).mockResolvedValue([]);
  vi.mocked(api.listProjectModelBindings).mockResolvedValue([]);
  vi.mocked(api.enqueueDocumentSegment).mockResolvedValue({
    jobs: [queuedJob("job-1")],
    queued: 3,
    skipped: 0,
  });
  vi.mocked(api.enqueueDocumentTranscribe).mockResolvedValue({
    jobs: [queuedJob("job-2")],
    queued: 3,
    skipped: 0,
  });
  vi.mocked(api.exportDocumentPageXml).mockResolvedValue(
    new Blob(["PK"], { type: "application/zip" }),
  );
  vi.mocked(api.exportDocumentTranscriptionPdf).mockResolvedValue(
    new Blob(["%PDF"], { type: "application/pdf" }),
  );
  vi.mocked(api.exportDocumentText).mockResolvedValue(
    new Blob(["text"], { type: "text/plain" }),
  );
  vi.mocked(api.exportDocumentDocx).mockResolvedValue(
    new Blob(["PK"], {
      type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }),
  );
}
