import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import {
  lastDownloadRef,
  openMenu,
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
      listProjectModelBindings: vi.fn().mockResolvedValue([]),
      createProjectModelBinding: vi.fn(),
      updateProjectModelBinding: vi.fn(),
      listProjectJobsPage: vi
        .fn()
        .mockResolvedValue({ items: [], next_cursor: null }),
      enqueueDocumentSegment: vi.fn(),
      enqueueDocumentTranscribe: vi.fn(),
      exportDocumentPageXml: vi.fn(),
      exportDocumentTranscriptionPdf: vi.fn(),
      exportDocumentText: vi.fn(),
      exportDocumentDocx: vi.fn(),
      rotateDocumentShareToken: vi.fn(),
    },
  };
});

describe("DocumentDetailPage downloads", () => {
  beforeEach(seedActionMocks);

  it("lists both download scopes with their counts", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");

    expect(within(menu).getByText("Whole chapter")).toBeTruthy();
    expect(within(menu).getByText("Reviewed only")).toBeTruthy();
    expect(within(menu).getByText("zip, 3")).toBeTruthy();
    expect(within(menu).getByText("zip, 1")).toBeTruthy();
    expect(within(menu).getByText("3 pages")).toBeTruthy();
    expect(within(menu).getByText("1 page")).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", { name: /Plain text/ }),
    ).toBeTruthy();
    // 3 pages, 1 of them checked, so 2 are about to be left out.
    expect(
      within(menu).getByText("Skips the 2 pages nobody has checked."),
    ).toBeTruthy();
  });

  it("downloads the reviewed-only archive through the authenticated client", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "PAGE XML + images, reviewed pages only",
      }),
    );

    // A plain link cannot reach these routes: a browser-initiated request
    // carries no Authorization header. The bytes come back through the client
    // and only then become a file.
    await waitFor(() => {
      expect(api.exportDocumentPageXml).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        true,
      );
    });
    await waitFor(() => {
      expect(lastDownloadRef.current?.download).toBe("Chapter_4_reviewed.zip");
    });
  });

  it("names the whole-chapter text export without the reviewed marker", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Download", "Download this document");
    fireEvent.click(within(menu).getByRole("menuitem", { name: /Plain text/ }));

    await waitFor(() => {
      expect(api.exportDocumentText).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        false,
      );
    });
    await waitFor(() => {
      expect(lastDownloadRef.current?.download).toBe("Chapter_4.txt");
    });
  });

  it("downloads the Word export for either scope through the authenticated client", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    let menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "Word (.docx), whole chapter",
      }),
    );
    await waitFor(() => {
      expect(api.exportDocumentDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        false,
      );
    });
    await waitFor(() => {
      expect(lastDownloadRef.current?.download).toBe("Chapter_4.docx");
    });

    menu = await openMenu("Download", "Download this document");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: "Word (.docx), reviewed pages only",
      }),
    );
    await waitFor(() => {
      expect(api.exportDocumentDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        true,
      );
    });
    await waitFor(() => {
      expect(lastDownloadRef.current?.download).toBe("Chapter_4_reviewed.docx");
    });
  });
});
