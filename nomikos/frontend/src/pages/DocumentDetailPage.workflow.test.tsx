import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import {
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
      enqueueDocumentSegment: vi.fn(),
      enqueueDocumentTranscribe: vi.fn(),
      exportDocumentPageXml: vi.fn(),
      exportDocumentTranscriptionPdf: vi.fn(),
      exportDocumentText: vi.fn(),
      rotateDocumentShareToken: vi.fn(),
    },
  };
});

describe("DocumentDetailPage workflow menu", () => {
  beforeEach(seedActionMocks);

  it("opens the Workflow menu with both segment scopes, the model captions and the warning", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");

    // The counts come from `workflow-counts`, not from the parts list: the
    // parts list has no idea which pages have lines or a pairing.
    expect(api.getDocumentWorkflowCounts).toHaveBeenCalledWith(
      "project-1",
      "doc-1",
    );
    expect(
      within(menu).getByRole("menuitem", {
        name: /Segment unsegmented pages\s*2/,
      }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", { name: /Re-segment every page\s*3/ }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", {
        name: /Transcribe unpaired pages\s*3/,
      }),
    ).toBeTruthy();
    expect(within(menu).getByText("Segment")).toBeTruthy();
    expect(within(menu).getByText("Transcribe")).toBeTruthy();
    // The segment engine caption became a picker: Default plus the catalog.
    expect(
      within(menu).getByRole("combobox", { name: "Segmentation model" }),
    ).toBeTruthy();
    expect(within(menu).getByRole("option", { name: "Default" })).toBeTruthy();
    expect(within(menu).getByText(/Model/)).toHaveTextContent(
      "Model blla-greek-v2",
    );
    expect(
      within(menu).getByText(/Re-segmenting discards unapproved machine text/),
    ).toBeTruthy();
  });

  it("queues the safe segment scope straight from the menu", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");
    fireEvent.click(
      within(menu).getByRole("menuitem", {
        name: /Segment unsegmented pages/,
      }),
    );

    await waitFor(() => {
      expect(api.enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        { scope: "unsegmented", model_id: null },
      );
    });
  });

  it("does nothing when Re-segment every page is chosen, until it is confirmed", async () => {
    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });

    const menu = await openMenu("Workflow", "Document workflow");
    fireEvent.click(
      within(menu).getByRole("menuitem", { name: /Re-segment every page/ }),
    );

    // The whole point of the confirm: the click that reaches the destructive
    // item must not be the click that discards the transcriptions.
    expect(api.enqueueDocumentSegment).not.toHaveBeenCalled();
    expect(
      screen.getByText(/the model's unapproved text on it is discarded/),
    ).toBeTruthy();
    expect(screen.getByText("Re-segment every page (3 pages)?")).toBeTruthy();

    // Backing out leaves the document exactly as it was.
    fireEvent.click(screen.getByRole("menuitem", { name: "Cancel" }));
    expect(api.enqueueDocumentSegment).not.toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("menuitem", { name: /Re-segment every page/ }),
    );
    fireEvent.click(
      screen.getByRole("menuitem", { name: /Yes, re-segment 3 pages/ }),
    );

    await waitFor(() => {
      expect(api.enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        { scope: "all", model_id: null },
      );
    });
  });
});
