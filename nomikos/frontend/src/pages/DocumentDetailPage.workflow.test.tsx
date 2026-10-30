import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import {
  openMenu,
  renderDocumentPage,
  seedActionMocks,
  seedProject,
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

describe("DocumentDetailPage workflow menu", () => {
  beforeEach(seedActionMocks);

  it("opens the Workflow menu with both segment scopes, the model pickers and the re-segment note", async () => {
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
        name: /Segment unsegmented pages\s*2 pages/,
      }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", {
        name: /Re-segment every page.*3 pages/,
      }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("menuitem", {
        name: /Transcribe unpaired pages\s*3 pages/,
      }),
    ).toBeTruthy();
    expect(within(menu).getByText("Segment")).toBeTruthy();
    expect(within(menu).getByText("Transcribe")).toBeTruthy();
    // Both engine captions became pickers: the catalog rows, or
    // No models while the catalog mock is empty.
    expect(
      within(menu).getByRole("combobox", { name: "Segmentation model" }),
    ).toBeTruthy();
    expect(
      within(menu).getByRole("combobox", { name: "HTR transcription model" }),
    ).toBeTruthy();
    expect(
      within(menu).getAllByRole("option", { name: "No models" }),
    ).toHaveLength(2);
    expect(
      within(menu).getByText(
        "Discards unapproved machine text on untouched lines.",
      ),
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

  it("lets a member who does not own the project set the project default", async () => {
    seedProject("someone-else");
    vi.mocked(api.listInferenceModels).mockResolvedValue([
      {
        id: "seg-a",
        name: "kraken",
        provider: "kraken",
        task: "segment",
        artifact_ref: "registry://blla-segment?tag=stable",
        default_params: {},
        created_at: "2026-09-01T00:00:00Z",
      },
      {
        id: "seg-b",
        name: "pp-ocr",
        provider: "ppocr",
        task: "segment",
        artifact_ref: "registry://ppocr-segment?tag=stable",
        default_params: {},
        created_at: "2026-09-01T00:00:00Z",
      },
    ]);
    vi.mocked(api.createProjectModelBinding).mockResolvedValue({
      id: "binding-1",
      project_id: "project-1",
      document_id: null,
      task: "segment",
      model_id: "seg-b",
      created_at: "2026-09-21T00:00:00Z",
      updated_at: "2026-09-21T00:00:00Z",
    });

    renderDocumentPage();
    await screen.findByRole("heading", { name: "Chapter 4" });
    await openMenu("Workflow", "Document workflow");
    await screen.findByRole("option", { name: "pp-ocr" });

    // The card that also sets this lives in the owner-only settings popover,
    // so the menu is the whole of a member's way to the project default.
    fireEvent.change(
      screen.getByRole("combobox", { name: "Segmentation model" }),
      { target: { value: "seg-b" } },
    );

    await waitFor(() =>
      expect(api.createProjectModelBinding).toHaveBeenCalledWith("project-1", {
        task: "segment",
        model_id: "seg-b",
      }),
    );
    expect(await screen.findByText("Saved as project default.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Undo" })).toBeTruthy();
  });
});
