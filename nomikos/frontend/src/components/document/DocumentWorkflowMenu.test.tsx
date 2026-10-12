import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

const listInferenceModels = vi.fn();
const enqueueDocumentSegment = vi.fn();
const enqueueDocumentTranscribe = vi.fn();
const listProjectModelBindings = vi.fn();
const listDocumentModelBindings = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    listDocumentModelBindings: (...args: unknown[]) =>
      listDocumentModelBindings(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 0 };
const SEGMENT_MODELS = [
  { id: "seg-a", task: "segment", name: "blla-segment" },
  { id: "seg-b", task: "segment", name: "pp-ocr" },
  { id: "htr-1", task: "transcribe", name: "htr" },
];

function openMenu() {
  render(
    <DocumentWorkflowMenu
      projectId="project-1"
      documentId="document-1"
      counts={COUNTS}
      onJobsQueued={() => {}}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
}

describe("DocumentWorkflowMenu segment picker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(SEGMENT_MODELS);
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("sends model_id null for Default and the id for a chosen model", async () => {
    openMenu();

    await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    const options = within(
      screen.getByRole("combobox", { name: "Segmentation model" }),
    )
      .getAllByRole("option")
      .map((option) => ({
        value: (option as HTMLOptionElement).value,
        text: option.textContent,
      }));
    expect(options).toEqual([
      { value: "", text: "Default" },
      { value: "seg-a", text: "blla-segment" },
      { value: "seg-b", text: "pp-ocr" },
    ]);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: null },
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    const reopenedSelect = screen.getByRole("combobox", {
      name: "Segmentation model",
    });
    fireEvent.change(reopenedSelect, { target: { value: "seg-b" } });
    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: "seg-b" },
      ),
    );
  });

  it("names the chosen model in the re-segment confirm, and the default when none is chosen", async () => {
    openMenu();
    await screen.findByRole("combobox", { name: "Segmentation model" });

    fireEvent.click(
      screen.getByRole("menuitem", { name: /re-segment every page/i }),
    );
    expect(screen.getByText(/runs with the default model/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: "Cancel" }));
    fireEvent.change(
      screen.getByRole("combobox", { name: "Segmentation model" }),
      { target: { value: "seg-b" } },
    );
    fireEvent.click(
      screen.getByRole("menuitem", { name: /re-segment every page/i }),
    );
    expect(screen.getByText(/runs with pp-ocr/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("menuitem", { name: /yes, re-segment/i }));
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "all", model_id: "seg-b" },
      ),
    );
  });

  it("leaves only Default and still segments when the catalog request fails", async () => {
    listInferenceModels.mockRejectedValue(new Error("offline"));
    openMenu();

    await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    expect(
      within(
        screen.getByRole("combobox", { name: "Segmentation model" }),
      ).getAllByRole("option"),
    ).toHaveLength(1);

    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: null },
      ),
    );
  });
});

describe("DocumentWorkflowMenu transcribe model", () => {
  const CATALOG = [
    ...SEGMENT_MODELS,
    { id: "htr-greek", task: "transcribe", name: "greek-v1" },
    { id: "htr-syriac", task: "transcribe", name: "syriac-v2" },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(CATALOG);
    listProjectModelBindings.mockResolvedValue([]);
    listDocumentModelBindings.mockResolvedValue([]);
    window.localStorage.clear();
  });

  it("names the model bound to the project", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "b-1", task: "transcribe", model_id: "htr-syriac" },
      { id: "b-2", task: "segment", model_id: "seg-a" },
    ]);
    openMenu();

    expect(await screen.findByText("syriac-v2")).toBeTruthy();
    expect(screen.queryByText("blla-greek-v2")).toBeNull();
  });

  it("prefers the document binding over the project one", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "b-1", task: "transcribe", model_id: "htr-syriac" },
    ]);
    listDocumentModelBindings.mockResolvedValue([
      { id: "b-3", task: "transcribe", model_id: "htr-greek" },
    ]);
    openMenu();

    expect(await screen.findByText("greek-v1")).toBeTruthy();
    expect(screen.queryByText("syriac-v2")).toBeNull();
  });

  it("offers the catalog's transcription models when nothing is bound and sends the choice", async () => {
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    render(
      <DocumentWorkflowMenu
        projectId="project-1"
        documentId="document-1"
        counts={{ ...COUNTS, unpaired: 2 }}
        onJobsQueued={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    const select = await screen.findByRole("combobox", {
      name: "Transcription model",
    });
    await waitFor(() =>
      expect(
        Array.from((select as HTMLSelectElement).options).map(
          (option) => option.text,
        ),
      ).toEqual(["htr", "greek-v1", "syriac-v2"]),
    );
    fireEvent.change(select, { target: { value: "htr-syriac" } });
    expect(screen.queryByText("Project default")).toBeNull();
    expect(
      window.localStorage.getItem("nomikos.transcribeModel.project-1"),
    ).toBe("htr-syriac");

    fireEvent.click(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentTranscribe).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unpaired", model_id: "htr-syriac" },
      ),
    );
  });

  it("remembers the last choice for the project", async () => {
    window.localStorage.setItem(
      "nomikos.transcribeModel.project-1",
      "htr-syriac",
    );
    openMenu();

    const select = await screen.findByRole("combobox", {
      name: "Transcription model",
    });
    await waitFor(() =>
      expect((select as HTMLSelectElement).value).toBe("htr-syriac"),
    );
  });

  it("shows no picker and sends no model when a binding exists", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "b-1", task: "transcribe", model_id: "htr-greek" },
    ]);
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    render(
      <DocumentWorkflowMenu
        projectId="project-1"
        documentId="document-1"
        counts={{ ...COUNTS, unpaired: 2 }}
        onJobsQueued={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    expect(await screen.findByText("greek-v1")).toBeTruthy();
    expect(
      screen.queryByRole("combobox", { name: "Transcription model" }),
    ).toBeNull();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentTranscribe).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unpaired", model_id: null },
      ),
    );
  });
});
