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
const createProjectModelBinding = vi.fn();
const updateProjectModelBinding = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    createProjectModelBinding: (...args: unknown[]) =>
      createProjectModelBinding(...args),
    updateProjectModelBinding: (...args: unknown[]) =>
      updateProjectModelBinding(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), info: vi.fn(), error: vi.fn() },
}));

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 0 };
const SEGMENT_MODELS = [
  {
    id: "seg-a",
    task: "segment",
    name: "kraken",
    artifact_ref: "registry://blla-segment?tag=stable",
  },
  {
    id: "seg-b",
    task: "segment",
    name: "pp-ocr",
    artifact_ref: "registry://ppocr-segment?tag=stable",
  },
  { id: "htr-1", task: "transcribe", name: "htr" },
  { id: "htr-2", task: "transcribe", name: "syriac" },
];

function openMenu(counts = COUNTS) {
  render(
    <DocumentWorkflowMenu
      projectId="project-1"
      documentId="document-1"
      counts={counts}
      onJobsQueued={() => {}}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
}

describe("DocumentWorkflowMenu segment picker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(SEGMENT_MODELS);
    listProjectModelBindings.mockResolvedValue([]);
    // Picking a model now saves it as the project default, so every test in
    // here needs the write to answer.
    createProjectModelBinding.mockImplementation(async (_projectId, body) => ({
      id: `binding-${body.task}`,
      ...body,
    }));
    updateProjectModelBinding.mockImplementation(
      async (_projectId, bindingId, body) => ({
        id: bindingId,
        task: "segment",
        ...body,
      }),
    );
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("renders no Default option and preselects the canonical row", async () => {
    openMenu();

    const select = await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    const options = within(select)
      .getAllByRole("option")
      .map((option) => ({
        value: (option as HTMLOptionElement).value,
        text: option.textContent,
      }));
    expect(options).toEqual([
      { value: "seg-a", text: "kraken" },
      { value: "seg-b", text: "pp-ocr" },
    ]);
    expect(
      screen.getByRole("combobox", { name: "Segmentation model" }),
    ).toHaveValue("seg-a");

    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: "seg-a" },
      ),
    );
  });

  it("labels each picker Model and counts the pages on each item", async () => {
    openMenu({ ...COUNTS, unsegmented: 1, total: 3, unpaired: 0 });
    await screen.findByRole("combobox", { name: "Segmentation model" });

    expect(screen.getAllByText("Model")).toHaveLength(2);
    expect(screen.queryByText("Seg")).toBeNull();
    expect(screen.queryByText("HTR")).toBeNull();
    expect(
      screen.getByRole("menuitem", {
        name: /segment unsegmented pages\s*1 page$/i,
      }),
    ).toBeTruthy();
    expect(
      screen.getByRole("menuitem", { name: /re-segment every page.*3 pages/i }),
    ).toBeTruthy();
    expect(
      screen.getByText("Discards unapproved machine text on untouched lines."),
    ).toBeTruthy();
    expect(screen.queryByText(/Only the top item is safe/)).toBeNull();
    expect(screen.queryByText(/project default/i)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /set as project default/i }),
    ).toBeNull();
  });

  it("sends the id for a chosen model", async () => {
    openMenu();

    const select = await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    fireEvent.change(select, { target: { value: "seg-b" } });
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

  it("names the chosen model in the re-segment confirm", async () => {
    openMenu();
    await screen.findByRole("combobox", { name: "Segmentation model" });

    fireEvent.click(
      screen.getByRole("menuitem", { name: /re-segment every page/i }),
    );
    expect(screen.getByText(/runs with kraken/i)).toBeTruthy();

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

  it("shows No models and still segments when the catalog request fails", async () => {
    listInferenceModels.mockRejectedValue(new Error("offline"));
    openMenu();

    const select = await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    expect(within(select).getAllByRole("option")).toHaveLength(1);
    expect(
      within(select).getByRole("option", { name: "No models" }),
    ).toBeTruthy();

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

describe("DocumentWorkflowMenu transcribe picker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(SEGMENT_MODELS);
    listProjectModelBindings.mockResolvedValue([]);
    // Picking a model now saves it as the project default, so every test in
    // here needs the write to answer.
    createProjectModelBinding.mockImplementation(async (_projectId, body) => ({
      id: `binding-${body.task}`,
      ...body,
    }));
    updateProjectModelBinding.mockImplementation(
      async (_projectId, bindingId, body) => ({
        id: bindingId,
        task: "segment",
        ...body,
      }),
    );
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("offers a transcribe picker and sends the chosen model id", async () => {
    openMenu({ ...COUNTS, unpaired: 2 });
    const select = await screen.findByRole("combobox", {
      name: "HTR transcription model",
    });
    await waitFor(() => expect(select).toHaveValue("htr-1"));
    fireEvent.change(select, { target: { value: "htr-2" } });
    fireEvent.click(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentTranscribe).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unpaired", model_id: "htr-2" },
      ),
    );
  });

  it("starts the transcribe picker from the project binding", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "transcribe", model_id: "htr-2" },
    ]);
    openMenu();
    const select = await screen.findByRole("combobox", {
      name: "HTR transcription model",
    });
    await waitFor(() => expect(select).toHaveValue("htr-2"));
  });

  it("starts the segment picker from the project binding, not the canonical row", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-2", task: "segment", model_id: "seg-b" },
    ]);
    openMenu();
    const select = await screen.findByRole("combobox", {
      name: "Segmentation model",
    });
    await waitFor(() => expect(select).toHaveValue("seg-b"));
  });
});
