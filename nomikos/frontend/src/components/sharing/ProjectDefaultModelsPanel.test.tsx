import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ProjectDefaultModelsPanel } from "./ProjectDefaultModelsPanel";

const listInferenceModels = vi.fn();
const listProjectModelBindings = vi.fn();
const createProjectModelBinding = vi.fn();
const updateProjectModelBinding = vi.fn();
const deleteProjectModelBinding = vi.fn();
const success = vi.fn();
const error = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    createProjectModelBinding: (...args: unknown[]) =>
      createProjectModelBinding(...args),
    updateProjectModelBinding: (...args: unknown[]) =>
      updateProjectModelBinding(...args),
    deleteProjectModelBinding: (...args: unknown[]) =>
      deleteProjectModelBinding(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: {
    success: (...args: unknown[]) => success(...args),
    error: (...args: unknown[]) => error(...args),
  },
}));

const CATALOG = [
  { id: "seg-kraken", task: "segment", name: "kraken" },
  { id: "seg-ppocr", task: "segment", name: "ppocr" },
  { id: "htr-greek", task: "transcribe", name: "greek-calamari-v1" },
  { id: "htr-syriac", task: "transcribe", name: "syriac-ppocr-v1" },
];

function panel() {
  render(<ProjectDefaultModelsPanel projectId="project-1" />);
}

function segmentRow(): HTMLSelectElement {
  return screen.getByLabelText("Segmentation") as HTMLSelectElement;
}

function transcribeRow(): HTMLSelectElement {
  return screen.getByLabelText("Transcription") as HTMLSelectElement;
}

describe("ProjectDefaultModelsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(CATALOG);
    listProjectModelBindings.mockResolvedValue([]);
  });

  it("renders both rows on the stored bindings", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-ppocr" },
    ]);
    panel();

    await waitFor(() => expect(segmentRow()).toHaveValue("seg-ppocr"));
    expect(transcribeRow()).toHaveValue("");
    expect(screen.getAllByRole("option", { name: "No default" })).toHaveLength(
      2,
    );
    // Each select offers only the models for its own task.
    expect(
      Array.from(segmentRow().options).map((option) => option.textContent),
    ).toEqual(["No default", "kraken", "ppocr"]);
    expect(
      Array.from(transcribeRow().options).map((option) => option.textContent),
    ).toEqual(["No default", "greek-calamari-v1", "syriac-ppocr-v1"]);
  });

  it("creates the binding when the project has none", async () => {
    createProjectModelBinding.mockResolvedValue({
      id: "binding-new",
      task: "transcribe",
      model_id: "htr-syriac",
    });
    panel();
    await waitFor(() =>
      expect(transcribeRow().options.length).toBeGreaterThan(1),
    );

    fireEvent.change(transcribeRow(), { target: { value: "htr-syriac" } });

    await waitFor(() =>
      expect(createProjectModelBinding).toHaveBeenCalledWith("project-1", {
        task: "transcribe",
        model_id: "htr-syriac",
      }),
    );
    await waitFor(() => expect(transcribeRow()).toHaveValue("htr-syriac"));
    expect(success).toHaveBeenCalledWith(
      "Default transcription model set to syriac-ppocr-v1",
    );
  });

  it("updates the binding the project already has", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-kraken" },
    ]);
    updateProjectModelBinding.mockResolvedValue({
      id: "binding-1",
      task: "segment",
      model_id: "seg-ppocr",
    });
    panel();
    await waitFor(() => expect(segmentRow()).toHaveValue("seg-kraken"));

    fireEvent.change(segmentRow(), { target: { value: "seg-ppocr" } });

    await waitFor(() =>
      expect(updateProjectModelBinding).toHaveBeenCalledWith(
        "project-1",
        "binding-1",
        { model_id: "seg-ppocr" },
      ),
    );
    expect(createProjectModelBinding).not.toHaveBeenCalled();
    await waitFor(() => expect(segmentRow()).toHaveValue("seg-ppocr"));
    expect(success).toHaveBeenCalledWith(
      "Default segmentation model set to ppocr",
    );
  });

  it("deletes the binding for No default", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-ppocr" },
    ]);
    deleteProjectModelBinding.mockResolvedValue(undefined);
    panel();
    await waitFor(() => expect(segmentRow()).toHaveValue("seg-ppocr"));

    fireEvent.change(segmentRow(), { target: { value: "" } });

    await waitFor(() =>
      expect(deleteProjectModelBinding).toHaveBeenCalledWith(
        "project-1",
        "binding-1",
      ),
    );
    await waitFor(() => expect(segmentRow()).toHaveValue(""));
    expect(success).toHaveBeenCalledWith("Default segmentation model cleared");
  });

  it("says it could not read the defaults instead of showing No default", async () => {
    listProjectModelBindings.mockRejectedValueOnce(new Error("offline"));
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-ppocr" },
    ]);
    panel();

    await screen.findByText("Could not load the project defaults.");
    // An unknown default is never drawn as a state the project is in.
    expect(segmentRow()).toBeDisabled();
    expect(transcribeRow()).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(segmentRow()).toHaveValue("seg-ppocr"));
    expect(
      screen.queryByText("Could not load the project defaults."),
    ).toBeNull();
    expect(segmentRow()).not.toBeDisabled();
  });

  it("returns the select to the stored value when the save fails", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-kraken" },
    ]);
    updateProjectModelBinding.mockRejectedValue(new Error("No access"));
    panel();
    await waitFor(() => expect(segmentRow()).toHaveValue("seg-kraken"));

    fireEvent.change(segmentRow(), { target: { value: "seg-ppocr" } });

    await waitFor(() => expect(error).toHaveBeenCalledWith("No access"));
    expect(segmentRow()).toHaveValue("seg-kraken");
  });
});
