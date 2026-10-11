import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { InferenceModelResponse } from "../../api/client";
import { PageEditorModelSelect } from "./PageEditorModelSelect";

function model(id: string, name: string): InferenceModelResponse {
  return {
    id,
    name,
    task: "segment",
    provider: "test",
    artifact_ref: `ref-${id}`,
    created_at: "2026-01-01T00:00:00Z",
    default_params: {},
  };
}

describe("PageEditorModelSelect", () => {
  it("renders Default first for the segment variant, then the catalog names", () => {
    render(
      <PageEditorModelSelect
        label="Seg"
        ariaLabel="Segmentation model"
        models={[model("seg-a", "seg-v1"), model("seg-b", "seg-v2")]}
        selectedModelId={null}
        onSelectedModelIdChange={() => {}}
        includeDefaultOption
      />,
    );

    const options = screen.getAllByRole("option").map((option) => ({
      value: (option as HTMLOptionElement).value,
      text: option.textContent,
    }));
    expect(options).toEqual([
      { value: "", text: "Default" },
      { value: "seg-a", text: "seg-v1" },
      { value: "seg-b", text: "seg-v2" },
    ]);
  });

  it("does not render Default for the HTR variant", () => {
    const onChange = vi.fn();
    render(
      <PageEditorModelSelect
        label="HTR"
        ariaLabel="HTR transcription model"
        models={[model("htr-1", "blla-greek-v2")]}
        selectedModelId="htr-1"
        onSelectedModelIdChange={onChange}
      />,
    );

    expect(screen.getByText("HTR")).toBeTruthy();
    const select = screen.getByRole("combobox", {
      name: "HTR transcription model",
    });
    expect(select).toBeTruthy();
    expect(screen.queryByRole("option", { name: "Default" })).toBeNull();
    expect(screen.getByRole("option", { name: "blla-greek-v2" })).toBeTruthy();

    fireEvent.change(select, { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("renders Default only and stays enabled with zero segment models", () => {
    render(
      <PageEditorModelSelect
        label="Seg"
        ariaLabel="Segmentation model"
        models={[]}
        selectedModelId={null}
        onSelectedModelIdChange={() => {}}
        includeDefaultOption
      />,
    );

    const select = screen.getByRole("combobox", {
      name: "Segmentation model",
    }) as HTMLSelectElement;
    expect(select.disabled).toBe(false);
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByRole("option", { name: "Default" })).toBeTruthy();
  });

  it("shows No models and disables the HTR variant when the catalog is empty", () => {
    render(
      <PageEditorModelSelect
        label="HTR"
        ariaLabel="HTR transcription model"
        models={[]}
        selectedModelId={null}
        onSelectedModelIdChange={() => {}}
      />,
    );

    const select = screen.getByRole("combobox", {
      name: "HTR transcription model",
    }) as HTMLSelectElement;
    expect(select.disabled).toBe(true);
    expect(screen.getByRole("option", { name: "No models" })).toBeTruthy();
  });
});
