import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentFontSettings } from "./DocumentFontSettings";
import { choosableScripts, scriptFontAttributes } from "./scriptFonts";

const updateDocument = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    updateDocument: (...args: unknown[]) => updateDocument(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

describe("scriptFonts", () => {
  it("offers a choice only for scripts with more than one face", () => {
    expect(choosableScripts()).toEqual(["syriac"]);
  });

  it("emits attributes only for known non-default choices", () => {
    expect(scriptFontAttributes(null)).toEqual({});
    expect(scriptFontAttributes({ syriac: "estrangela" })).toEqual({});
    expect(scriptFontAttributes({ syriac: "eastern" })).toEqual({
      "data-font-syriac": "eastern",
    });
    expect(scriptFontAttributes({ syriac: "meltho", klingon: "x" })).toEqual(
      {},
    );
  });
});

describe("DocumentFontSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows Estrangela by default and patches the chosen style", async () => {
    updateDocument.mockResolvedValue({ script_fonts: { syriac: "western" } });
    const onUpdated = vi.fn();
    render(
      <DocumentFontSettings
        projectId="p"
        documentId="d"
        scriptFonts={{}}
        onUpdated={onUpdated}
      />,
    );
    const select = screen.getByLabelText("Syriac") as HTMLSelectElement;
    expect(select.value).toBe("estrangela");
    expect(screen.queryByLabelText("Greek")).toBeNull();

    fireEvent.change(select, { target: { value: "western" } });
    await waitFor(() =>
      expect(onUpdated).toHaveBeenCalledWith({ syriac: "western" }),
    );
    expect(updateDocument).toHaveBeenCalledWith("p", "d", {
      script_fonts: { syriac: "western" },
    });
  });

  it("stores the default as no entry", async () => {
    updateDocument.mockResolvedValue({ script_fonts: {} });
    render(
      <DocumentFontSettings
        projectId="p"
        documentId="d"
        scriptFonts={{ syriac: "eastern" }}
        onUpdated={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByLabelText("Syriac"), {
      target: { value: "estrangela" },
    });
    await waitFor(() =>
      expect(updateDocument).toHaveBeenCalledWith("p", "d", {
        script_fonts: {},
      }),
    );
  });
});
