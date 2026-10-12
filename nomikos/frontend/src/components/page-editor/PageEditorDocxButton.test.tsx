import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { toast } from "../ui/toast";
import { PageEditorDocxButton } from "./PageEditorDocxButton";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      exportPartDocx: vi.fn(),
    },
  };
});

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));
const mockedExportPartDocx = vi.mocked(api.exportPartDocx);
const mockedToastError = vi.mocked(toast.error);
function renderButton() {
  return render(
    <PageEditorDocxButton
      projectId="project-1"
      documentId="doc-1"
      partId="part-1"
      downloadFilename="Codex_Syriacus_1_page_2.docx"
      className="pe-dd-item"
      role="menuitem"
      onActivate={vi.fn()}
    />,
  );
}

describe("PageEditorDocxButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedExportPartDocx.mockResolvedValue(
      new Blob(["PK"], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    );
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:docx"),
      revokeObjectURL: vi.fn(),
    });
  });
  it("labels the menu item Word (.docx)", () => {
    renderButton();
    expect(screen.getByText("Word (.docx)")).toBeTruthy();
  });

  it("downloads the page as Word through the per-page docx route", async () => {
    // Spy on the click rather than on document.createElement: a createElement
    // stub returning a plain object breaks React's own rendering.
    const clicked: { href: string | null; download: string }[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clicked.push({
          href: this.getAttribute("href"),
          download: this.download,
        });
      });

    renderButton();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /download word document/i }),
    );

    await waitFor(() => {
      expect(mockedExportPartDocx).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
      );
      expect(clicked).toEqual([
        { href: "blob:docx", download: "Codex_Syriacus_1_page_2.docx" },
      ]);
    });

    clickSpy.mockRestore();
    vi.unstubAllGlobals();
  });

  it("shows preparing while the Word document builds", async () => {
    let release!: (blob: Blob) => void;
    mockedExportPartDocx.mockReturnValue(
      new Promise<Blob>((resolve) => {
        release = resolve;
      }),
    );

    renderButton();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /download word document/i }),
    );

    expect(await screen.findByText("preparing…")).toBeTruthy();

    release(
      new Blob(["PK"], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      }),
    );
    await waitFor(() => {
      expect(screen.queryByText("preparing…")).toBeNull();
    });

    vi.unstubAllGlobals();
  });

  it("toasts when the Word document cannot be built", async () => {
    mockedExportPartDocx.mockRejectedValue(new Error("boom"));

    renderButton();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /download word document/i }),
    );

    await waitFor(() => {
      expect(mockedToastError).toHaveBeenCalledWith(
        "Could not build the Word document",
      );
    });

    vi.unstubAllGlobals();
  });

  it("surfaces the API message when the server refuses the export", async () => {
    mockedExportPartDocx.mockRejectedValue(new ApiError("Forbidden", 403));

    renderButton();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /download word document/i }),
    );

    await waitFor(() => {
      expect(mockedToastError).toHaveBeenCalledWith("Forbidden");
    });

    vi.unstubAllGlobals();
  });
});
