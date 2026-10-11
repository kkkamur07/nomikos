import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

function groundTruthLine(id: string, order: number, text: string) {
  return line({
    id,
    order,
    points: [
      [10, 10 + order * 50],
      [110, 10 + order * 50],
      [110, 40 + order * 50],
      [10, 40 + order * 50],
    ],
    line_transcriptions: [
      {
        id: `${id}-tx-1`,
        transcription_id: "ground-truth-1",
        transcription_kind: "ground_truth",
        text,
        confidence: null,
      },
    ],
  });
}

function twoLines() {
  return [
    groundTruthLine("line-1", 0, "first words"),
    groundTruthLine("line-2", 1, "second words"),
  ];
}

function seedSideBySide(on: boolean) {
  localStorage.setItem(
    "nomikos_page_editor_settings",
    JSON.stringify({ sideBySide: on }),
  );
}

function seedEditor() {
  mockedApi.getDocument.mockResolvedValue(DOCUMENT);
  mockedApi.listPartLines.mockResolvedValue(twoLines());
}

function transcriptRow(container: HTMLElement, lineId: string): HTMLElement {
  const row = container.querySelector(
    `.pe-transcript [data-line-id="${lineId}"]`,
  );
  expect(row).not.toBeNull();
  return row as HTMLElement;
}

describe("PageEditorPlaceholderPage side by side", () => {
  beforeEach(() => {
    localStorage.clear();
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    await flushPageEditorEffects();
    localStorage.clear();
  });

  it("renders the canvas left and the transcript right when the setting is on", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();

    const canvas = await screen.findByLabelText("Page geometry canvas");
    expect(container.querySelector(".pe-split")).not.toBeNull();
    expect(container.querySelector(".pe-split-left")?.contains(canvas)).toBe(
      true,
    );
    expect(
      container.querySelector(".pe-split-right .pe-transcript"),
    ).not.toBeNull();
  });

  it("mirrors hovering a transcript row onto the same canvas segment", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.mouseEnter(transcriptRow(container, "line-1"));

    await waitFor(() => {
      expect(
        screen
          .getByRole("button", { name: /^Segment 1/ })
          .classList.contains("is-hovered"),
      ).toBe(true);
    });
  });

  it("selects the segment when its transcript row is focused", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    expect(transcriptRow(container, "line-1")).toBeTruthy();

    fireEvent.focus(screen.getByLabelText("Segment 1 text"));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /^Segment 1/ }),
      ).toHaveAttribute("aria-current", "true");
    });
  });

  it("saves the edited text and focuses the next row on Enter", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    expect(transcriptRow(container, "line-1")).toBeTruthy();

    const editor = screen.getByLabelText("Segment 1 text");
    fireEvent.change(editor, { target: { value: "  first words edited  " } });
    fireEvent.keyDown(editor, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-1",
        { text: "first words edited" },
      );
    });
    await waitFor(() => {
      expect(screen.getByLabelText("Segment 2 text")).toBe(
        document.activeElement,
      );
    });
  });

  it("saves any row through the pairing hook and refreshes pairing progress", async () => {
    seedSideBySide(true);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    const pairingCallsBefore = mockedApi.getPagePairing.mock.calls.length;

    const second = screen.getByLabelText("Segment 2 text");
    second.focus();
    fireEvent.change(second, { target: { value: "second edited" } });
    fireEvent.keyDown(second, { key: "Enter", ctrlKey: true });

    await waitFor(() => {
      expect(mockedApi.updateGroundTruthLineText).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "ground-truth-1",
        "line-2",
        { text: "second edited" },
      );
    });
    await waitFor(() => {
      expect(mockedApi.getPagePairing.mock.calls.length).toBeGreaterThan(
        pairingCallsBefore,
      );
    });
    expect(document.activeElement).toBe(second);
  });

  it("keeps the draft and shows the row unsaved when the server rejects", async () => {
    seedSideBySide(true);
    seedEditor();
    mockedApi.updateGroundTruthLineText.mockRejectedValueOnce(
      new Error("Server said no"),
    );
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const editor = screen.getByLabelText(
      "Segment 1 text",
    ) as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "rejected words" } });
    fireEvent.blur(editor);

    await waitFor(() => {
      expect(
        transcriptRow(container, "line-1").getAttribute("data-status"),
      ).toBe("unsaved");
    });
    expect(editor.value).toBe("rejected words");
  });

  it("lays a Syriac row out right to left", async () => {
    seedSideBySide(true);
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      groundTruthLine("line-1", 0, "ܫܠܡܐ ܥܠܝܟܘܢ"),
      groundTruthLine("line-2", 1, "first words"),
    ]);
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    const syriac = screen.getByLabelText("Segment 1 text");
    expect(syriac.getAttribute("dir")).toBe("rtl");
    expect(screen.getByLabelText("Segment 2 text").getAttribute("dir")).toBe(
      "ltr",
    );
  });

  it("marks the transcript row selected without focusing it when its canvas segment is clicked", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.click(screen.getByRole("button", { name: /^Segment 2/ }));

    await waitFor(() => {
      expect(
        transcriptRow(container, "line-2").classList.contains("is-selected"),
      ).toBe(true);
    });
    expect(screen.getByLabelText("Segment 2 text")).not.toBe(
      document.activeElement,
    );
  });

  it("keeps the segment strip closed when a canvas segment is clicked and the setting is on", async () => {
    seedSideBySide(true);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /^Segment 1/ }),
      ).toHaveAttribute("aria-current", "true");
    });
    expect(
      screen.queryByLabelText(/ground truth text for selected segment/i),
    ).toBeNull();
    expect(screen.queryByRole("heading", { name: /segment 1/i })).toBeNull();
  });

  it("still opens the segment strip when a canvas segment is clicked and the setting is off", async () => {
    seedSideBySide(false);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));

    expect(
      await screen.findByLabelText(/ground truth text for selected segment/i),
    ).toBeTruthy();
  });

  it("renders no split pane when the setting is off", async () => {
    seedSideBySide(false);
    seedEditor();
    const { container } = renderPageEditor();

    expect(await screen.findByLabelText("Page geometry canvas")).toBeTruthy();
    expect(container.querySelector(".pe-split")).toBeNull();
    expect(container.querySelector(".pe-transcript")).toBeNull();
  });

  it("hides the pane with the header x and offers the Transcript button back", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    expect(
      screen.queryByRole("button", { name: /show transcript pane/i }),
    ).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Hide transcript pane" }),
    );

    await waitFor(() => {
      expect(container.querySelector(".pe-split")).toBeNull();
    });
    expect(container.querySelector(".pe-transcript")).toBeNull();
    expect(
      JSON.parse(localStorage.getItem("nomikos_page_editor_settings") ?? "{}")
        .sideBySide,
    ).toBe(false);
    expect(
      screen.getByRole("button", { name: /show transcript pane \(t\)/i }),
    ).toBeTruthy();
  });

  it("reopens the pane from the toolbar Transcript button", async () => {
    seedSideBySide(false);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.click(
      screen.getByRole("button", { name: /show transcript pane \(t\)/i }),
    );

    await waitFor(() => {
      expect(
        container.querySelector(".pe-split-right .pe-transcript"),
      ).not.toBeNull();
    });
    expect(
      screen.queryByRole("button", { name: /show transcript pane/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: "Hide transcript pane" }),
    ).toBeTruthy();
  });

  it("toggles the pane with T but ignores T while typing", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    expect(
      container.querySelector(".pe-split-right .pe-transcript"),
    ).not.toBeNull();

    fireEvent.keyDown(window, { key: "t" });
    await waitFor(() => {
      expect(container.querySelector(".pe-split")).toBeNull();
    });

    fireEvent.keyDown(window, { key: "T" });
    await waitFor(() => {
      expect(
        container.querySelector(".pe-split-right .pe-transcript"),
      ).not.toBeNull();
    });

    const first = screen.getByLabelText("Segment 1 text");
    first.focus();
    fireEvent.keyDown(first, { key: "t" });
    expect(
      container.querySelector(".pe-split-right .pe-transcript"),
    ).not.toBeNull();
  });
});
