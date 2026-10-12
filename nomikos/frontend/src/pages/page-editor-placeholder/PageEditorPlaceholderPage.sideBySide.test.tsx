import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DOCUMENT,
  documentWithPages,
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
    // Unsaved row drafts persist per tab; one test's draft must not seed the next.
    sessionStorage.clear();
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

  it("numbers a Segment drawn after line 2 as row 3 and shifts the rest", async () => {
    // The server places a drawn Segment and renumbers the ones after it (ADR
    // 0008); the pane must show that order at once, not the drawn one last.
    seedSideBySide(true);
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    const before = [
      groundTruthLine("line-1", 0, "first words"),
      groundTruthLine("line-2", 1, "second words"),
      groundTruthLine("line-3", 2, "third words"),
    ];
    let drawn = false;
    mockedApi.listPartLines.mockImplementation(async () =>
      drawn
        ? [
            before[0],
            before[1],
            line({ id: "z-drawn", order: 2 }),
            { ...before[2], order: 3 },
          ]
        : before,
    );
    mockedApi.createPartLine.mockImplementation(
      async (_projectId, _documentId, _partId, body) => {
        drawn = true;
        return line({ id: "z-drawn", order: 2, points: body.points });
      },
    );
    const { container } = renderPageEditor();
    await waitFor(() => transcriptRow(container, "line-3"));

    fireEvent.click(
      await screen.findByRole("button", { name: /rectangle segment/i }),
    );
    const canvas = screen.getByRole("group", {
      name: /page geometry canvas/i,
    });
    fireEvent.pointerDown(canvas, { clientX: 20, clientY: 92 });
    fireEvent.pointerMove(canvas, { clientX: 100, clientY: 98 });
    fireEvent.pointerUp(canvas, { clientX: 100, clientY: 98 });

    await waitFor(() => {
      const rows = [
        ...container.querySelectorAll(".pe-transcript [data-line-id]"),
      ].map(
        (row) =>
          `${row.querySelector(".pe-transcript-num")?.textContent}:${row.getAttribute("data-line-id")}`,
      );
      expect(rows).toEqual(["1:line-1", "2:line-2", "3:z-drawn", "4:line-3"]);
    });
    // No `order` goes out: the server decides where the Segment sits.
    expect(mockedApi.createPartLine.mock.calls[0][3]).not.toHaveProperty(
      "order",
    );
  });

  it("keeps the canvas and its zoom when T toggles the pane", async () => {
    seedSideBySide(false);
    seedEditor();
    renderPageEditor();
    const canvas = await screen.findByLabelText("Page geometry canvas");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    const zoomLabel = () =>
      screen
        .getByRole("button", { name: /^Zoom \d+%/ })
        .getAttribute("aria-label");
    // The button animates the step (100%, 101%, 105%, ... 113%), so read the
    // zoom once it has stopped moving, not on the first frame that left 100%.
    await waitFor(async () => {
      const before = zoomLabel();
      await new Promise((resolve) => setTimeout(resolve, 80));
      expect(before).not.toMatch(/^Zoom 100%/);
      expect(zoomLabel()).toBe(before);
    });
    const zoomed = zoomLabel();

    fireEvent.keyDown(window, { key: "t" });
    await screen.findByLabelText("Segment 1 text");
    fireEvent.keyDown(window, { key: "t" });
    await waitFor(() =>
      expect(screen.queryByLabelText("Segment 1 text")).toBeNull(),
    );

    expect(screen.getByLabelText("Page geometry canvas")).toBe(canvas);
    expect(zoomLabel()).toBe(zoomed);
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
        { text: "first words edited", previous_text: "first words" },
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
        { text: "second edited", previous_text: "second words" },
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

  it("lifts the jobs panel by the strip height only while the strip is open", async () => {
    // .pe-job-panel sits at bottom: 16px + --nk-bottom-inset, so the strip's
    // Save button is never under it.
    const inset = () =>
      document.documentElement.style.getPropertyValue("--nk-bottom-inset");
    seedSideBySide(false);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    expect(inset()).toBe("0px");

    fireEvent.click(screen.getByRole("button", { name: /^Segment 1/ }));
    await screen.findByLabelText(/ground truth text for selected segment/i);
    expect(inset()).toBe("240px");

    fireEvent.click(
      screen.getByRole("button", { name: "Hide transcription panel" }),
    );
    await waitFor(() => expect(inset()).toBe("0px"));
  });

  it("renders no split pane when the setting is off", async () => {
    seedSideBySide(false);
    seedEditor();
    const { container } = renderPageEditor();

    expect(await screen.findByLabelText("Page geometry canvas")).toBeTruthy();
    expect(container.querySelector(".pe-split-right:not([hidden])")).toBeNull();
    expect(container.querySelector(".pe-transcript")).toBeNull();
  });

  it("hides the pane with the header x and unpresses the top bar toggle", async () => {
    seedSideBySide(true);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    const toggle = screen.getByRole("button", { name: "Transcript" });
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(
      screen.getByRole("button", { name: "Hide transcript pane" }),
    );

    await waitFor(() => {
      expect(
        container.querySelector(".pe-split-right:not([hidden])"),
      ).toBeNull();
    });
    expect(container.querySelector(".pe-transcript")).toBeNull();
    expect(
      JSON.parse(localStorage.getItem("nomikos_page_editor_settings") ?? "{}")
        .sideBySide,
    ).toBe(false);
    expect(
      screen
        .getByRole("button", { name: "Transcript" })
        .getAttribute("aria-pressed"),
    ).toBe("false");
  });

  it("opens and closes the pane from the top bar Transcript toggle", async () => {
    seedSideBySide(false);
    seedEditor();
    const { container } = renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    // The toggle lives in the top bar only: the island no longer has one.
    expect(
      screen.queryByRole("button", { name: /^transcript \(t\)$/i }),
    ).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Transcript" }));

    await waitFor(() => {
      expect(
        container.querySelector(".pe-split-right .pe-transcript"),
      ).not.toBeNull();
    });
    expect(
      screen
        .getByRole("button", { name: "Transcript" })
        .getAttribute("aria-pressed"),
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "Transcript" }));
    await waitFor(() => {
      expect(
        container.querySelector(".pe-split-right:not([hidden])"),
      ).toBeNull();
    });
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
      expect(
        container.querySelector(".pe-split-right:not([hidden])"),
      ).toBeNull();
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

  it("focuses the selected row when T opens the pane", async () => {
    seedSideBySide(false);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    fireEvent.click(screen.getByRole("button", { name: /^Segment 2/ }));

    fireEvent.keyDown(window, { key: "t" });

    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByLabelText("Segment 2 text"),
      );
    });
  });

  it("goes from the selected canvas segment to its row on Enter", async () => {
    seedSideBySide(false);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    const segment = screen.getByRole("button", { name: /^Segment 2/ });
    fireEvent.click(segment);
    await waitFor(() => {
      expect(segment.getAttribute("aria-current")).toBe("true");
    });

    fireEvent.keyDown(segment, { key: "Enter" });

    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByLabelText("Segment 2 text"),
      );
    });
  });

  it("returns focus to the selected segment on Escape from a clean row", async () => {
    seedSideBySide(true);
    seedEditor();
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    const row = screen.getByLabelText("Segment 2 text");
    row.focus();
    await waitFor(() => {
      expect(
        screen
          .getByRole("button", { name: /^Segment 2/ })
          .getAttribute("aria-current"),
      ).toBe("true");
    });

    fireEvent.keyDown(row, { key: "Escape" });

    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /^Segment 2/ }),
    );
  });

  it("asks before a page turn drops an unsaved row and stays when told to", async () => {
    seedSideBySide(true);
    mockedApi.getDocument.mockResolvedValue(documentWithPages(2));
    mockedApi.listPartLines.mockResolvedValue(twoLines());
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");
    const row = screen.getByLabelText("Segment 1 text") as HTMLTextAreaElement;
    fireEvent.change(row, { target: { value: "not saved yet" } });

    fireEvent.keyDown(window, { key: "PageDown" });

    expect(confirm).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      /p\.1$/,
    );
    expect(row.value).toBe("not saved yet");
    confirm.mockRestore();
  });

  it("turns the page without asking when no row holds unsaved text", async () => {
    seedSideBySide(true);
    mockedApi.getDocument.mockResolvedValue(documentWithPages(2));
    mockedApi.listPartLines.mockResolvedValue(twoLines());
    const confirm = vi.spyOn(window, "confirm");
    renderPageEditor();
    await screen.findByLabelText("Page geometry canvas");

    fireEvent.keyDown(window, { key: "PageDown" });

    await waitFor(() => {
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
        /p\.2$/,
      );
    });
    expect(confirm).not.toHaveBeenCalled();
    confirm.mockRestore();
  });
});
