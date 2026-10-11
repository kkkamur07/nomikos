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
});
