import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../api/client";
import { PageEditorTextPanel } from "./PageEditorTextPanel";

const RECT = [
  [10, 10],
  [110, 10],
  [110, 40],
  [10, 40],
];

function makeLine(
  id: string,
  order: number,
  texts: { kind: "ground_truth" | "model"; text: string }[],
): LineResponse {
  return {
    id,
    order,
    baseline: [
      [10, 35],
      [110, 35],
    ],
    mask: RECT,
    points: RECT,
    line_transcriptions: texts.map((entry, index) => ({
      id: `${id}-tx-${index}`,
      transcription_id: `${entry.kind}-${id}`,
      transcription_kind: entry.kind,
      text: entry.text,
      confidence: null,
    })),
  } as unknown as LineResponse;
}

const LINE_A = () =>
  makeLine("line-a", 0, [
    { kind: "model", text: "model words" },
    { kind: "ground_truth", text: "true words" },
  ]);
const LINE_B = () =>
  makeLine("line-b", 1, [{ kind: "model", text: "second line" }]);

function panelProps(overrides: Record<string, unknown> = {}) {
  return {
    lines: [LINE_A(), LINE_B()],
    imageWidth: 200,
    imageHeight: 100,
    selectedSegmentId: null,
    hoveredSegmentId: null,
    editingSegmentId: null,
    textDirection: "ltr" as const,
    onSelectSegment: () => {},
    onHoverSegment: () => {},
    onRequestEdit: () => {},
    onCommitText: () => {},
    ...overrides,
  };
}

describe("PageEditorTextPanel", () => {
  it("draws one textPath per line, preferring ground truth", () => {
    const { container } = render(<PageEditorTextPanel {...panelProps()} />);
    const paths = container.querySelectorAll("textPath");
    expect(paths).toHaveLength(2);
    expect(paths[0]?.textContent).toBe("true words");
    expect(paths[1]?.textContent).toBe("second line");
  });

  it("sets rtl direction and end anchoring", () => {
    const { container } = render(
      <PageEditorTextPanel {...panelProps({ textDirection: "rtl" })} />,
    );
    const texts = container.querySelectorAll("text");
    expect(texts).toHaveLength(2);
    for (const text of texts) {
      expect(text.getAttribute("direction")).toBe("rtl");
      expect(text.getAttribute("text-anchor")).toBe("end");
    }
    const paths = container.querySelectorAll("textPath");
    for (const path of paths) {
      expect(path.getAttribute("startOffset")).toBe("100%");
    }
  });

  it("fires hover and select callbacks", () => {
    const onHoverSegment = vi.fn();
    const onSelectSegment = vi.fn();
    const { container } = render(
      <PageEditorTextPanel
        {...panelProps({ onHoverSegment, onSelectSegment })}
      />,
    );
    const group = container.querySelector('[data-line-id="line-a"]');
    expect(group).not.toBeNull();
    fireEvent.mouseEnter(group!);
    expect(onHoverSegment).toHaveBeenCalledWith("line-a");
    fireEvent.mouseLeave(group!);
    expect(onHoverSegment).toHaveBeenCalledWith(null);
    fireEvent.click(group!);
    expect(onSelectSegment).toHaveBeenCalledWith("line-a");
  });

  it("edits in place: Enter commits the trimmed value", async () => {
    const onCommitText = vi.fn().mockResolvedValue(undefined);
    const onRequestEdit = vi.fn();
    const onCommitted = vi.fn();
    render(
      <PageEditorTextPanel
        {...panelProps({
          textDirection: "rtl",
          editingSegmentId: "line-a",
          onCommitText,
          onRequestEdit,
          onCommitted,
        })}
      />,
    );
    const editor = screen.getByDisplayValue("true words");
    expect(editor.tagName.toLowerCase()).toBe("textarea");
    expect(editor.getAttribute("dir")).toBe("rtl");
    fireEvent.change(editor, { target: { value: "  new words  " } });
    fireEvent.keyDown(editor, { key: "Enter", shiftKey: false });
    await vi.waitFor(() => {
      expect(onCommitText).toHaveBeenCalledWith("line-a", "new words");
    });
    expect(onRequestEdit).toHaveBeenCalledWith(null);
    expect(onCommitted).toHaveBeenCalledWith("line-a");
  });

  it("focuses once on mount and keeps the caret while typing", () => {
    const focusSpy = vi.spyOn(HTMLTextAreaElement.prototype, "focus");
    try {
      const view = render(
        <PageEditorTextPanel {...panelProps({ editingSegmentId: "line-a" })} />,
      );
      const editor = screen.getByDisplayValue(
        "true words",
      ) as HTMLTextAreaElement;
      fireEvent.change(editor, { target: { value: "true words!" } });
      // A programmatic value set moves the caret in jsdom, so place it
      // mid-line afterwards, the way a user editing there would.
      editor.setSelectionRange(2, 2);
      view.rerender(
        <PageEditorTextPanel {...panelProps({ editingSegmentId: "line-a" })} />,
      );
      expect(editor.selectionStart).toBe(2);
      fireEvent.change(editor, { target: { value: "true words!?" } });
      expect(focusSpy).toHaveBeenCalledTimes(1);
    } finally {
      focusSpy.mockRestore();
    }
  });

  it("cancels on Escape without committing", () => {
    const onCommitText = vi.fn();
    const onRequestEdit = vi.fn();
    render(
      <PageEditorTextPanel
        {...panelProps({
          editingSegmentId: "line-a",
          onCommitText,
          onRequestEdit,
        })}
      />,
    );
    const editor = screen.getByDisplayValue("true words");
    fireEvent.change(editor, { target: { value: "changed" } });
    fireEvent.keyDown(editor, { key: "Escape" });
    expect(onRequestEdit).toHaveBeenCalledWith(null);
    expect(onCommitText).not.toHaveBeenCalled();
  });
});
