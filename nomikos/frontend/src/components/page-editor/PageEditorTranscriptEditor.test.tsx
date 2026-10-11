import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../api/client";
import { PageEditorTranscriptEditor } from "./PageEditorTranscriptEditor";
import type { PageEditorTranscriptEditorProps } from "./PageEditorTranscriptEditor";

function makeLine(
  id: string,
  order: number,
  top: number,
  texts: { kind: "ground_truth" | "model"; text: string }[],
): LineResponse {
  const mask: [number, number][] = [
    [10, top],
    [110, top],
    [110, top + 30],
    [10, top + 30],
  ];
  return {
    id,
    order,
    baseline: [
      [10, top + 25],
      [110, top + 25],
    ],
    mask,
    points: mask,
    line_transcriptions: texts.map((entry, index) => ({
      id: `${id}-tx-${index}`,
      transcription_id: `${entry.kind}-${id}`,
      transcription_kind: entry.kind,
      text: entry.text,
      confidence: null,
    })),
  } as unknown as LineResponse;
}

function editorProps(
  overrides: Partial<PageEditorTranscriptEditorProps> = {},
): PageEditorTranscriptEditorProps {
  return {
    lines: [
      makeLine("line-a", 0, 10, [{ kind: "ground_truth", text: "first" }]),
      makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "second" }]),
    ],
    selectedSegmentId: null,
    hoveredSegmentId: null,
    textDirection: "ltr",
    onSelectSegment: () => {},
    onHoverSegment: () => {},
    onFocusSegment: () => {},
    onCommitText: () => Promise.resolve(),
    ...overrides,
  };
}

function textarea(label: string): HTMLTextAreaElement {
  return screen.getByLabelText(label) as HTMLTextAreaElement;
}

function rowOf(element: HTMLElement): HTMLElement {
  const row = element.closest(".pe-transcript-row");
  if (!row) throw new Error("expected element inside a transcript row");
  return row as HTMLElement;
}

describe("PageEditorTranscriptEditor", () => {
  it("renders rows in reading order with the canvas numbers", () => {
    const lines = [
      makeLine("line-c", 3, 90, [{ kind: "ground_truth", text: "third" }]),
      makeLine("line-a", 5, 10, [{ kind: "ground_truth", text: "first" }]),
      makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "second" }]),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    const rows = screen.getAllByRole("listitem");
    expect(rows.map((row) => row.getAttribute("data-line-id"))).toEqual([
      "line-a",
      "line-b",
      "line-c",
    ]);
    const numbers = rows.map(
      (row) => row.querySelector(".pe-transcript-num")?.textContent,
    );
    expect(numbers).toEqual(["3", "1", "2"]);
    expect(textarea("Segment 3 text").value).toBe("first");
  });

  it("seeds model text with the is-model class", () => {
    const lines = [
      makeLine("line-a", 0, 10, [{ kind: "model", text: "model words" }]),
      makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "true words" }]),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    expect(textarea("Segment 1 text").value).toBe("model words");
    expect(rowOf(textarea("Segment 1 text")).className).toContain("is-model");
    expect(rowOf(textarea("Segment 2 text")).className).not.toContain(
      "is-model",
    );
  });

  it("commits trimmed text on Enter and moves focus to the next row", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    fireEvent.change(first, { target: { value: "  edited first  " } });
    fireEvent.keyDown(first, { key: "Enter" });
    await waitFor(() =>
      expect(onCommitText).toHaveBeenCalledWith("line-a", "edited first"),
    );
    await waitFor(() =>
      expect(document.activeElement).toBe(textarea("Segment 2 text")),
    );
  });

  it("does not commit on blur when the draft is unchanged", () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    fireEvent.blur(textarea("Segment 1 text"));
    expect(onCommitText).not.toHaveBeenCalled();
  });

  it("commits on blur when the draft changed", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    fireEvent.change(first, { target: { value: "blur edit" } });
    fireEvent.blur(first);
    await waitFor(() =>
      expect(onCommitText).toHaveBeenCalledWith("line-a", "blur edit"),
    );
  });

  it("shows the error and keeps the draft when the commit rejects", async () => {
    const onCommitText = vi.fn(() => Promise.reject(new Error("save failed")));
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    fireEvent.change(first, { target: { value: "bad edit" } });
    fireEvent.blur(first);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBe("save failed");
    expect(rowOf(first).className).toContain("has-error");
    expect(first.value).toBe("bad edit");
  });

  it("reverts the draft on Escape and keeps focus", () => {
    render(<PageEditorTranscriptEditor {...editorProps()} />);
    const first = textarea("Segment 1 text");
    first.focus();
    fireEvent.change(first, { target: { value: "scratch" } });
    expect(first.value).toBe("scratch");
    fireEvent.keyDown(first, { key: "Escape" });
    expect(first.value).toBe("first");
    expect(document.activeElement).toBe(first);
  });

  it("moves focus with ArrowDown", () => {
    render(<PageEditorTranscriptEditor {...editorProps()} />);
    const first = textarea("Segment 1 text");
    first.focus();
    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(document.activeElement).toBe(textarea("Segment 2 text"));
  });

  it("calls onSelectSegment and onFocusSegment when a row is focused", () => {
    const onSelectSegment = vi.fn();
    const onFocusSegment = vi.fn();
    render(
      <PageEditorTranscriptEditor
        {...editorProps({ onSelectSegment, onFocusSegment })}
      />,
    );
    fireEvent.focus(textarea("Segment 2 text"));
    expect(onSelectSegment).toHaveBeenCalledWith("line-b");
    expect(onFocusSegment).toHaveBeenCalledWith("line-b");
  });

  it("marks the incoming selection without stealing focus", () => {
    const { rerender } = render(
      <PageEditorTranscriptEditor {...editorProps()} />,
    );
    const first = textarea("Segment 1 text");
    first.focus();
    rerender(
      <PageEditorTranscriptEditor
        {...editorProps({ selectedSegmentId: "line-b" })}
      />,
    );
    expect(rowOf(textarea("Segment 2 text")).className).toContain(
      "is-selected",
    );
    expect(document.activeElement).toBe(first);
  });

  it("renders the empty state when there are no lines", () => {
    render(<PageEditorTranscriptEditor {...editorProps({ lines: [] })} />);
    const empty = screen.getByText("No segments on this page yet.");
    expect(empty.className).toContain("pe-transcript-empty");
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);
  });
});
