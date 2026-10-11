import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../api/client";
import {
  centerRowInScroller,
  PageEditorTranscriptEditor,
} from "./PageEditorTranscriptEditor";
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
    onSelectSegment: () => {},
    onHoverSegment: () => {},
    onFocusSegment: () => {},
    onCommitText: () => Promise.resolve(),
    onClose: () => {},
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
  it("renders rows in stored segment order with the canvas numbers", () => {
    const lines = [
      makeLine("line-c", 3, 90, [{ kind: "ground_truth", text: "third" }]),
      makeLine("line-a", 5, 10, [{ kind: "ground_truth", text: "first" }]),
      makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "second" }]),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    const rows = screen.getAllByRole("listitem");
    expect(rows.map((row) => row.getAttribute("data-line-id"))).toEqual([
      "line-b",
      "line-c",
      "line-a",
    ]);
    const numbers = rows.map(
      (row) => row.querySelector(".pe-transcript-num")?.textContent,
    );
    expect(numbers).toEqual(["1", "2", "3"]);
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
    expect(alert.textContent).toBe("Not saved: save failed");
    expect(rowOf(first).className).toContain("has-error");
    expect(rowOf(first).getAttribute("data-status")).toBe("unsaved");
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

  it("renders an empty pane with no placeholder when there are no lines", () => {
    const { container } = render(
      <PageEditorTranscriptEditor {...editorProps({ lines: [] })} />,
    );
    const list = screen.getByRole("list");
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);
    expect(list.querySelector("p")).toBeNull();
    expect(list.textContent).toBe("");
    // The header stays: title and the close button.
    expect(container.querySelector(".pe-transcript-header")).not.toBeNull();
    expect(
      screen.getByRole("button", { name: "Hide transcript pane" }),
    ).toBeTruthy();
  });

  it("marks a row dirty while edited and saved after the commit", async () => {
    let resolveCommit: () => void = () => {};
    const onCommitText = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveCommit = resolve;
        }),
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    expect(rowOf(first).getAttribute("data-status")).toBeNull();
    fireEvent.change(first, { target: { value: "draft" } });
    expect(rowOf(first).className).toContain("is-dirty");
    expect(rowOf(first).getAttribute("data-status")).toBe("dirty");
    expect(screen.getByLabelText("Unsaved changes")).toBeTruthy();
    fireEvent.keyDown(first, { key: "Enter", ctrlKey: true });
    await waitFor(() =>
      expect(rowOf(first).getAttribute("data-status")).toBe("saving"),
    );
    resolveCommit();
    await waitFor(() =>
      expect(rowOf(first).getAttribute("data-status")).toBe("saved"),
    );
    expect(rowOf(first).className).not.toContain("is-dirty");
  });

  it("commits without advancing on Ctrl+Enter and Cmd+Enter", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    first.focus();
    fireEvent.change(first, { target: { value: "stay here" } });
    fireEvent.keyDown(first, { key: "Enter", metaKey: true });
    await waitFor(() =>
      expect(onCommitText).toHaveBeenCalledWith("line-a", "stay here"),
    );
    expect(document.activeElement).toBe(first);
  });

  it("moves rows with Tab and Shift+Tab and leaves the list past the ends", () => {
    render(<PageEditorTranscriptEditor {...editorProps()} />);
    const first = textarea("Segment 1 text");
    const second = textarea("Segment 2 text");
    first.focus();
    fireEvent.keyDown(first, { key: "Tab" });
    expect(document.activeElement).toBe(second);
    fireEvent.keyDown(second, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(first);
    // Shift+Tab on the first row is not swallowed, so focus can leave.
    const event = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    first.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it("sets right-to-left direction per row from the row's own text", () => {
    const lines = [
      makeLine("line-a", 0, 10, [
        { kind: "ground_truth", text: "ܫܠܡܐ ܥܠܝܟܘܢ" },
      ]),
      makeLine("line-b", 1, 50, [
        { kind: "ground_truth", text: "καὶ ἐγένετο" },
      ]),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    const syriac = textarea("Segment 1 text");
    const greek = textarea("Segment 2 text");
    expect(syriac.getAttribute("dir")).toBe("rtl");
    expect(syriac.style.textAlign).toBe("right");
    expect(syriac.getAttribute("data-script")).toBe("syriac");
    expect(greek.getAttribute("dir")).toBe("ltr");
    expect(greek.style.textAlign).toBe("left");
    fireEvent.change(greek, { target: { value: "ܐܒܐ" } });
    expect(greek.getAttribute("dir")).toBe("rtl");
  });

  it("gives empty rows the page's dominant direction", () => {
    const lines = [
      makeLine("line-a", 0, 10, [{ kind: "ground_truth", text: "ܫܠܡܐ" }]),
      makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "ܥܠܝܟܘܢ" }]),
      makeLine("line-c", 2, 90, []),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    expect(textarea("Segment 3 text").getAttribute("dir")).toBe("rtl");
    expect(
      screen
        .getAllByRole("listitem")
        .every((row) => row.closest("[dir]") === null),
    ).toBe(true);
  });

  it("keeps rows mounted and an unsaved draft when a segment is added or deleted", () => {
    const base = editorProps();
    const { rerender } = render(<PageEditorTranscriptEditor {...base} />);
    const second = textarea("Segment 2 text");
    fireEvent.change(second, { target: { value: "half typed" } });
    const added = [...base.lines, makeLine("line-new", 2, 90, [])];
    rerender(<PageEditorTranscriptEditor {...base} lines={added} />);
    expect(textarea("Segment 2 text")).toBe(second);
    expect(second.value).toBe("half typed");
    rerender(
      <PageEditorTranscriptEditor
        {...base}
        lines={added.filter((line) => line.id !== "line-a")}
      />,
    );
    expect(textarea("Segment 1 text")).toBe(second);
    expect(second.value).toBe("half typed");
    expect(rowOf(second).getAttribute("data-status")).toBe("dirty");
  });

  it("sizes rows by mask height only when the option is on", () => {
    const short = makeLine("line-a", 0, 10, [
      { kind: "ground_truth", text: "a" },
    ]);
    const tall = {
      ...makeLine("line-b", 1, 50, [{ kind: "ground_truth", text: "b" }]),
      mask: [
        [10, 50],
        [110, 50],
        [110, 110],
        [10, 110],
      ],
    } as unknown as LineResponse;
    const { rerender } = render(
      <PageEditorTranscriptEditor {...editorProps({ lines: [short, tall] })} />,
    );
    expect(rowOf(textarea("Segment 2 text")).style.fontSize).toBe("16px");
    rerender(
      <PageEditorTranscriptEditor
        {...editorProps({ lines: [short, tall], scaleFontToMask: true })}
      />,
    );
    const shortSize = parseFloat(
      rowOf(textarea("Segment 1 text")).style.fontSize,
    );
    const tallSize = parseFloat(
      rowOf(textarea("Segment 2 text")).style.fontSize,
    );
    expect(tallSize).toBeGreaterThan(shortSize);
  });

  it("shows an icon-only close button that calls onClose", () => {
    const onClose = vi.fn();
    render(<PageEditorTranscriptEditor {...editorProps({ onClose })} />);
    const close = screen.getByRole("button", { name: "Hide transcript pane" });
    expect(close).toHaveAttribute("title", "Hide transcript pane");
    expect(close.textContent).toBe("");
    fireEvent.click(close);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("centres a row in the pane by setting the pane's own scrollTop", () => {
    const scroller = document.createElement("div");
    const row = document.createElement("div");
    // Pane 400px tall at y=100, content 2000px; the row is 30px tall and sits
    // 600px below the pane top at the current scroll of 50.
    Object.defineProperty(scroller, "clientHeight", { value: 400 });
    Object.defineProperty(scroller, "scrollHeight", { value: 2000 });
    scroller.scrollTop = 50;
    scroller.getBoundingClientRect = () =>
      ({ top: 100, height: 400 }) as DOMRect;
    row.getBoundingClientRect = () => ({ top: 700, height: 30 }) as DOMRect;

    centerRowInScroller(row, scroller);

    // 50 + 600 - (400 - 30) / 2 = 465: the row's middle at the pane's middle.
    expect(scroller.scrollTop).toBe(465);
  });

  it("clamps the centring scroll at the top and bottom of the pane", () => {
    const scroller = document.createElement("div");
    const row = document.createElement("div");
    Object.defineProperty(scroller, "clientHeight", { value: 400 });
    Object.defineProperty(scroller, "scrollHeight", { value: 1000 });
    scroller.getBoundingClientRect = () => ({ top: 0, height: 400 }) as DOMRect;

    row.getBoundingClientRect = () => ({ top: 20, height: 30 }) as DOMRect;
    centerRowInScroller(row, scroller);
    expect(scroller.scrollTop).toBe(0);

    row.getBoundingClientRect = () => ({ top: 990, height: 30 }) as DOMRect;
    centerRowInScroller(row, scroller);
    expect(scroller.scrollTop).toBe(600);
  });

  it("centres a selection from the canvas without scrollIntoView", () => {
    const scrollIntoView = vi.fn();
    const original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = scrollIntoView;
    const props = editorProps();
    const { container, rerender } = render(
      <PageEditorTranscriptEditor {...props} />,
    );
    const scroller = container.querySelector(".pe-transcript") as HTMLElement;
    Object.defineProperty(scroller, "clientHeight", { value: 100 });
    Object.defineProperty(scroller, "scrollHeight", { value: 1000 });
    scroller.getBoundingClientRect = () => ({ top: 0, height: 100 }) as DOMRect;
    rowOf(textarea("Segment 2 text")).getBoundingClientRect = () =>
      ({ top: 500, height: 20 }) as DOMRect;

    rerender(
      <PageEditorTranscriptEditor {...props} selectedSegmentId="line-b" />,
    );

    expect(scroller.scrollTop).toBe(460);
    expect(scrollIntoView).not.toHaveBeenCalled();
    Element.prototype.scrollIntoView = original;
  });

  it("centres the next row when the selection moves by keyboard", () => {
    const { container } = render(
      <PageEditorTranscriptEditor {...editorProps()} />,
    );
    const scroller = container.querySelector(".pe-transcript") as HTMLElement;
    Object.defineProperty(scroller, "clientHeight", { value: 100 });
    Object.defineProperty(scroller, "scrollHeight", { value: 1000 });
    scroller.getBoundingClientRect = () => ({ top: 0, height: 100 }) as DOMRect;
    rowOf(textarea("Segment 2 text")).getBoundingClientRect = () =>
      ({ top: 300, height: 20 }) as DOMRect;
    const focus = vi.spyOn(textarea("Segment 2 text"), "focus");

    fireEvent.keyDown(textarea("Segment 1 text"), { key: "ArrowDown" });

    expect(focus).toHaveBeenCalledWith({ preventScroll: true });
    expect(scroller.scrollTop).toBe(260);
  });
});
