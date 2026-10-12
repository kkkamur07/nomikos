/**
 * A transcript row must never lose or overwrite text: not when the user keeps
 * typing during a save, not when the page's lines reload under a focused or
 * dirty row, and not when someone else saved the same line in between.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../api/client";
import { ApiError } from "../../api/errors";
import { PageEditorTranscriptEditor } from "./PageEditorTranscriptEditor";
import type { PageEditorTranscriptEditorProps } from "./PageEditorTranscriptEditor";

function makeLine(
  id: string,
  order: number,
  texts: { kind: "ground_truth" | "model"; text: string }[],
): LineResponse {
  const top = 10 + order * 40;
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
      transcription_id: `${entry.kind}-layer`,
      transcription_kind: entry.kind,
      text: entry.text,
      confidence: null,
    })),
  } as unknown as LineResponse;
}

function twoLines(first = "first", second = "second"): LineResponse[] {
  return [
    makeLine("line-a", 0, [{ kind: "ground_truth", text: first }]),
    makeLine("line-b", 1, [{ kind: "ground_truth", text: second }]),
  ];
}

function editorProps(
  overrides: Partial<PageEditorTranscriptEditorProps> = {},
): PageEditorTranscriptEditorProps {
  return {
    lines: twoLines(),
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
  return element.closest(".pe-transcript-row") as HTMLElement;
}

function deferred() {
  let resolve: () => void = () => {};
  let reject: (error: unknown) => void = () => {};
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function settle() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

afterEach(() => {
  window.sessionStorage.clear();
});

describe("typing while a row saves (F1)", () => {
  it("keeps text typed during the save and sends it with a second save", async () => {
    const saves = [deferred(), deferred()];
    const onCommitText = vi.fn<PageEditorTranscriptEditorProps["onCommitText"]>(
      () => saves[onCommitText.mock.calls.length - 1].promise,
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    row.focus();

    fireEvent.change(row, { target: { value: "first AAA" } });
    fireEvent.keyDown(row, { key: "Enter", ctrlKey: true });
    expect(onCommitText).toHaveBeenCalledTimes(1);
    expect(onCommitText.mock.calls[0][1]).toBe("first AAA");

    // Typed and committed again while the first save is still out.
    fireEvent.change(row, { target: { value: "first AAA BBB" } });
    fireEvent.keyDown(row, { key: "Enter", ctrlKey: true });
    expect(onCommitText).toHaveBeenCalledTimes(1);

    await act(async () => saves[0].resolve());
    expect(row.value).toBe("first AAA BBB");
    expect(onCommitText).toHaveBeenCalledTimes(2);
    expect(onCommitText.mock.calls[1][1]).toBe("first AAA BBB");
    expect(rowOf(row).getAttribute("data-status")).toBe("saving");

    await act(async () => saves[1].resolve());
    expect(row.value).toBe("first AAA BBB");
    expect(rowOf(row).getAttribute("data-status")).toBe("saved");
  });

  it("leaves the row dirty, not saved, when text was typed after the save left", async () => {
    const save = deferred();
    const onCommitText = vi.fn(() => save.promise);
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    row.focus();

    fireEvent.change(row, { target: { value: "first AAA" } });
    fireEvent.keyDown(row, { key: "Enter", ctrlKey: true });
    fireEvent.change(row, { target: { value: "first AAA BBB" } });

    await act(async () => save.resolve());
    expect(row.value).toBe("first AAA BBB");
    expect(rowOf(row).getAttribute("data-status")).toBe("dirty");
    expect(onCommitText).toHaveBeenCalledTimes(1);
  });

  it("does not report a commit asked for during a save as done before it is", async () => {
    const saves = [deferred(), deferred()];
    const onCommitText = vi.fn<PageEditorTranscriptEditorProps["onCommitText"]>(
      () => saves[onCommitText.mock.calls.length - 1].promise,
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const first = textarea("Segment 1 text");
    first.focus();

    fireEvent.change(first, { target: { value: "first AAA" } });
    fireEvent.keyDown(first, { key: "Enter", ctrlKey: true });
    fireEvent.change(first, { target: { value: "first AAA BBB" } });
    // Plain Enter saves and moves on only once the save holds the latest text.
    fireEvent.keyDown(first, { key: "Enter" });
    await settle();
    expect(document.activeElement).toBe(first);

    await act(async () => saves[0].resolve());
    expect(document.activeElement).toBe(first);
    await act(async () => saves[1].resolve());
    await settle();
    expect(document.activeElement).toBe(textarea("Segment 2 text"));
  });
});

describe("lines reloading under a row (F2, F8)", () => {
  it("adopts new text into a focused row the user has not typed in", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    const props = editorProps({ onCommitText });
    const { rerender } = render(<PageEditorTranscriptEditor {...props} />);
    const row = textarea("Segment 1 text");
    row.focus();

    rerender(
      <PageEditorTranscriptEditor
        {...props}
        lines={twoLines("text from the model run")}
      />,
    );
    expect(row.value).toBe("text from the model run");

    fireEvent.blur(row);
    await settle();
    expect(onCommitText).not.toHaveBeenCalled();
    expect(row.value).toBe("text from the model run");
  });

  it("keeps a dirty row's draft when the lines reload", () => {
    const props = editorProps();
    const { rerender } = render(<PageEditorTranscriptEditor {...props} />);
    const row = textarea("Segment 1 text");
    fireEvent.change(row, { target: { value: "my unsaved words" } });

    rerender(
      <PageEditorTranscriptEditor {...props} lines={twoLines("reloaded")} />,
    );

    expect(row.value).toBe("my unsaved words");
    expect(rowOf(row).getAttribute("data-status")).toBe("dirty");
  });

  it("keeps a focused row's draft when the lines reload", () => {
    const props = editorProps();
    const { rerender } = render(<PageEditorTranscriptEditor {...props} />);
    const row = textarea("Segment 1 text");
    row.focus();
    fireEvent.change(row, { target: { value: "typing right now" } });

    rerender(
      <PageEditorTranscriptEditor {...props} lines={twoLines("reloaded")} />,
    );

    expect(row.value).toBe("typing right now");
    expect(document.activeElement).toBe(row);
  });

  it("does not move the pane when the selection lands on the focused row", () => {
    const props = editorProps();
    const { container, rerender } = render(
      <PageEditorTranscriptEditor {...props} />,
    );
    const scroller = container.querySelector(".pe-transcript") as HTMLElement;
    Object.defineProperty(scroller, "clientHeight", { value: 100 });
    Object.defineProperty(scroller, "scrollHeight", { value: 1000 });
    scroller.getBoundingClientRect = () => ({ top: 0, height: 100 }) as DOMRect;
    const row = textarea("Segment 2 text");
    rowOf(row).getBoundingClientRect = () =>
      ({ top: 500, height: 20 }) as DOMRect;
    scroller.scrollTop = 120;
    row.focus();

    rerender(
      <PageEditorTranscriptEditor {...props} selectedSegmentId="line-b" />,
    );

    expect(scroller.scrollTop).toBe(120);
  });
});

describe("saving over someone else's text (F3)", () => {
  it("sends the Ground truth text the row was seeded with", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    const lines = [
      makeLine("line-a", 0, [{ kind: "ground_truth", text: "first" }]),
      makeLine("line-b", 1, [{ kind: "model", text: "model guess" }]),
    ];
    render(
      <PageEditorTranscriptEditor {...editorProps({ lines, onCommitText })} />,
    );

    fireEvent.change(textarea("Segment 1 text"), {
      target: { value: "first edited" },
    });
    fireEvent.blur(textarea("Segment 1 text"));
    fireEvent.change(textarea("Segment 2 text"), {
      target: { value: "model guess fixed" },
    });
    fireEvent.blur(textarea("Segment 2 text"));
    await settle();

    expect(onCommitText).toHaveBeenNthCalledWith(
      1,
      "line-a",
      "first edited",
      "first",
    );
    // A row seeded from model text has no Ground truth behind it yet.
    expect(onCommitText).toHaveBeenNthCalledWith(
      2,
      "line-b",
      "model guess fixed",
      "",
    );
  });

  it("says the row changed elsewhere on a 409 and reloads it on request", async () => {
    const onCommitText = vi.fn(() =>
      Promise.reject(new ApiError("conflict", 409)),
    );
    const onReloadLine = vi.fn(() =>
      Promise.resolve(
        makeLine("line-a", 0, [{ kind: "ground_truth", text: "FROM_A" }]),
      ),
    );
    render(
      <PageEditorTranscriptEditor
        {...editorProps({ onCommitText, onReloadLine })}
      />,
    );
    const row = textarea("Segment 1 text");
    fireEvent.change(row, { target: { value: "FROM_B" } });
    fireEvent.blur(row);
    await settle();

    expect(screen.getByRole("alert").textContent).toContain(
      "Changed elsewhere.",
    );
    expect(row.value).toBe("FROM_B");

    fireEvent.click(screen.getByRole("button", { name: "Reload row" }));
    await settle();

    expect(onReloadLine).toHaveBeenCalledWith("line-a");
    expect(row.value).toBe("FROM_A");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(rowOf(row).getAttribute("data-status")).toBeNull();
  });

  it("keeps the draft and names the deletion on a 404", async () => {
    const onCommitText = vi.fn(() =>
      Promise.reject(new ApiError("Resource not found", 404)),
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    fireEvent.change(row, { target: { value: "orphaned words" } });
    fireEvent.blur(row);
    await settle();

    expect(screen.getByRole("alert").textContent).toBe(
      "This segment was deleted elsewhere. Your text is kept below.",
    );
    expect(row.value).toBe("orphaned words");
  });
});

describe("signing out mid-edit (F4)", () => {
  it("says the session expired and keeps the text on a 401", async () => {
    const onCommitText = vi.fn(() =>
      Promise.reject(new ApiError("Unauthorized", 401)),
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    fireEvent.change(row, { target: { value: "first X401" } });
    fireEvent.keyDown(row, { key: "Enter" });
    await settle();

    expect(screen.getByRole("alert").textContent).toBe(
      "Session expired, sign in again. Your text is kept.",
    );
    expect(row.value).toBe("first X401");
  });

  it("restores an unsaved draft after the page reloads", async () => {
    const onCommitText = vi.fn(() => Promise.resolve());
    const first = render(<PageEditorTranscriptEditor {...editorProps()} />);
    fireEvent.change(textarea("Segment 1 text"), {
      target: { value: "first, then a reload" },
    });
    first.unmount();

    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    expect(row.value).toBe("first, then a reload");
    expect(rowOf(row).getAttribute("data-status")).toBe("dirty");

    fireEvent.blur(row);
    await settle();
    expect(onCommitText).toHaveBeenCalledWith(
      "line-a",
      "first, then a reload",
      "first",
    );
  });

  it("forgets the stored draft once it is saved", async () => {
    const first = render(<PageEditorTranscriptEditor {...editorProps()} />);
    const row = textarea("Segment 1 text");
    fireEvent.change(row, { target: { value: "first saved" } });
    fireEvent.blur(row);
    await settle();
    first.unmount();

    render(<PageEditorTranscriptEditor {...editorProps()} />);
    expect(textarea("Segment 1 text").value).toBe("first");
  });
});

describe("keyboard into and out of the pane (F10)", () => {
  it("focuses the selected row when opened from the keyboard", () => {
    render(
      <PageEditorTranscriptEditor
        {...editorProps({
          selectedSegmentId: "line-b",
          takeFocusOnOpen: () => true,
        })}
      />,
    );
    expect(document.activeElement).toBe(textarea("Segment 2 text"));
  });

  it("does not take focus when opened another way", () => {
    render(
      <PageEditorTranscriptEditor
        {...editorProps({
          selectedSegmentId: "line-b",
          takeFocusOnOpen: () => false,
        })}
      />,
    );
    expect(document.activeElement).not.toBe(textarea("Segment 2 text"));
  });

  it("reverts on the first Escape and returns to the canvas on the second", () => {
    const onReturnToCanvas = vi.fn();
    render(
      <PageEditorTranscriptEditor {...editorProps({ onReturnToCanvas })} />,
    );
    const row = textarea("Segment 1 text");
    row.focus();
    fireEvent.change(row, { target: { value: "first typo" } });

    fireEvent.keyDown(row, { key: "Escape" });
    expect(row.value).toBe("first");
    expect(onReturnToCanvas).not.toHaveBeenCalled();

    fireEvent.keyDown(row, { key: "Escape" });
    expect(onReturnToCanvas).toHaveBeenCalledWith("line-a");
  });
});

describe("row direction (F11)", () => {
  const syriacPage = () => [
    makeLine("line-a", 0, [{ kind: "ground_truth", text: "ܫܠܡܐ ܥܠܝܟܘܢ" }]),
    makeLine("line-b", 1, [{ kind: "ground_truth", text: "12." }]),
    makeLine("line-c", 2, [{ kind: "ground_truth", text: "" }]),
  ];

  it("lays a digits-only row out by its own content on a right-to-left page", () => {
    render(
      <PageEditorTranscriptEditor {...editorProps({ lines: syriacPage() })} />,
    );
    expect(textarea("Segment 1 text").getAttribute("dir")).toBe("rtl");
    expect(textarea("Segment 2 text").getAttribute("dir")).toBe("auto");
  });

  it("keeps a row's direction while it is typed in and settles it on blur", () => {
    render(
      <PageEditorTranscriptEditor {...editorProps({ lines: syriacPage() })} />,
    );
    const empty = textarea("Segment 3 text");
    expect(empty.getAttribute("dir")).toBe("rtl");
    empty.focus();
    fireEvent.focus(empty);
    fireEvent.change(empty, { target: { value: "abc" } });
    expect(empty.getAttribute("dir")).toBe("rtl");

    fireEvent.blur(empty);
    expect(empty.getAttribute("dir")).toBe("ltr");
  });

  it("marks an Armenian row with its script for the font rule", () => {
    const lines = [
      makeLine("line-a", 0, [{ kind: "ground_truth", text: "Ա Բ Գ Դ ԺԱ" }]),
    ];
    render(<PageEditorTranscriptEditor {...editorProps({ lines })} />);
    expect(textarea("Segment 1 text").getAttribute("data-script")).toBe(
      "armenian",
    );
  });
});

describe("a row in conflict (F3)", () => {
  it("does not resend the refused text when the row loses focus", async () => {
    const onCommitText = vi.fn(() =>
      Promise.reject(new ApiError("conflict", 409)),
    );
    render(<PageEditorTranscriptEditor {...editorProps({ onCommitText })} />);
    const row = textarea("Segment 1 text");
    row.focus();
    fireEvent.change(row, { target: { value: "FROM_B" } });
    fireEvent.keyDown(row, { key: "Enter", ctrlKey: true });
    await settle();
    expect(onCommitText).toHaveBeenCalledTimes(1);

    // Clicking "Reload row" blurs the textarea first.
    fireEvent.blur(row);
    await settle();

    expect(onCommitText).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("alert").textContent).toContain(
      "Changed elsewhere.",
    );
  });
});
