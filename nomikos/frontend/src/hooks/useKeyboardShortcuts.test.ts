import { fireEvent, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useKeyboardShortcuts } from "./useKeyboardShortcuts";

function focused<K extends keyof HTMLElementTagNameMap>(tag: K) {
  const element = document.body.appendChild(document.createElement(tag));
  element.focus();
  return element;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("useKeyboardShortcuts", () => {
  it("runs the handler bound to a key pressed on the page", () => {
    const onDrawBox = vi.fn();
    const onUndo = vi.fn();
    const onRedo = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onDrawBox, onUndo, onRedo }));

    fireEvent.keyDown(window, { key: "B" });
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    fireEvent.keyDown(window, { key: "z", metaKey: true, shiftKey: true });
    fireEvent.keyDown(window, { key: "y", ctrlKey: true });

    expect(onDrawBox).toHaveBeenCalledTimes(1);
    expect(onUndo).toHaveBeenCalledTimes(1);
    expect(onRedo).toHaveBeenCalledTimes(2);
  });

  it("does not treat Ctrl+B as the box tool", () => {
    const onDrawBox = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onDrawBox }));

    const notPrevented = fireEvent.keyDown(window, { key: "b", ctrlKey: true });

    expect(onDrawBox).not.toHaveBeenCalled();
    expect(notPrevented).toBe(true);
  });

  it.each(["input", "textarea", "select"] as const)(
    "leaves Delete, Backspace and letters to a focused <%s>",
    (tag) => {
      const onDelete = vi.fn();
      const onDrawPolygon = vi.fn();
      renderHook(() => useKeyboardShortcuts({ onDelete, onDrawPolygon }));
      const control = focused(tag);

      const results = ["Backspace", "Delete", "p"].map((key) =>
        fireEvent.keyDown(control, { key }),
      );

      expect(onDelete).not.toHaveBeenCalled();
      expect(onDrawPolygon).not.toHaveBeenCalled();
      expect(results).toEqual([true, true, true]);
    },
  );

  it("leaves keys to a contenteditable region and to a listbox", () => {
    const onDelete = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onDelete }));
    const editable = focused("div");
    editable.setAttribute("contenteditable", "true");
    const listbox = focused("ul");
    listbox.setAttribute("role", "listbox");

    fireEvent.keyDown(editable, { key: "Delete" });
    fireEvent.keyDown(listbox, { key: "Backspace" });

    expect(onDelete).not.toHaveBeenCalled();
  });

  it("deletes when focus is on a button, which does not take text", () => {
    const onDelete = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onDelete }));

    fireEvent.keyDown(focused("button"), { key: "Delete" });

    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it("keeps the browser default for keys with no handler", () => {
    renderHook(() => useKeyboardShortcuts({ onDelete: undefined }));

    const keys = ["ArrowDown", "ArrowUp", "Backspace", "v", "PageDown"];
    const results = keys.map((key) => fireEvent.keyDown(window, { key }));

    expect(results).toEqual(keys.map(() => true));
  });

  it("prevents the default only for keys it handles, never for Enter", () => {
    const onNextPage = vi.fn();
    const onEnter = vi.fn();
    renderHook(() => useKeyboardShortcuts({ onNextPage, onEnter }));

    const pageDown = fireEvent.keyDown(window, { key: "PageDown" });
    const enter = fireEvent.keyDown(window, { key: "Enter" });

    expect([pageDown, enter]).toEqual([false, true]);
    expect(onNextPage).toHaveBeenCalledTimes(1);
    expect(onEnter).toHaveBeenCalledTimes(1);
  });

  it("calls the handler from the latest render", () => {
    const first = vi.fn();
    const second = vi.fn();
    const view = renderHook(
      ({ onEscape }) => useKeyboardShortcuts({ onEscape }),
      { initialProps: { onEscape: first } },
    );
    view.rerender({ onEscape: second });

    fireEvent.keyDown(window, { key: "Escape" });

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});
