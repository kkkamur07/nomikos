/**
 * Unsaved row drafts are kept in the tab, and leaving the page while any
 * exists asks first: closing or reloading the tab, and in-app links like Back.
 */
import { fireEvent, render, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  confirmLeaveWithUnsavedRows,
  hasUnsavedTranscriptDrafts,
  readStoredDraft,
  useUnsavedTranscriptGuard,
} from "./PageEditorTranscriptEditor";

function storeDraft(lineId: string, text: string) {
  sessionStorage.setItem(
    `nomikos_transcript_draft:${lineId}`,
    JSON.stringify({ text, base: "" }),
  );
}

afterEach(() => {
  sessionStorage.clear();
  vi.restoreAllMocks();
});

describe("unsaved transcript drafts", () => {
  it("counts only drafts for the lines asked about", () => {
    storeDraft("line-a", "draft");
    expect(hasUnsavedTranscriptDrafts(["line-a", "line-b"])).toBe(true);
    expect(hasUnsavedTranscriptDrafts(["line-b"])).toBe(false);
  });

  it("ignores a stored value it cannot read", () => {
    sessionStorage.setItem("nomikos_transcript_draft:line-a", "{not json");
    expect(readStoredDraft("line-a")).toBeNull();
  });

  it("confirms only when there is something to lose", () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    expect(confirmLeaveWithUnsavedRows(["line-a"])).toBe(true);
    expect(confirm).not.toHaveBeenCalled();

    storeDraft("line-a", "draft");
    expect(confirmLeaveWithUnsavedRows(["line-a"])).toBe(false);
    expect(confirm).toHaveBeenCalledTimes(1);
  });

  it("asks before the tab unloads while a draft is unsaved", () => {
    renderHook(() => useUnsavedTranscriptGuard(["line-a"]));
    const clean = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(clean);
    expect(clean.defaultPrevented).toBe(false);

    storeDraft("line-a", "draft");
    const dirty = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(dirty);
    expect(dirty.defaultPrevented).toBe(true);
  });

  it("holds an in-app link such as Back when the user stays", () => {
    function Guarded() {
      useUnsavedTranscriptGuard(["line-a"]);
      return <a href="/projects/p/documents/d">Back</a>;
    }
    const { getByText } = render(<Guarded />);
    storeDraft("line-a", "draft");
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const followed = vi.fn();
    getByText("Back").addEventListener("click", followed);

    const event = new MouseEvent("click", { bubbles: true, cancelable: true });
    fireEvent(getByText("Back"), event);

    expect(event.defaultPrevented).toBe(true);
    expect(followed).not.toHaveBeenCalled();
  });
});
