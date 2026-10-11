import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useFeedbackButton } from "./useFeedbackButton";

describe("useFeedbackButton", () => {
  it("opens and closes the dialog", () => {
    const { result } = renderHook(() => useFeedbackButton());

    expect(result.current.dialogOpen).toBe(false);
    act(() => result.current.openDialog());
    expect(result.current.dialogOpen).toBe(true);
    act(() => result.current.closeDialog());
    expect(result.current.dialogOpen).toBe(false);
  });

  it("shows Thanks for two seconds after a send", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() => useFeedbackButton());

      act(() => result.current.openDialog());
      act(() => result.current.handleSent());

      expect(result.current.dialogOpen).toBe(false);
      expect(result.current.thanks).toBe(true);
      act(() => vi.advanceTimersByTime(2000));
      expect(result.current.thanks).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});
