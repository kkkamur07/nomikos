import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useStripResize } from "./useStripResize";

const INSET = "--nk-bottom-inset";

afterEach(() => {
  document.documentElement.style.removeProperty(INSET);
});

describe("useStripResize bottom inset", () => {
  it("publishes the strip height and removes it on unmount", () => {
    const { result, unmount } = renderHook(() => useStripResize(240));

    expect(result.current.height).toBe(240);
    expect(document.documentElement.style.getPropertyValue(INSET)).toBe(
      "240px",
    );

    act(() => {
      result.current.setHeight(300);
    });
    expect(document.documentElement.style.getPropertyValue(INSET)).toBe(
      "300px",
    );

    unmount();
    expect(document.documentElement.style.getPropertyValue(INSET)).toBe("");
  });
});
