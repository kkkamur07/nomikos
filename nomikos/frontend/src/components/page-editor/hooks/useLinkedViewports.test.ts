import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";

import { useLinkedViewports, type ViewportState } from "./useLinkedViewports";

function fakeViewport(
  initial: ViewportState,
  onSetTransform?: (state: ViewportState) => void,
): {
  ref: ReactZoomPanPinchRef;
  setTransform: ReturnType<typeof vi.fn>;
} {
  const setTransform = vi.fn(
    (positionX: number, positionY: number, scale: number) => {
      onSetTransform?.({ positionX, positionY, scale });
    },
  );
  const ref = {
    state: { ...initial, previousScale: initial.scale },
    instance: {
      transformState: { ...initial, previousScale: initial.scale },
    },
    setTransform,
  } as unknown as ReactZoomPanPinchRef;
  return { ref, setTransform };
}

describe("useLinkedViewports", () => {
  it("propagates a left transform to the right with no animation", () => {
    const { result } = renderHook(() => useLinkedViewports());
    const left = fakeViewport({ scale: 1, positionX: 0, positionY: 0 });
    const right = fakeViewport({ scale: 1, positionX: 0, positionY: 0 });
    result.current.leftRef.current = left.ref;
    result.current.rightRef.current = right.ref;

    result.current.onLeftTransformed({
      scale: 2,
      positionX: 10,
      positionY: 20,
    });

    expect(right.setTransform).toHaveBeenCalledTimes(1);
    expect(right.setTransform).toHaveBeenCalledWith(10, 20, 2, 0);
    expect(left.setTransform).not.toHaveBeenCalled();
  });

  it("skips the replay when the other side already matches", () => {
    const { result } = renderHook(() => useLinkedViewports());
    const left = fakeViewport({ scale: 1, positionX: 0, positionY: 0 });
    const state = { scale: 2, positionX: 10, positionY: 20 };
    const right = fakeViewport(state);
    result.current.leftRef.current = left.ref;
    result.current.rightRef.current = right.ref;

    result.current.onLeftTransformed(state);

    expect(right.setTransform).not.toHaveBeenCalled();
  });

  it("does not bounce a propagated call back to its source", () => {
    const { result } = renderHook(() => useLinkedViewports());
    const left = fakeViewport({ scale: 1, positionX: 0, positionY: 0 });
    // The library echoing the replay back while it is still being applied.
    const right = fakeViewport({ scale: 1, positionX: 0, positionY: 0 }, (s) =>
      result.current.onRightTransformed(s),
    );
    result.current.leftRef.current = left.ref;
    result.current.rightRef.current = right.ref;

    result.current.onLeftTransformed({
      scale: 2,
      positionX: 10,
      positionY: 20,
    });

    expect(right.setTransform).toHaveBeenCalledTimes(1);
    expect(left.setTransform).not.toHaveBeenCalled();
  });
});
