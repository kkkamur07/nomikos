import { useRef } from "react";
import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";

/** Zoom/pan state shared between two linked viewports. */
export type ViewportState = {
  scale: number;
  positionX: number;
  positionY: number;
};

const POSITION_TOLERANCE_PX = 0.5;
const SCALE_TOLERANCE = 0.001;

function statesMatch(current: ViewportState, next: ViewportState): boolean {
  return (
    Math.abs(current.positionX - next.positionX) <= POSITION_TOLERANCE_PX &&
    Math.abs(current.positionY - next.positionY) <= POSITION_TOLERANCE_PX &&
    Math.abs(current.scale - next.scale) <= SCALE_TOLERANCE
  );
}

/**
 * Keep two react-zoom-pan-pinch viewports in lock step.
 *
 * Attach `leftRef`/`rightRef` to the two `viewportRef`s and forward each
 * side's `onTransformed` to the matching handler. A transform reported by
 * one side is replayed on the other with no animation, unless the other
 * side already matches (which is also what stops the replay echoing back).
 */
export function useLinkedViewports(): {
  leftRef: React.RefObject<ReactZoomPanPinchRef | null>;
  rightRef: React.RefObject<ReactZoomPanPinchRef | null>;
  onLeftTransformed: (state: ViewportState) => void;
  onRightTransformed: (state: ViewportState) => void;
} {
  const leftRef = useRef<ReactZoomPanPinchRef | null>(null);
  const rightRef = useRef<ReactZoomPanPinchRef | null>(null);
  const syncing = useRef(false);

  const propagate = (
    target: React.RefObject<ReactZoomPanPinchRef | null>,
    state: ViewportState,
  ): void => {
    if (syncing.current) return;
    const viewport = target.current;
    if (!viewport) return;
    const current = viewport.instance.transformState;
    if (current && statesMatch(current, state)) return;
    syncing.current = true;
    try {
      viewport.setTransform(state.positionX, state.positionY, state.scale, 0);
    } finally {
      syncing.current = false;
    }
  };

  return {
    leftRef,
    rightRef,
    onLeftTransformed: (state) => propagate(rightRef, state),
    onRightTransformed: (state) => propagate(leftRef, state),
  };
}
