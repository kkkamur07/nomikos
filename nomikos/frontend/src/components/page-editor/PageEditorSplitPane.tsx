import { useCallback, useEffect, useRef, useState } from "react";
import type { JSX } from "react";

export const SPLIT_RATIO_MIN = 0.2;
export const SPLIT_RATIO_MAX = 0.8;
const SPLIT_RATIO_DEFAULT = 0.5;

function clampSplitRatio(value: number): number {
  return Math.min(SPLIT_RATIO_MAX, Math.max(SPLIT_RATIO_MIN, value));
}

export type PageEditorSplitPaneProps = {
  /** Width share of the left pane, clamped to [0.2, 0.8]. */
  ratio: number;
  onRatioChange: (ratio: number) => void;
  left: React.ReactNode;
  right: React.ReactNode;
  leftLabel?: string;
  rightLabel?: string;
};

export function PageEditorSplitPane({
  ratio,
  onRatioChange,
  left,
  right,
  leftLabel = "Manuscript",
  rightLabel = "Transcript",
}: PageEditorSplitPaneProps): JSX.Element {
  const clamped = clampSplitRatio(ratio);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const [dragging, setDragging] = useState(false);
  const onRatioChangeRef = useRef(onRatioChange);
  onRatioChangeRef.current = onRatioChange;
  const clampedRef = useRef(clamped);
  clampedRef.current = clamped;

  const ratioFromClientX = useCallback((clientX: number): number => {
    const container = containerRef.current;
    if (!container) return clampedRef.current;
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0) return clampedRef.current;
    return clampSplitRatio((clientX - rect.left) / rect.width);
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const handleMove = (event: PointerEvent) => {
      if (!draggingRef.current) return;
      onRatioChangeRef.current(ratioFromClientX(event.clientX));
    };
    const handleUp = () => {
      draggingRef.current = false;
      setDragging(false);
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointercancel", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointercancel", handleUp);
    };
  }, [dragging, ratioFromClientX]);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      onRatioChange(clampSplitRatio(clamped - step));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      onRatioChange(clampSplitRatio(clamped + step));
    } else if (event.key === "Home") {
      event.preventDefault();
      onRatioChange(SPLIT_RATIO_MIN);
    } else if (event.key === "End") {
      event.preventDefault();
      onRatioChange(SPLIT_RATIO_MAX);
    }
  };

  return (
    <div
      ref={containerRef}
      className={dragging ? "pe-split is-dragging" : "pe-split"}
    >
      <div
        className="pe-split-left"
        style={{ flexBasis: `${clamped * 100}%` }}
        aria-label={leftLabel}
      >
        {left}
      </div>
      <div
        className="pe-split-handle"
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={Math.round(clamped * 100)}
        aria-valuemin={20}
        aria-valuemax={80}
        aria-label="Resize panes"
        tabIndex={0}
        onPointerDown={(event) => {
          draggingRef.current = true;
          setDragging(true);
          const target = event.currentTarget;
          if (typeof target.setPointerCapture === "function") {
            try {
              target.setPointerCapture(event.pointerId);
            } catch {
              // jsdom and some browsers may not support pointer capture.
            }
          }
          onRatioChange(ratioFromClientX(event.clientX));
        }}
        onPointerMove={(event) => {
          if (!draggingRef.current) return;
          onRatioChange(ratioFromClientX(event.clientX));
        }}
        onPointerUp={(event) => {
          draggingRef.current = false;
          setDragging(false);
          const target = event.currentTarget;
          if (typeof target.releasePointerCapture === "function") {
            try {
              if (typeof target.hasPointerCapture === "function") {
                if (target.hasPointerCapture(event.pointerId)) {
                  target.releasePointerCapture(event.pointerId);
                }
              } else {
                target.releasePointerCapture(event.pointerId);
              }
            } catch {
              // Ignore release errors when capture was never held.
            }
          }
        }}
        onPointerCancel={() => {
          draggingRef.current = false;
          setDragging(false);
        }}
        onKeyDown={handleKeyDown}
        onDoubleClick={() => onRatioChange(SPLIT_RATIO_DEFAULT)}
      />
      <div
        className="pe-split-right"
        style={{ flex: 1 }}
        aria-label={rightLabel}
      >
        {right}
      </div>
    </div>
  );
}
