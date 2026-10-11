import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";

/**
 * How far the editor bar has folded to stay on one row.
 *
 * 0: every action button carries its label.
 * 1: the labelled buttons (Segment, Transcribe, Transcript, Workflow) are
 *    icon-only.
 * 2: those buttons fold into one "More" menu at the right, beside Check page,
 *    which the Workflow menu carries at the first two levels.
 */
export type ToolbarCollapseLevel = 0 | 1 | 2;

export const MAX_TOOLBAR_COLLAPSE_LEVEL: ToolbarCollapseLevel = 2;

export type ToolbarMeasure = {
  /** Width the bar has: its ``clientWidth``. */
  available: number;
  /** Width the bar's content asks for at the current level: ``scrollWidth``. */
  required: number;
};

/**
 * One step of the fold. A bar that overflows folds one level further and
 * remembers what the level it left needed; it unfolds only once that width is
 * there again, so it never flickers between two levels at one width.
 * ``needed`` is updated in place and returned with the new level.
 */
export function nextCollapseLevel(
  level: ToolbarCollapseLevel,
  measure: ToolbarMeasure,
  needed: Partial<Record<ToolbarCollapseLevel, number>>,
): ToolbarCollapseLevel {
  if (measure.available <= 0) return level;
  if (measure.required > measure.available) {
    needed[level] = measure.required;
    return level < MAX_TOOLBAR_COLLAPSE_LEVEL
      ? ((level + 1) as ToolbarCollapseLevel)
      : level;
  }
  if (level > 0) {
    const wider = (level - 1) as ToolbarCollapseLevel;
    const want = needed[wider];
    if (want === undefined || measure.available >= want) return wider;
  }
  return level;
}

/**
 * Measure the bar on every resize and every render (a longer document name or
 * a "Segmenting" pill changes what fits as surely as the window does) and
 * fold or unfold until the content fits on one row.
 */
export function useToolbarCollapse(
  ref: RefObject<HTMLElement | null>,
): ToolbarCollapseLevel {
  const [level, setLevel] = useState<ToolbarCollapseLevel>(0);
  const neededRef = useRef<Partial<Record<ToolbarCollapseLevel, number>>>({});
  const levelRef = useRef(level);
  levelRef.current = level;

  const measure = useCallback(() => {
    const element = ref.current;
    if (!element) return;
    const next = nextCollapseLevel(
      levelRef.current,
      { available: element.clientWidth, required: element.scrollWidth },
      neededRef.current,
    );
    if (next !== levelRef.current) setLevel(next);
  }, [ref]);

  // Runs after every render, before paint, so a fold settles without a frame
  // of wrapped or clipped bar showing.
  useLayoutEffect(measure);

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, measure]);

  return level;
}
