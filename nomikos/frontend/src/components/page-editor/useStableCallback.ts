import { useCallback, useRef } from "react";

/**
 * A function whose identity never changes but which always runs the latest
 * `callback`. Memoised canvas layers take these, so a parent re-render that
 * hands in a fresh closure does not re-render every polygon, and an effect
 * that calls one does not re-run on every render.
 */
export function useStableCallback<Args extends unknown[], Result>(
  callback: (...args: Args) => Result,
): (...args: Args) => Result {
  const latest = useRef(callback);
  latest.current = callback;
  return useCallback((...args: Args) => latest.current(...args), []);
}
