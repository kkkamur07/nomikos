import { useEffect, useRef, type FocusEvent } from "react";
import { useStableCallback } from "../useStableCallback";

/**
 * Closes an open popover on a click anywhere outside it, on Escape, and once
 * focus moves outside it. Returns the ref to put on the element that counts
 * as inside (the trigger and the popover) and the blur handler for it.
 */
export function usePopoverDismiss<T extends HTMLElement>(
  open: boolean,
  onDismiss: () => void,
) {
  const ref = useRef<T>(null);
  const dismiss = useStableCallback(onDismiss);

  useEffect(() => {
    if (!open) return;
    function handleClick(event: MouseEvent) {
      const inside = ref.current;
      if (
        inside &&
        event.target instanceof Node &&
        !inside.contains(event.target)
      ) {
        dismiss();
      }
    }
    // Escape closes what is open and hands focus back to its trigger. Capture
    // phase, so the editor's own Escape (deselect) does not also fire.
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      dismiss();
      ref.current?.querySelector<HTMLElement>("button[aria-expanded]")?.focus();
    }
    globalThis.document.addEventListener("click", handleClick);
    globalThis.document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      globalThis.document.removeEventListener("click", handleClick);
      globalThis.document.removeEventListener("keydown", handleKeyDown, true);
    };
  }, [open, dismiss]);

  // A null target (a click on the panel's own text) is not leaving it.
  function onBlur(event: FocusEvent<T>) {
    const next = event.relatedTarget as Node | null;
    if (open && next && !event.currentTarget.contains(next)) dismiss();
  }

  return { ref, onBlur };
}
