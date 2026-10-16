import { useEffect, useRef } from "react";
import { useStableCallback } from "../useStableCallback";

/**
 * Closes an open popover on a click anywhere outside it. Returns the ref to
 * put on the element that counts as inside (the trigger and the popover).
 */
export function useDismissOnOutsideClick<T extends HTMLElement>(
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
    globalThis.document.addEventListener("click", handleClick);
    return () => globalThis.document.removeEventListener("click", handleClick);
  }, [open, dismiss]);

  return ref;
}
