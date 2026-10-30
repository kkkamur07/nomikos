import { useEffect, useState } from "react";

/**
 * Whatever has focus gets Space before the canvas does.
 *
 * A field types it, and a focused button is activated by it - Space is how
 * half the keyboard world presses a button, and swallowing it here would make
 * the page rail and the pager reachable by Tab but dead on arrival. Space only
 * means "pan" when nothing else has a claim on it.
 */
function ownsSpaceKey(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "BUTTON" || tag === "SUMMARY") return true;
  return target.getAttribute("role") === "button";
}

/**
 * Hold Space to pan from any tool.
 *
 * Without this the only way to reach another part of the page mid-drawing is
 * to disarm the tool, pan, and arm it again, which is why a drawing tool and
 * a pan gesture kept fighting over the left button. Every canvas tool a
 * researcher already uses (Figma, Photoshop, Excalidraw, tldraw) binds Space
 * to exactly this, so it costs nothing to learn.
 *
 * Typing a space into the transcription strip must not pan the page, hence
 * the ownership guard. preventDefault stops the browser scrolling the pane out
 * from under the gesture.
 */
export function useSpacePan(): boolean {
  const [spaceHeld, setSpaceHeld] = useState(false);
  useEffect(() => {
    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.code !== "Space" || event.repeat) return;
      if (ownsSpaceKey(event.target)) return;
      event.preventDefault();
      setSpaceHeld(true);
    }
    function handleKeyUp(event: globalThis.KeyboardEvent) {
      if (event.code !== "Space") return;
      setSpaceHeld(false);
    }
    // Alt-tabbing away with Space down would otherwise leave the canvas stuck
    // in pan override, because the keyup lands on another window.
    function releaseOnBlur() {
      setSpaceHeld(false);
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", releaseOnBlur);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", releaseOnBlur);
    };
  }, []);
  return spaceHeld;
}
