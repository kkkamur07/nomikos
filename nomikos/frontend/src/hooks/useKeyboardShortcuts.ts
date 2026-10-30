import { useEffect, useRef } from "react";

interface ShortcutHandlers {
  onDrawBox?: () => void;
  onDrawPolygon?: () => void;
  onDelete?: () => void;
  onEscape?: () => void;
  onEnter?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onPreviousPage?: () => void;
  onNextPage?: () => void;
}

type Shortcut = {
  handler: keyof ShortcutHandlers;
  matches: (key: string, ctrl: boolean, shift: boolean) => boolean;
  /** Enter keeps its default so a focused button still activates. */
  keepDefault?: boolean;
};

/**
 * First match wins, in this order. PageUp/PageDown mean previous/next page of
 * the document: nothing in the editor scrolls with these keys.
 */
const SHORTCUTS: readonly Shortcut[] = [
  {
    handler: "onPreviousPage",
    matches: (key, ctrl) => key === "pageup" && !ctrl,
  },
  {
    handler: "onNextPage",
    matches: (key, ctrl) => key === "pagedown" && !ctrl,
  },
  { handler: "onDrawBox", matches: (key, ctrl) => key === "b" && !ctrl },
  { handler: "onDrawPolygon", matches: (key, ctrl) => key === "p" && !ctrl },
  {
    handler: "onDelete",
    matches: (key, ctrl) => (key === "delete" || key === "backspace") && !ctrl,
  },
  { handler: "onEscape", matches: (key) => key === "escape" },
  { handler: "onEnter", matches: (key) => key === "enter", keepDefault: true },
  {
    handler: "onUndo",
    matches: (key, ctrl, shift) => ctrl && key === "z" && !shift,
  },
  {
    handler: "onRedo",
    matches: (key, ctrl, shift) =>
      (ctrl && shift && key === "z") || (ctrl && key === "y"),
  },
];

/**
 * A focused control that takes keyboard input keeps its keys: Backspace in a
 * select, type-ahead letters in a listbox, Delete in a contenteditable.
 */
function ownsKeyboard(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.closest('[contenteditable]:not([contenteditable="false"])'))
    return true;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  const role = target.getAttribute("role");
  return role === "listbox" || role === "option" || role === "combobox";
}

export const useKeyboardShortcuts = (handlers: ShortcutHandlers) => {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (ownsKeyboard(e.target)) return;
      const key = e.key.toLowerCase();
      const ctrl = e.ctrlKey || e.metaKey;
      const shortcut = SHORTCUTS.find((entry) =>
        entry.matches(key, ctrl, e.shiftKey),
      );
      const handler = shortcut && handlersRef.current[shortcut.handler];
      if (!shortcut || !handler) return;
      if (!shortcut.keepDefault) e.preventDefault();
      handler();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
};
