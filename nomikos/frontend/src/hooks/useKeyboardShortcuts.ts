import { useEffect, useRef } from "react";

interface ShortcutHandlers {
  onDrawBox?: () => void;
  onDrawPolygon?: () => void;
  onEditVertices?: () => void;
  onMerge?: () => void;
  onDelete?: () => void;
  onEscape?: () => void;
  onEnter?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onMoveLeft?: () => void;
  onMoveRight?: () => void;
  onPreviousPage?: () => void;
  onNextPage?: () => void;
}

export const useKeyboardShortcuts = (handlers: ShortcutHandlers) => {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input/textarea
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const current = handlersRef.current;
      const key = e.key.toLowerCase();
      const ctrl = e.ctrlKey || e.metaKey;

      // PageUp/PageDown - previous/next page of the document. Nothing in the
      // editor scrolls with these keys, so they are free to mean what they say.
      if (key === "pageup" && !ctrl) {
        e.preventDefault();
        current.onPreviousPage?.();
      } else if (key === "pagedown" && !ctrl) {
        e.preventDefault();
        current.onNextPage?.();
      }
      // B - Draw Box
      else if (key === "b" && !ctrl) {
        e.preventDefault();
        current.onDrawBox?.();
      }
      // P - Draw Polygon
      else if (key === "p" && !ctrl) {
        e.preventDefault();
        current.onDrawPolygon?.();
      }
      // M - Merge the selected segments. Free: the editor binds B, P, V,
      // Delete, Escape, Enter, Ctrl+Z/Y and PageUp/PageDown, and nothing else.
      else if (key === "m" && !ctrl) {
        e.preventDefault();
        current.onMerge?.();
      }
      // V - Edit Vertices
      else if (key === "v" && !ctrl) {
        e.preventDefault();
        current.onEditVertices?.();
      }
      // Delete/Backspace - Delete selected
      else if ((key === "delete" || key === "backspace") && !ctrl) {
        e.preventDefault();
        current.onDelete?.();
      }
      // Escape - Cancel
      else if (key === "escape") {
        e.preventDefault();
        current.onEscape?.();
      }
      // Enter - Complete (e.g. polygon)
      else if (key === "enter") {
        current.onEnter?.();
      }
      // Ctrl+Z - Undo
      else if (ctrl && key === "z" && !e.shiftKey) {
        e.preventDefault();
        current.onUndo?.();
      }
      // Ctrl+Shift+Z or Ctrl+Y - Redo
      else if ((ctrl && e.shiftKey && key === "z") || (ctrl && key === "y")) {
        e.preventDefault();
        current.onRedo?.();
      }
      // Arrow keys - Move selected region
      else if (key === "arrowup") {
        e.preventDefault();
        current.onMoveUp?.();
      } else if (key === "arrowdown") {
        e.preventDefault();
        current.onMoveDown?.();
      } else if (key === "arrowleft") {
        e.preventDefault();
        current.onMoveLeft?.();
      } else if (key === "arrowright") {
        e.preventDefault();
        current.onMoveRight?.();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
};
