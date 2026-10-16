import { useCallback, useRef, useState, type KeyboardEvent } from "react";

import { PAGE_EDITOR_SHORTCUTS } from "./pageEditorShortcuts";

/**
 * The tools ride the canvas, not the page header.
 *
 * Drawing a segment is a hand-on-the-image act, and the header is where
 * navigation, export and job control live. Keeping the two apart is what lets
 * the header stop growing every time a tool is added, and it puts the tool
 * within a short mouse travel of the thing being drawn. eScriptorium, Figma
 * and tldraw all land on some version of this.
 *
 * Zoom sits in the same island rather than in its own corner cluster: it is
 * navigation of the same image by the same hand, and two floating clusters
 * over one canvas is one too many.
 */

type CanvasTool = "none" | "rectangle" | "polygon";

type PageEditorCanvasIslandProps = {
  tool: CanvasTool;
  onSelectTool: () => void;
  onPickDrawMode: (mode: "rectangle" | "polygon") => void;
  canDelete: boolean;
  onDeleteSelected: () => void;
  zoomPercent: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToView: () => void;
  onResetZoom: () => void;
  /** Space is held, so every tool is temporarily the pan tool. */
  panOverride: boolean;
};

function toolClass(active: boolean): string {
  return active ? "pe-island__btn pe-island__btn--active" : "pe-island__btn";
}

/**
 * ``role="toolbar"`` promises one tab stop and arrow keys inside it, so the
 * island has to keep that promise: eight separate tab stops floating over the
 * canvas would put the page image eight presses away for anyone working from
 * the keyboard.
 */
const ISLAND_BUTTON_COUNT = 8;
/** The one button that can be disabled, and so the one that can lose focus. */
const DELETE_BUTTON_INDEX = 3;

export function PageEditorCanvasIsland({
  tool,
  onSelectTool,
  onPickDrawMode,
  canDelete,
  onDeleteSelected,
  zoomPercent,
  onZoomIn,
  onZoomOut,
  onFitToView,
  onResetZoom,
  panOverride,
}: PageEditorCanvasIslandProps) {
  const islandRef = useRef<HTMLDivElement>(null);
  const [focusIndex, setFocusIndex] = useState(0);

  const isFocusable = useCallback(
    (index: number) => index !== DELETE_BUTTON_INDEX || canDelete,
    [canDelete],
  );
  // Delete greys out with nothing selected. Were the tab stop left sitting on
  // it the island would have no reachable stop at all, so it falls back to
  // Select, which is never disabled.
  const tabStop = isFocusable(focusIndex) ? focusIndex : 0;

  const focusButton = useCallback((index: number) => {
    const buttons = islandRef.current?.querySelectorAll("button");
    buttons?.[index]?.focus();
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const step =
        event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (step !== 0) {
        event.preventDefault();
        let next = focusIndex;
        for (let hop = 0; hop < ISLAND_BUTTON_COUNT; hop += 1) {
          next = (next + step + ISLAND_BUTTON_COUNT) % ISLAND_BUTTON_COUNT;
          if (isFocusable(next)) break;
        }
        focusButton(next);
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        focusButton(0);
      } else if (event.key === "End") {
        event.preventDefault();
        focusButton(ISLAND_BUTTON_COUNT - 1);
      }
    },
    [focusIndex, isFocusable, focusButton],
  );

  // Focus is what moves; the roving tab stop follows it, whether it arrived by
  // arrow key, by Tab or by click.
  const rove = (index: number) => ({
    tabIndex: index === tabStop ? 0 : -1,
    onFocus: () => setFocusIndex(index),
  });

  return (
    <div
      ref={islandRef}
      className={`pe-island${panOverride ? " pe-island--panning" : ""}`}
      role="toolbar"
      aria-label="Canvas tools"
      aria-orientation="horizontal"
      onKeyDown={handleKeyDown}
    >
      <button
        type="button"
        {...rove(0)}
        className={toolClass(tool === "none")}
        aria-pressed={tool === "none"}
        aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.SELECT}
        aria-label={`Select and pan (${PAGE_EDITOR_SHORTCUTS.SELECT})`}
        title={`Select and pan (${PAGE_EDITOR_SHORTCUTS.SELECT}). Hold Space to pan from any tool.`}
        onClick={onSelectTool}
      >
        <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M3 1.6l9.4 5.9-4 .7-1.9 4.1L3 1.6z" />
        </svg>
        <span className="pe-island__label">Select</span>
      </button>

      <button
        type="button"
        {...rove(1)}
        className={toolClass(tool === "rectangle")}
        aria-pressed={tool === "rectangle"}
        aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.RECTANGLE}
        aria-label={`Rectangle segment (${PAGE_EDITOR_SHORTCUTS.RECTANGLE})`}
        title={`Draw rectangle segment (${PAGE_EDITOR_SHORTCUTS.RECTANGLE})`}
        onClick={() => onPickDrawMode("rectangle")}
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          aria-hidden="true"
        >
          <rect x="2.5" y="3.5" width="11" height="9" rx="1" />
        </svg>
        <span className="pe-island__label">Rect</span>
      </button>

      <button
        type="button"
        {...rove(2)}
        className={toolClass(tool === "polygon")}
        aria-pressed={tool === "polygon"}
        aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.POLYGON}
        aria-label={`Polygon segment (${PAGE_EDITOR_SHORTCUTS.POLYGON})`}
        title={`Draw polygon segment (${PAGE_EDITOR_SHORTCUTS.POLYGON})`}
        onClick={() => onPickDrawMode("polygon")}
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M8 2.2l5.6 4-2.1 6.6H4.5L2.4 6.2z" />
        </svg>
        <span className="pe-island__label">Poly</span>
      </button>

      <button
        type="button"
        {...rove(3)}
        className="pe-island__btn"
        disabled={!canDelete}
        aria-keyshortcuts={PAGE_EDITOR_SHORTCUTS.DELETE}
        aria-label={`Delete selected (${PAGE_EDITOR_SHORTCUTS.DELETE})`}
        title={`Delete selected (${PAGE_EDITOR_SHORTCUTS.DELETE})`}
        onClick={onDeleteSelected}
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M3 4.3h10M6.4 4.3V2.8h3.2v1.5M4.5 4.3l.7 8.5h5.6l.7-8.5" />
        </svg>
        <span className="pe-island__label">Delete</span>
      </button>

      <span className="pe-island__divider" aria-hidden="true" />

      <button
        type="button"
        {...rove(4)}
        className="pe-island__btn pe-island__btn--icon"
        onClick={onZoomOut}
        aria-label="Zoom out"
        title="Zoom out"
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M4 8h8" />
        </svg>
      </button>
      <button
        type="button"
        {...rove(5)}
        className="pe-island__zoom"
        onClick={onResetZoom}
        aria-label={`Zoom ${zoomPercent}%. Reset to 100%`}
        title="Reset zoom to 100%"
      >
        <span aria-live="polite">{zoomPercent}%</span>
      </button>
      <button
        type="button"
        {...rove(6)}
        className="pe-island__btn pe-island__btn--icon"
        onClick={onZoomIn}
        aria-label="Zoom in"
        title="Zoom in"
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M8 4v8M4 8h8" />
        </svg>
      </button>
      <button
        type="button"
        {...rove(7)}
        className="pe-island__btn pe-island__btn--icon"
        onClick={onFitToView}
        aria-label="Fit page to view"
        title="Fit page to view"
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" />
        </svg>
      </button>
    </div>
  );
}
