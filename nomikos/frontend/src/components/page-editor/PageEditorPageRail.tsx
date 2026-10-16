import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import type { DocumentPartResponse } from "../../api/client";
import { AuthenticatedImage } from "../AuthenticatedImage";

/**
 * One rail row, in CSS pixels. The rail windows its thumbnails off `scrollTop`
 * rather than off an IntersectionObserver, so the two numbers have to agree:
 * `.pe-page-rail__item` is this tall including its gap.
 */
const RAIL_ITEM_HEIGHT_PX = 96;
/** Rows kept loaded either side of the visible window, so a scroll is not a wait. */
const RAIL_OVERSCAN = 4;
/**
 * Rows kept when the rail has no height to measure - a test environment with no
 * layout, and the first paint before the effect runs. Bounded on purpose: a
 * three-hundred-page manuscript must not open three hundred image requests
 * because the rail is on screen.
 */
const RAIL_UNMEASURED_SPAN = 12;
/** Thumbnails are decorative-sized; the full scan is megabytes. */
const RAIL_THUMB_WIDTH = 200;

/** The rail and the toolbar button that shows it, named so each can find the other. */
const PAGE_RAIL_ID = "pe-page-rail";
export const PAGE_RAIL_TOGGLE_ID = "pe-page-rail-toggle";

/** The keys that move focus along the rail: focused row and last row to target row. */
const RAIL_FOCUS_KEYS: Partial<
  Record<string, (focused: number, last: number) => number>
> = {
  ArrowDown: (focused, last) => Math.min(last, focused + 1),
  ArrowUp: (focused) => Math.max(0, focused - 1),
  Home: () => 0,
  End: (_focused, last) => last,
};

/**
 * What stands where the rail was once it is hidden: one narrow control on the
 * canvas's left edge, so hiding the page list is not a one-way door. It also
 * carries the id the page hands focus to once the rail's own collapse button
 * has unmounted underneath the researcher's finger.
 */
export function PageEditorPageRailTab({ onExpand }: { onExpand: () => void }) {
  return (
    <button
      type="button"
      id={PAGE_RAIL_TOGGLE_ID}
      className="pe-page-rail-tab"
      aria-expanded={false}
      aria-controls={PAGE_RAIL_ID}
      aria-label="Show the page list"
      title="Show the page list"
      onClick={onExpand}
    >
      ›
    </button>
  );
}

function thumbnailUrl(part: DocumentPartResponse): string | null {
  if (!part.image_url) return null;
  const separator = part.image_url.includes("?") ? "&" : "?";
  return `${part.image_url}${separator}w=${RAIL_THUMB_WIDTH}`;
}

type PageEditorPageRailProps = {
  parts: DocumentPartResponse[];
  activePartId: string | undefined;
  onSelectPart: (partId: string) => void;
  onCollapse: () => void;
};

/**
 * The document, as a scrollable strip of its pages.
 *
 * This is the part of the editor that answers "let me reach page 12 without
 * going back": the whole document is here, it scrolls, and picking a page
 * swaps the canvas underneath without unmounting anything around it. Rows are
 * real buttons, so Enter and Space activate them the way the browser already
 * promises - a link with a click handler would navigate on Enter instead, and
 * this editor has been bitten by exactly that before.
 */
export function PageEditorPageRail({
  parts,
  activePartId,
  onSelectPart,
  onCollapse,
}: PageEditorPageRailProps) {
  const listRef = useRef<HTMLOListElement>(null);
  const activeItemRef = useRef<HTMLLIElement | null>(null);
  const activeIndex = parts.findIndex((part) => part.id === activePartId);
  /** Null until the rail has a height to window against. */
  const [scrollWindow, setScrollWindow] = useState<{
    first: number;
    last: number;
  } | null>(null);

  const measure = useCallback(() => {
    const list = listRef.current;
    const height = list?.clientHeight ?? 0;
    // No layout to read - keep the unmeasured window rather than guessing a
    // viewport, which would either load nothing or load everything.
    if (!list || height <= 0) {
      setScrollWindow(null);
      return;
    }
    const first = Math.max(
      0,
      Math.floor(list.scrollTop / RAIL_ITEM_HEIGHT_PX) - RAIL_OVERSCAN,
    );
    const last =
      first + Math.ceil(height / RAIL_ITEM_HEIGHT_PX) + RAIL_OVERSCAN * 2;
    setScrollWindow((current) =>
      current && current.first === first && current.last === last
        ? current
        : { first, last },
    );
  }, []);

  useEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure, parts.length]);

  // Paging with the keyboard or the toolbar arrows moves the active row, and a
  // rail that does not follow it is a rail that lies about where you are.
  useEffect(() => {
    activeItemRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [activePartId]);

  /**
   * Move focus along the rail without turning a page.
   *
   * Tab alone would make a three-hundred-page document three hundred tab stops
   * before anything else in the editor is reachable, and the editor binds the
   * bare arrow keys globally, so the rail has to claim them while focus is
   * inside it. Stopping propagation is what keeps that claim local: React's
   * listener sits under `window`, where the editor's own handler waits.
   */
  function handleListKeyDown(event: ReactKeyboardEvent<HTMLOListElement>) {
    const move = RAIL_FOCUS_KEYS[event.key];
    if (!move) return;
    const pages = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>(
        ".pe-page-rail__page",
      ) ?? [],
    );
    const focused = pages.indexOf(
      globalThis.document.activeElement as HTMLButtonElement,
    );
    if (focused < 0) return;
    event.preventDefault();
    event.stopPropagation();
    pages[move(focused, pages.length - 1)].focus();
  }

  const unmeasuredCentre = activeIndex < 0 ? 0 : activeIndex;
  const loadedWindow = scrollWindow ?? {
    first: Math.max(0, unmeasuredCentre - RAIL_UNMEASURED_SPAN / 2),
    last: unmeasuredCentre + RAIL_UNMEASURED_SPAN / 2,
  };

  return (
    <nav className="pe-page-rail" id={PAGE_RAIL_ID} aria-label="Document pages">
      <div className="pe-page-rail__head">
        <span className="pe-page-rail__title">
          {parts.length} {parts.length === 1 ? "page" : "pages"}
        </span>
        <button
          type="button"
          className="pe-page-rail__collapse"
          onClick={onCollapse}
          aria-label="Hide the page list"
          title="Hide the page list"
        >
          ‹
        </button>
      </div>
      <ol
        className="pe-page-rail__list"
        ref={listRef}
        onScroll={measure}
        onKeyDown={handleListKeyDown}
        aria-label="Pages in this document"
      >
        {parts.map((part, index) => {
          const active = part.id === activePartId;
          const url = thumbnailUrl(part);
          const loaded =
            index >= loadedWindow.first && index <= loadedWindow.last;
          return (
            <li
              key={part.id}
              className="pe-page-rail__item"
              ref={active ? activeItemRef : undefined}
            >
              <button
                type="button"
                className={`pe-page-rail__page${active ? " pe-page-rail__page--on" : ""}`}
                aria-current={active ? "page" : undefined}
                aria-label={`Page ${index + 1}${part.reviewed ? ", reviewed" : ""}`}
                onClick={() => onSelectPart(part.id)}
              >
                <span className="pe-page-rail__thumb">
                  {url && loaded ? (
                    <AuthenticatedImage
                      compact
                      src={url}
                      alt={`Page ${index + 1} thumbnail`}
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        display: "block",
                      }}
                    />
                  ) : null}
                </span>
                <span className="pe-page-rail__num">{index + 1}</span>
                {part.reviewed && (
                  <span
                    className="pe-page-rail__reviewed"
                    aria-hidden="true"
                    title="Reviewed"
                  />
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
