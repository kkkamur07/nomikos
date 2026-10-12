import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { CSSProperties, JSX } from "react";

import type { LineResponse } from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  dominantScript,
  rowDirection,
  textDirection,
} from "../../utils/scriptDirection";
import type {
  RowDirection,
  Script,
  TextDirection,
} from "../../utils/scriptDirection";
import { approvedText } from "./hooks/utils";
import { segmentNumbersById, segmentsInNumberOrder } from "./segmentNumbering";
import { regionSeparators, type Region } from "./regions";
import { displayText, maskStripHeight } from "./textPanelGeometry";
import type { TextSource } from "./textPanelGeometry";
import { CloseIcon } from "./EditorIcons";

const BASE_FONT_SIZE = 16;
const EMPTY_REGIONS: Region[] = [];
const REGION_SEPARATOR_STYLE: CSSProperties = {
  padding: "10px 12px 4px",
  borderTop: "1px solid var(--border, #ebe8e3)",
  fontSize: "0.75rem",
  fontWeight: 600,
  letterSpacing: "0.02em",
  color: "var(--text-3, #6b6b6b)",
};
const MIN_MASK_FONT_SIZE = 11;
const MAX_MASK_FONT_SIZE = 28;

export type PageEditorTranscriptEditorProps = {
  lines: LineResponse[];
  /**
   * The Page's regions (ADR 0011). When there are any, each region's rows
   * open with its name, and the rows in no region with "Unassigned".
   */
  regions?: Region[];
  selectedSegmentId: string | null;
  hoveredSegmentId: string | null;
  preferredLayerId?: string | null;
  fontScale?: number;
  /** Size each row by its segment's mask height relative to the page median. */
  scaleFontToMask?: boolean;
  onSelectSegment: (lineId: string) => void;
  onHoverSegment: (lineId: string | null) => void;
  onFocusSegment: (lineId: string) => void;
  /**
   * Saves a row. ``previousText`` is the Ground truth text the edit started
   * from, so the server can refuse (409) when someone else changed it since.
   */
  onCommitText: (
    lineId: string,
    text: string,
    previousText: string,
  ) => Promise<void>;
  /** Fetches one line again; null when it no longer exists on the server. */
  onReloadLine?: (lineId: string) => Promise<LineResponse | null>;
  /** Escape on a clean row hands focus back to its segment on the canvas. */
  onReturnToCanvas?: (lineId: string) => void;
  /**
   * Asked once when the pane mounts: true when it was opened from the
   * keyboard and the selected row should take focus.
   */
  takeFocusOnOpen?: () => boolean;
  /** Hide the side-by-side pane through the same settings path as the gear. */
  onClose: () => void;
};

type RowError = {
  kind: "conflict" | "deleted" | "session" | "other";
  message: string;
};

function rowError(error: unknown): RowError {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return { kind: "conflict", message: "Changed elsewhere." };
    }
    if (error.status === 404) {
      return {
        kind: "deleted",
        message: "This segment was deleted elsewhere. Your text is kept below.",
      };
    }
    if (error.status === 401) {
      return {
        kind: "session",
        message: "Session expired, sign in again. Your text is kept.",
      };
    }
  }
  const message = error instanceof Error ? error.message : String(error);
  return { kind: "other", message: `Not saved: ${message}` };
}

// Unsaved row drafts live in this tab's sessionStorage, keyed by line id, so a
// reload, a sign-in redirect or a page turn does not lose them. ``base`` is
// the Ground truth text the draft was typed over, sent as ``previous_text``.
const DRAFT_KEY_PREFIX = "nomikos_transcript_draft:";

type StoredDraft = { text: string; base: string };

export function readStoredDraft(lineId: string): StoredDraft | null {
  try {
    const raw = window.sessionStorage.getItem(DRAFT_KEY_PREFIX + lineId);
    if (raw === null) return null;
    const parsed = JSON.parse(raw) as Partial<StoredDraft>;
    if (typeof parsed.text !== "string" || typeof parsed.base !== "string") {
      return null;
    }
    return { text: parsed.text, base: parsed.base };
  } catch {
    return null;
  }
}

function writeStoredDraft(lineId: string, draft: StoredDraft): void {
  try {
    window.sessionStorage.setItem(
      DRAFT_KEY_PREFIX + lineId,
      JSON.stringify(draft),
    );
  } catch {
    // Storage full or blocked: the row still holds the draft in memory.
  }
}

function clearStoredDraft(lineId: string): void {
  try {
    window.sessionStorage.removeItem(DRAFT_KEY_PREFIX + lineId);
  } catch {
    // Nothing to clear when storage is blocked.
  }
}

/** True when any of these lines holds a draft that has not been saved. */
export function hasUnsavedTranscriptDrafts(
  lineIds: readonly string[],
): boolean {
  return lineIds.some((id) => readStoredDraft(id) !== null);
}

export const LEAVE_WITH_UNSAVED_ROWS_MESSAGE =
  "Some transcript rows are not saved. Leave this page anyway? Their text stays in this tab and comes back when you return.";

/** Asks before leaving a page whose rows hold unsaved text. */
export function confirmLeaveWithUnsavedRows(
  lineIds: readonly string[],
): boolean {
  if (!hasUnsavedTranscriptDrafts(lineIds)) return true;
  return window.confirm(LEAVE_WITH_UNSAVED_ROWS_MESSAGE);
}

/**
 * While any of these lines holds an unsaved draft, closing or reloading the
 * tab asks first, and so does following an in-app link such as Back.
 */
export function useUnsavedTranscriptGuard(lineIds: readonly string[]): void {
  const idsRef = useRef(lineIds);
  idsRef.current = lineIds;
  useEffect(() => {
    function onBeforeUnload(event: BeforeUnloadEvent) {
      if (!hasUnsavedTranscriptDrafts(idsRef.current)) return;
      event.preventDefault();
      event.returnValue = "";
    }
    function onClick(event: MouseEvent) {
      if (event.defaultPrevented || event.button !== 0) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      const anchor =
        event.target instanceof Element
          ? event.target.closest("a[href]")
          : null;
      if (!anchor || anchor.getAttribute("target") === "_blank") return;
      if (confirmLeaveWithUnsavedRows(idsRef.current)) return;
      event.preventDefault();
      event.stopPropagation();
    }
    window.addEventListener("beforeunload", onBeforeUnload);
    document.addEventListener("click", onClick, true);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      document.removeEventListener("click", onClick, true);
    };
  }, []);
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Row font sizes that follow the canvas: each segment's mask strip height
 * against the page median, so a tall initial line reads larger than the
 * marginal note beside it. Segments without a mask keep the base size.
 */
function maskScaledFontSizes(
  lines: LineResponse[],
  baseSize: number,
): Map<string, number> {
  const heights = new Map<string, number>();
  for (const line of lines) {
    const height = maskStripHeight(line);
    if (height !== null && height > 0) heights.set(line.id, height);
  }
  const typical = median([...heights.values()]);
  const sizes = new Map<string, number>();
  if (typical === null) return sizes;
  for (const [id, height] of heights) {
    const size = Math.round((baseSize * height) / typical);
    sizes.set(
      id,
      Math.min(MAX_MASK_FONT_SIZE, Math.max(MIN_MASK_FONT_SIZE, size)),
    );
  }
  return sizes;
}

/**
 * Scroll ``scroller`` so ``row`` rests in the middle of its visible area.
 *
 * ``scrollIntoView`` would also scroll every scrolling ancestor, which moves
 * the whole editor; setting the pane's own ``scrollTop`` moves only the pane.
 */
export function centerRowInScroller(
  row: HTMLElement,
  scroller: HTMLElement,
): void {
  const rowBox = row.getBoundingClientRect();
  const scrollerBox = scroller.getBoundingClientRect();
  const offset = rowBox.top - scrollerBox.top;
  const target =
    scroller.scrollTop + offset - (scroller.clientHeight - rowBox.height) / 2;
  const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
  scroller.scrollTop = Math.round(Math.min(max, Math.max(0, target)));
}

function growToContent(element: HTMLTextAreaElement): void {
  element.style.height = "auto";
  element.style.height = `${element.scrollHeight}px`;
}

type RowStatus = "dirty" | "saving" | "saved" | "unsaved";

const STATUS_LABELS: Record<RowStatus, string> = {
  dirty: "Unsaved changes",
  saving: "Saving",
  saved: "Saved",
  unsaved: "Not saved",
};

const STATUS_MARKS: Record<RowStatus, string> = {
  dirty: "●",
  saving: "…",
  saved: "✓",
  unsaved: "!",
};

type TranscriptRowProps = {
  line: LineResponse;
  number: number;
  seedText: string;
  /** The line's stored Ground truth text ("" when none). */
  baseText: string;
  seedSource: TextSource;
  selected: boolean;
  hovered: boolean;
  pageDirection: TextDirection;
  pageScript: Script | null;
  fontSize: number;
  prevId: string | null;
  nextId: string | null;
  onSelectSegment: (lineId: string) => void;
  onHoverSegment: (lineId: string | null) => void;
  onFocusSegment: (lineId: string) => void;
  onCommitText: (
    lineId: string,
    text: string,
    previousText: string,
  ) => Promise<void>;
  onReloadRow: (
    lineId: string,
  ) => Promise<{ seed: string; base: string } | null>;
  onReturnToCanvas?: (lineId: string) => void;
  onFocusId: (lineId: string) => void;
  registerTextarea: (lineId: string, el: HTMLTextAreaElement | null) => void;
  registerRow: (lineId: string, el: HTMLDivElement | null) => void;
};

function TranscriptRow(props: TranscriptRowProps): JSX.Element {
  const {
    line,
    number,
    seedText,
    baseText,
    seedSource,
    selected,
    hovered,
    pageDirection,
    pageScript,
    fontSize,
    prevId,
    nextId,
    onSelectSegment,
    onHoverSegment,
    onFocusSegment,
    onCommitText,
    onReloadRow,
    onReturnToCanvas,
    onFocusId,
    registerTextarea,
    registerRow,
  } = props;

  // A draft left unsaved by a reload or a sign-in redirect comes back here.
  const [restored] = useState(() => {
    const stored = readStoredDraft(line.id);
    if (stored === null) return null;
    if (stored.text === seedText) {
      clearStoredDraft(line.id);
      return null;
    }
    return stored;
  });
  const [draft, setDraft] = useState(restored?.text ?? seedText);
  const [edited, setEdited] = useState(restored !== null);
  const [committed, setCommitted] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<RowError | null>(null);
  // The direction a row has when it takes focus holds until it loses it, so
  // the row does not flip sides mid-word while it is typed in.
  const [focusDirection, setFocusDirection] = useState<RowDirection | null>(
    null,
  );

  const draftRef = useRef(draft);
  const committedRef = useRef(committed);
  const focusedRef = useRef(false);
  const editedRef = useRef(edited);
  editedRef.current = edited;
  const seedRef = useRef(seedText);
  const baseRef = useRef(restored?.base ?? baseText);
  const inFlightRef = useRef<Promise<boolean> | null>(null);
  const pendingRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const effectiveSource: TextSource =
    committed !== null ? "ground_truth" : seedSource;

  // Grow to fit the content after every render that changes the draft.
  useEffect(() => {
    const element = textareaRef.current;
    if (element) growToContent(element);
  }, [draft]);

  // Re-seed the draft when the saved text changes from outside, as long as
  // the row holds no unsaved draft. A focused row the user has not typed in
  // takes the new text too, or leaving it would save the old text back.
  useEffect(() => {
    if (seedText === seedRef.current) return;
    seedRef.current = seedText;
    if (editedRef.current) return;
    baseRef.current = baseText;
    draftRef.current = seedText;
    setDraft(seedText);
    // Our own save coming back as the page's text keeps its "Saved" mark.
    if (seedText !== committedRef.current) {
      committedRef.current = null;
      setCommitted(null);
    }
    setEdited(false);
    setError(null);
  }, [seedText, baseText]);

  const saveLoop = useCallback(async (): Promise<boolean> => {
    setSaving(true);
    setError(null);
    let failed = false;
    do {
      pendingRef.current = false;
      const sent = draftRef.current;
      if (sent === (committedRef.current ?? seedRef.current)) break;
      const trimmed = sent.trim();
      try {
        await onCommitText(line.id, trimmed, baseRef.current);
      } catch (unknownError) {
        setError(rowError(unknownError));
        failed = true;
        break;
      }
      baseRef.current = trimmed;
      committedRef.current = trimmed;
      setCommitted(trimmed);
      // Text typed while the save was out stays in the row, still unsaved.
      if (draftRef.current === sent) {
        draftRef.current = trimmed;
        setDraft(trimmed);
        setEdited(false);
        clearStoredDraft(line.id);
      } else {
        writeStoredDraft(line.id, {
          text: draftRef.current,
          base: baseRef.current,
        });
      }
    } while (pendingRef.current);
    setSaving(false);
    inFlightRef.current = null;
    return (
      !failed && draftRef.current === (committedRef.current ?? seedRef.current)
    );
  }, [line.id, onCommitText]);

  // A commit asked for while a save is out is queued: it runs once with the
  // latest draft when that save returns, and only then reports its result.
  const commitNow = useCallback((): Promise<boolean> => {
    if (inFlightRef.current !== null) {
      pendingRef.current = true;
      return inFlightRef.current;
    }
    const seed = committedRef.current ?? seedRef.current;
    if (draftRef.current === seed) return Promise.resolve(true);
    const run = saveLoop();
    inFlightRef.current = run;
    return run;
  }, [saveLoop]);

  async function reloadRow(): Promise<void> {
    const fresh = await onReloadRow(line.id);
    if (fresh === null) {
      setError(rowError(new ApiError("Not found", 404)));
      return;
    }
    seedRef.current = fresh.seed;
    baseRef.current = fresh.base;
    draftRef.current = fresh.seed;
    setDraft(fresh.seed);
    committedRef.current = null;
    setCommitted(null);
    setEdited(false);
    setError(null);
    clearStoredDraft(line.id);
  }

  // Script follows the row's own text as it is typed; an empty row takes the
  // page's, so the caret starts on the side the script reads from.
  const direction = focusDirection ?? rowDirection(draft, pageDirection);
  const script = dominantScript(draft) ?? pageScript;
  const status =
    error !== null
      ? "unsaved"
      : saving
        ? "saving"
        : edited
          ? "dirty"
          : committed !== null
            ? "saved"
            : null;

  const classNames = ["pe-transcript-row"];
  if (selected) classNames.push("is-selected");
  if (hovered) classNames.push("is-hovered");
  if (effectiveSource === "model" && !edited) classNames.push("is-model");
  if (saving) classNames.push("is-saving");
  if (error !== null) classNames.push("has-error");
  if (edited) classNames.push("is-dirty");

  return (
    <div
      ref={(element) => registerRow(line.id, element)}
      className={classNames.join(" ")}
      role="listitem"
      data-line-id={line.id}
      data-status={status ?? undefined}
      style={{ fontSize: `${fontSize}px` }}
      onMouseEnter={() => onHoverSegment(line.id)}
      onMouseLeave={() => onHoverSegment(null)}
    >
      <span className="pe-transcript-num">{number}</span>
      <textarea
        ref={(element) => {
          textareaRef.current = element;
          registerTextarea(line.id, element);
        }}
        className="pe-transcript-input"
        rows={1}
        spellCheck={false}
        dir={direction}
        data-script={script ?? undefined}
        style={{
          textAlign:
            direction === "rtl"
              ? "right"
              : direction === "ltr"
                ? "left"
                : "start",
        }}
        aria-label={`Segment ${number} text`}
        value={draft}
        onChange={(event) => {
          const value = event.target.value;
          draftRef.current = value;
          setDraft(value);
          setEdited(true);
          setError(null);
          if (value === (committedRef.current ?? seedRef.current)) {
            clearStoredDraft(line.id);
          } else {
            writeStoredDraft(line.id, { text: value, base: baseRef.current });
          }
        }}
        onFocus={() => {
          focusedRef.current = true;
          setFocusDirection(rowDirection(draftRef.current, pageDirection));
          onSelectSegment(line.id);
          onFocusSegment(line.id);
        }}
        onBlur={() => {
          focusedRef.current = false;
          setFocusDirection(null);
          void commitNow();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            void commitNow();
          } else if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void commitNow().then((ok) => {
              if (ok && nextId !== null) onFocusId(nextId);
            });
          } else if (event.key === "Escape") {
            const seed = committedRef.current ?? seedRef.current;
            if (draftRef.current === seed && error === null) {
              onReturnToCanvas?.(line.id);
              return;
            }
            draftRef.current = seed;
            setDraft(seed);
            setEdited(false);
            setError(null);
            clearStoredDraft(line.id);
          } else if (event.key === "ArrowDown") {
            event.preventDefault();
            if (nextId !== null) onFocusId(nextId);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            if (prevId !== null) onFocusId(prevId);
          } else if (event.key === "Tab") {
            // Past the first or last row Tab leaves the list as usual.
            const targetId = event.shiftKey ? prevId : nextId;
            if (targetId !== null) {
              event.preventDefault();
              onFocusId(targetId);
            }
          }
        }}
      />
      <span
        className="pe-transcript-status"
        aria-label={status === null ? undefined : STATUS_LABELS[status]}
        title={status === null ? undefined : STATUS_LABELS[status]}
      >
        {status === null ? "" : STATUS_MARKS[status]}
      </span>
      {error !== null ? (
        <p className="pe-transcript-error" role="alert">
          {error.message}
          {error.kind === "conflict" ? (
            <>
              {" "}
              <button
                type="button"
                className="pe-transcript-reload"
                onClick={() => void reloadRow()}
              >
                Reload row
              </button>
            </>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

export function PageEditorTranscriptEditor(
  props: PageEditorTranscriptEditorProps,
): JSX.Element {
  const {
    lines,
    regions = EMPTY_REGIONS,
    selectedSegmentId,
    hoveredSegmentId,
    preferredLayerId = null,
    fontScale = 1,
    scaleFontToMask = false,
    onSelectSegment,
    onHoverSegment,
    onFocusSegment,
    onCommitText,
    onReloadLine,
    onReturnToCanvas,
    takeFocusOnOpen,
    onClose,
  } = props;

  // Rows follow the stored Segment order, the same sequence the canvas numbers.
  const ordered = useMemo(() => segmentsInNumberOrder(lines), [lines]);
  const numbers = useMemo(() => segmentNumbersById(lines), [lines]);
  const separators = useMemo(
    () => regionSeparators(ordered, regions),
    [ordered, regions],
  );
  const seeds = useMemo(
    () =>
      new Map(
        lines.map((line) => [line.id, displayText(line, preferredLayerId)]),
      ),
    [lines, preferredLayerId],
  );
  const pageScript = useMemo(
    () => dominantScript([...seeds.values()].map((seed) => seed.text)),
    [seeds],
  );
  const pageDirection = useMemo(
    () => textDirection([...seeds.values()].map((seed) => seed.text)),
    [seeds],
  );
  const baseFontSize = BASE_FONT_SIZE * fontScale;
  const maskFontSizes = useMemo(
    () =>
      scaleFontToMask
        ? maskScaledFontSizes(lines, baseFontSize)
        : new Map<string, number>(),
    [lines, scaleFontToMask, baseFontSize],
  );

  const textareaEls = useRef(new Map<string, HTMLTextAreaElement>());
  const rowEls = useRef(new Map<string, HTMLDivElement>());
  const scrollerRef = useRef<HTMLDivElement>(null);

  const centerRow = useCallback((id: string): void => {
    const row = rowEls.current.get(id);
    const scroller = scrollerRef.current;
    if (row && scroller) centerRowInScroller(row, scroller);
  }, []);

  // Arrow keys, Enter and Tab move between rows here. The browser's own
  // focus scroll would leave the new row at an edge, so it is suppressed and
  // the row is centred like a selection from the canvas.
  const focusId = useCallback(
    (id: string): void => {
      textareaEls.current.get(id)?.focus({ preventScroll: true });
      centerRow(id);
    },
    [centerRow],
  );

  // A reloaded row reads its text the same way the seeds above do.
  const reloadRow = useCallback(
    async (id: string): Promise<{ seed: string; base: string } | null> => {
      if (!onReloadLine) return null;
      const fresh = await onReloadLine(id);
      if (fresh === null) return null;
      return {
        seed: displayText(fresh, preferredLayerId).text,
        base: approvedText(fresh) ?? "",
      };
    },
    [onReloadLine, preferredLayerId],
  );

  // Opened from the keyboard (T), the pane takes focus on the selected row.
  const takeFocusOnOpenRef = useRef(takeFocusOnOpen);
  const selectedOnOpenRef = useRef(selectedSegmentId);
  useEffect(() => {
    const id = selectedOnOpenRef.current;
    if (id === null || !takeFocusOnOpenRef.current?.()) return;
    textareaEls.current.get(id)?.focus({ preventScroll: true });
    const row = rowEls.current.get(id);
    const scroller = scrollerRef.current;
    if (row && scroller) centerRowInScroller(row, scroller);
  }, []);

  const registerTextarea = useCallback(
    (id: string, element: HTMLTextAreaElement | null): void => {
      if (element) textareaEls.current.set(id, element);
      else textareaEls.current.delete(id);
    },
    [],
  );

  const registerRow = useCallback(
    (id: string, element: HTMLDivElement | null): void => {
      if (element) rowEls.current.set(id, element);
      else rowEls.current.delete(id);
    },
    [],
  );

  // Centre an incoming selection (canvas click, canvas keyboard navigation)
  // without moving keyboard focus. A row that already holds focus was picked
  // here, by a click that must not slide it out from under the pointer or by
  // a key that ``focusId`` has already centred.
  useEffect(() => {
    if (selectedSegmentId === null) return;
    const row = rowEls.current.get(selectedSegmentId);
    if (!row) return;
    if (row.contains(document.activeElement)) return;
    centerRow(selectedSegmentId);
  }, [selectedSegmentId, centerRow]);

  return (
    <div
      className="pe-transcript-wrap"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: 0,
      }}
    >
      <div
        className="pe-transcript-header"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "4px 8px 4px 16px",
          borderBottom: "1px solid var(--border, #ebe8e3)",
          fontSize: "0.75rem",
          fontWeight: 600,
          letterSpacing: "0.02em",
          color: "var(--text-3, #6b6b6b)",
        }}
      >
        <span>Transcript</span>
        <button
          type="button"
          className="pe-icon-btn"
          title="Hide transcript pane"
          aria-label="Hide transcript pane"
          onClick={onClose}
        >
          <CloseIcon />
        </button>
      </div>
      <div
        ref={scrollerRef}
        className="pe-transcript"
        role="list"
        style={{ flex: 1, minHeight: 0, height: "auto" }}
      >
        {/* An empty page shows an empty pane: no placeholder text. */}
        {ordered.map((line, index) => (
          <Fragment key={line.id}>
            {separators.has(line.id) ? (
              <div
                className="pe-transcript-region"
                role="heading"
                aria-level={3}
                style={REGION_SEPARATOR_STYLE}
              >
                {separators.get(line.id)}
              </div>
            ) : null}
            <TranscriptRow
              line={line}
              number={numbers.get(line.id) ?? index + 1}
              seedText={seeds.get(line.id)?.text ?? ""}
              baseText={approvedText(line) ?? ""}
              seedSource={seeds.get(line.id)?.source ?? "none"}
              selected={line.id === selectedSegmentId}
              hovered={line.id === hoveredSegmentId}
              pageDirection={pageDirection}
              pageScript={pageScript}
              fontSize={maskFontSizes.get(line.id) ?? baseFontSize}
              prevId={ordered[index - 1]?.id ?? null}
              nextId={ordered[index + 1]?.id ?? null}
              onSelectSegment={onSelectSegment}
              onHoverSegment={onHoverSegment}
              onFocusSegment={onFocusSegment}
              onCommitText={onCommitText}
              onReloadRow={reloadRow}
              onReturnToCanvas={onReturnToCanvas}
              onFocusId={focusId}
              registerTextarea={registerTextarea}
              registerRow={registerRow}
            />
          </Fragment>
        ))}
      </div>
    </div>
  );
}
