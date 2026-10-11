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
import { dominantScript, textDirection } from "../../utils/scriptDirection";
import type { Script, TextDirection } from "../../utils/scriptDirection";
import { segmentNumbersById, segmentsInNumberOrder } from "./segmentNumbering";
import { regionSeparators, type Region } from "./regions";
import { displayText, maskStripHeight } from "./textPanelGeometry";
import type { TextSource } from "./textPanelGeometry";

const BASE_FONT_SIZE = 16;
const EMPTY_REGIONS: Region[] = [];
const REGION_SEPARATOR_STYLE: CSSProperties = {
  padding: "10px 12px 4px",
  borderTop: "1px solid var(--border, #ebe8e3)",
  fontSize: "0.75rem",
  fontWeight: 600,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
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
  onCommitText: (lineId: string, text: string) => Promise<void>;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
  onCommitText: (lineId: string, text: string) => Promise<void>;
  onFocusId: (lineId: string) => void;
  registerTextarea: (lineId: string, el: HTMLTextAreaElement | null) => void;
  registerRow: (lineId: string, el: HTMLDivElement | null) => void;
};

function TranscriptRow(props: TranscriptRowProps): JSX.Element {
  const {
    line,
    number,
    seedText,
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
    onFocusId,
    registerTextarea,
    registerRow,
  } = props;

  const [draft, setDraft] = useState(seedText);
  const [edited, setEdited] = useState(false);
  const [committed, setCommitted] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const draftRef = useRef(draft);
  const committedRef = useRef(committed);
  const savingRef = useRef(saving);
  const focusedRef = useRef(false);
  const editedRef = useRef(edited);
  editedRef.current = edited;
  const seedRef = useRef(seedText);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const effectiveSource: TextSource =
    committed !== null ? "ground_truth" : seedSource;

  // Grow to fit the content after every render that changes the draft.
  useEffect(() => {
    const element = textareaRef.current;
    if (element) growToContent(element);
  }, [draft]);

  // Re-seed the draft when the saved text changes from outside, as long as
  // the row is not being edited right now and holds no unsaved draft.
  useEffect(() => {
    if (seedText === seedRef.current) return;
    seedRef.current = seedText;
    if (focusedRef.current || editedRef.current) return;
    draftRef.current = seedText;
    setDraft(seedText);
    committedRef.current = null;
    setCommitted(null);
    setEdited(false);
    setError(null);
  }, [seedText]);

  const commitNow = useCallback(async (): Promise<boolean> => {
    const seed = committedRef.current ?? seedRef.current;
    if (draftRef.current === seed || savingRef.current) return true;
    const trimmed = draftRef.current.trim();
    savingRef.current = true;
    setSaving(true);
    setError(null);
    try {
      await onCommitText(line.id, trimmed);
    } catch (unknownError) {
      savingRef.current = false;
      setSaving(false);
      setError(errorMessage(unknownError));
      return false;
    }
    savingRef.current = false;
    setSaving(false);
    committedRef.current = trimmed;
    setCommitted(trimmed);
    draftRef.current = trimmed;
    setDraft(trimmed);
    setEdited(false);
    return true;
  }, [line.id, onCommitText]);

  // Direction and script follow the row's own text as it is typed; an empty
  // row takes the page's, so the caret starts on the side the script reads from.
  const direction = textDirection(draft, pageDirection);
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
        style={{ textAlign: direction === "rtl" ? "right" : "left" }}
        aria-label={`Segment ${number} text`}
        value={draft}
        onChange={(event) => {
          draftRef.current = event.target.value;
          setDraft(event.target.value);
          setEdited(true);
          setError(null);
        }}
        onFocus={() => {
          focusedRef.current = true;
          onSelectSegment(line.id);
          onFocusSegment(line.id);
        }}
        onBlur={() => {
          focusedRef.current = false;
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
            draftRef.current = seed;
            setDraft(seed);
            setEdited(false);
            setError(null);
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
          Not saved: {error}
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

  const focusId = useCallback((id: string): void => {
    textareaEls.current.get(id)?.focus();
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

  // Scroll an incoming selection into view without moving keyboard focus.
  useEffect(() => {
    if (selectedSegmentId === null) return;
    const row = rowEls.current.get(selectedSegmentId);
    if (!row) return;
    if (row.contains(document.activeElement)) return;
    if (typeof row.scrollIntoView === "function") {
      row.scrollIntoView({ block: "nearest" });
    }
  }, [selectedSegmentId]);

  return (
    <div className="pe-transcript" role="list">
      {ordered.length === 0 ? (
        <p className="pe-transcript-empty">No segments on this page yet.</p>
      ) : (
        ordered.map((line, index) => (
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
              onFocusId={focusId}
              registerTextarea={registerTextarea}
              registerRow={registerRow}
            />
          </Fragment>
        ))
      )}
    </div>
  );
}
