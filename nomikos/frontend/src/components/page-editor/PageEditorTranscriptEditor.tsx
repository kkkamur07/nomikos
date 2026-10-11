import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { JSX } from "react";

import type { LineResponse } from "../../api/client";
import { readingOrder } from "./readingOrder";
import { segmentNumbersById } from "./segmentNumbering";
import { displayText } from "./textPanelGeometry";
import type { TextSource } from "./textPanelGeometry";

export type PageEditorTranscriptEditorProps = {
  lines: LineResponse[];
  selectedSegmentId: string | null;
  hoveredSegmentId: string | null;
  textDirection: "ltr" | "rtl";
  preferredLayerId?: string | null;
  fontScale?: number;
  onSelectSegment: (lineId: string) => void;
  onHoverSegment: (lineId: string | null) => void;
  onFocusSegment: (lineId: string) => void;
  onCommitText: (lineId: string, text: string) => Promise<void>;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function growToContent(element: HTMLTextAreaElement): void {
  element.style.height = "auto";
  element.style.height = `${element.scrollHeight}px`;
}

type TranscriptRowProps = {
  line: LineResponse;
  number: number;
  seedText: string;
  seedSource: TextSource;
  selected: boolean;
  hovered: boolean;
  textDirection: "ltr" | "rtl";
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
    textDirection,
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
  // the row is not being edited right now.
  useEffect(() => {
    if (seedText === seedRef.current) return;
    seedRef.current = seedText;
    if (focusedRef.current) return;
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

  const classNames = ["pe-transcript-row"];
  if (selected) classNames.push("is-selected");
  if (hovered) classNames.push("is-hovered");
  if (effectiveSource === "model" && !edited) classNames.push("is-model");
  if (saving) classNames.push("is-saving");
  if (error !== null) classNames.push("has-error");

  return (
    <div
      ref={(element) => registerRow(line.id, element)}
      className={classNames.join(" ")}
      role="listitem"
      data-line-id={line.id}
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
        dir={textDirection}
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
          if (event.key === "Enter" && !event.shiftKey) {
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
          }
        }}
      />
      {error !== null ? (
        <p className="pe-transcript-error" role="alert">
          {error}
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
    selectedSegmentId,
    hoveredSegmentId,
    textDirection,
    preferredLayerId = null,
    fontScale = 1,
    onSelectSegment,
    onHoverSegment,
    onFocusSegment,
    onCommitText,
  } = props;

  const ordered = useMemo(
    () => readingOrder(lines, { direction: textDirection }),
    [lines, textDirection],
  );
  const numbers = useMemo(() => segmentNumbersById(lines), [lines]);
  const seeds = useMemo(
    () =>
      new Map(
        lines.map((line) => [line.id, displayText(line, preferredLayerId)]),
      ),
    [lines, preferredLayerId],
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
    <div className="pe-transcript" role="list" dir={textDirection}>
      {ordered.length === 0 ? (
        <p className="pe-transcript-empty">No segments on this page yet.</p>
      ) : (
        ordered.map((line, index) => (
          <TranscriptRow
            key={line.id}
            line={line}
            number={numbers.get(line.id) ?? index + 1}
            seedText={seeds.get(line.id)?.text ?? ""}
            seedSource={seeds.get(line.id)?.source ?? "none"}
            selected={line.id === selectedSegmentId}
            hovered={line.id === hoveredSegmentId}
            textDirection={textDirection}
            fontSize={16 * fontScale}
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
        ))
      )}
    </div>
  );
}
