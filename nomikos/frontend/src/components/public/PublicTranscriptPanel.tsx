import type { PublicLayoutResponse } from "../../api/client";
import { lineTextForLayer, linesForPart } from "../../utils/publicLayout";

type PublicTranscriptPanelProps = {
  partId: string;
  layout: PublicLayoutResponse | null;
  selectedLineIndex: number | null;
  onSelectLine: (index: number | null) => void;
};

function previewText(text: string, max = 72): string {
  const trimmed = text.trim();
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max - 1)}…`;
}

export function PublicTranscriptPanel({
  partId,
  layout,
  selectedLineIndex,
  onSelectLine,
}: PublicTranscriptPanelProps) {
  // The line number is the position in reading order, as in the editor, not
  // `order + 1`: a page written before orders were kept dense can carry gaps.
  const partLines = linesForPart(layout?.lines, partId);
  const selectedLine =
    selectedLineIndex !== null && selectedLineIndex >= 0
      ? (partLines[selectedLineIndex] ?? null)
      : null;
  const selectedText = selectedLine
    ? lineTextForLayer(selectedLine, null)
    : null;

  return (
    <aside className="pub-transcript" aria-labelledby="transcript-heading">
      <header className="pub-transcript__head">
        <h2 className="pub-transcript__heading" id="transcript-heading">
          Transcription
        </h2>
        <span className="pub-transcript__count">
          {partLines.length} line{partLines.length === 1 ? "" : "s"}
        </span>
      </header>

      {selectedLine && (
        <div className="pub-transcript__focus" aria-live="polite">
          <span className="pub-transcript__focus-label">
            Line {partLines.indexOf(selectedLine) + 1}
          </span>
          <p className="pub-transcript__focus-text">
            {selectedText ?? (
              <span className="text-block__empty">
                No transcription for this line
              </span>
            )}
          </p>
        </div>
      )}

      {partLines.length === 0 ? (
        <p className="pub-transcript__empty">
          No line geometry on this page yet.
        </p>
      ) : (
        <ol className="pub-line-index" aria-label="Page transcription lines">
          {partLines.map((line, index) => {
            const text = lineTextForLayer(line, null);
            const selected = selectedLineIndex === index;
            return (
              <li key={line.id}>
                <button
                  type="button"
                  className={`pub-line-index__item${selected ? " pub-line-index__item--selected" : ""}`}
                  aria-pressed={selected}
                  onClick={() => onSelectLine(selected ? null : index)}
                >
                  <span className="pub-line-index__num">{index + 1}</span>
                  <span className="pub-line-index__text">
                    {text ? previewText(text) : "-"}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </aside>
  );
}
