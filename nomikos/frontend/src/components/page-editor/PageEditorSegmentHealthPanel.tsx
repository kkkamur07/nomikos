/**
 * The three mistakes kraken makes on a page, offered one at a time.
 *
 * Nothing here applies anything on its own. Each finding is a row with the fix
 * spelled out and a button next to it, because the whole point of the feature
 * is that a human decides. Suspects go further and carry no one-click action at
 * all: they are things that *look* like noise, some of which is real ink nobody
 * has transcribed yet, so the row states the reasons and asks for a second
 * click before it will delete anything.
 */
import { useState } from "react";

import type {
  SegmentHealthResponse,
  SegmentMergeResponse,
  SegmentOverlapResponse,
  SegmentSplitResponse,
  SegmentSuspectResponse,
} from "../../api/client";

export type SegmentHealthAction =
  | { kind: "split"; lineId: string }
  | { kind: "merge"; primaryId: string; fragmentId: string }
  | { kind: "trim"; upperId: string; lowerId: string }
  | { kind: "delete"; lineId: string };

type PageEditorSegmentHealthPanelProps = {
  report: SegmentHealthResponse | null;
  loading: boolean;
  error: string | null;
  /** The finding currently being applied, so only its own row shows a spinner. */
  pending: string | null;
  onApply: (action: SegmentHealthAction) => void;
  onRefresh: () => void;
  /** Draws attention to a finding's segments on the canvas. */
  onHighlight?: (lineIds: string[]) => void;
};

/** Short, stable label for a line, so a row can name one without a full uuid. */
function shortLineId(lineId: string): string {
  return lineId.slice(0, 8);
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function PageEditorSegmentHealthPanel({
  report,
  loading,
  error,
  pending,
  onApply,
  onRefresh,
  onHighlight,
}: PageEditorSegmentHealthPanelProps) {
  // Which suspect has been armed for deletion. Only ever one at a time, and it
  // resets on every render of a new report, so a click cannot land on a row
  // that has since moved.
  const [armedSuspect, setArmedSuspect] = useState<string | null>(null);

  if (loading && !report) {
    return (
      <div
        className="pe-dropdown pe-dropdown--segment-health"
        role="dialog"
        aria-label="Segment health"
      >
        <div className="pe-dd-section">Checking the page…</div>
      </div>
    );
  }

  // Only when there is nothing else to show. A fix the server refuses leaves
  // the report standing (and freshly re-read), and swapping the findings out
  // for a bare "Try again" would hide the very list that now explains the
  // refusal: the finding the reviewer clicked is usually gone from it.
  if (error && !report) {
    return (
      <div
        className="pe-dropdown pe-dropdown--segment-health"
        role="dialog"
        aria-label="Segment health"
      >
        <div className="pe-dd-section">Segment health</div>
        <p className="pe-dd-hint" role="alert">
          {error}
        </p>
        <button type="button" className="pe-dd-btn" onClick={onRefresh}>
          Try again
        </button>
      </div>
    );
  }

  if (!report) {
    return null;
  }

  const hasFindings = report.finding_count > 0;

  return (
    <div
      className="pe-dropdown pe-dropdown--segment-health"
      role="dialog"
      aria-label="Segment health"
    >
      <div className="pe-dd-section">
        Segment health
        {report.measured_page ? null : (
          // The thresholds are all relative to the page, so a page whose size
          // had to be guessed from the ink is worth saying out loud rather than
          // quietly reporting slightly different findings.
          <span className="pe-dd-hint">
            {" "}
            page size estimated from the segments
          </span>
        )}
      </div>

      {error ? (
        <p className="pe-dd-hint" role="alert">
          {error}
        </p>
      ) : null}

      {!hasFindings ? (
        <p className="pe-dd-hint">
          Nothing systematic found across {report.considered_count} segments.
        </p>
      ) : null}

      {report.spanning.length > 0 ? (
        <>
          <div className="pe-dd-section">Across two columns</div>
          {report.spanning.map((split: SegmentSplitResponse) => (
            <div
              key={`split-${split.line_id}`}
              className="pe-dd-row"
              onMouseEnter={() => onHighlight?.([split.line_id])}
            >
              <span className="pe-dd-check__label">
                Segment {shortLineId(split.line_id)} spans the gutter. Cut into{" "}
                {split.piece_count} pieces.
              </span>
              <button
                type="button"
                className="pe-dd-btn"
                disabled={pending === split.line_id}
                onClick={() =>
                  onApply({ kind: "split", lineId: split.line_id })
                }
              >
                {pending === split.line_id ? "Cutting…" : "Cut at the gutter"}
              </button>
            </div>
          ))}
        </>
      ) : null}

      {report.fragments.length > 0 ? (
        <>
          <div className="pe-dd-section">Broken into pieces</div>
          {report.fragments.map((merge: SegmentMergeResponse) => (
            <div
              key={`merge-${merge.fragment_id}`}
              className="pe-dd-row"
              onMouseEnter={() =>
                onHighlight?.([merge.primary_id, merge.fragment_id])
              }
            >
              <span className="pe-dd-check__label">
                {shortLineId(merge.fragment_id)} is a piece of{" "}
                {shortLineId(merge.primary_id)}. Merging keeps{" "}
                {shortLineId(merge.primary_id)}, so its transcription survives.
              </span>
              <button
                type="button"
                className="pe-dd-btn"
                disabled={pending === merge.fragment_id}
                onClick={() =>
                  onApply({
                    kind: "merge",
                    primaryId: merge.primary_id,
                    fragmentId: merge.fragment_id,
                  })
                }
              >
                {pending === merge.fragment_id ? "Merging…" : "Merge"}
              </button>
            </div>
          ))}
        </>
      ) : null}

      {report.overlaps.length > 0 ? (
        <>
          <div className="pe-dd-section">Grown into each other</div>
          {report.overlaps.map((overlap: SegmentOverlapResponse) => (
            <div
              key={`trim-${overlap.upper_id}-${overlap.lower_id}`}
              className="pe-dd-row"
              onMouseEnter={() =>
                onHighlight?.([overlap.upper_id, overlap.lower_id])
              }
            >
              <span className="pe-dd-check__label">
                {shortLineId(overlap.upper_id)} and{" "}
                {shortLineId(overlap.lower_id)} share {percent(overlap.ratio)}{" "}
                of the smaller mask.
                {overlap.duplicate
                  ? " These are one line drawn twice; delete one rather than trimming both."
                  : ` Trimming costs ${percent(overlap.upper_loss)} and ${percent(
                      overlap.lower_loss,
                    )} of the two outlines.`}
              </span>
              {/* A duplicate pair gets no button: a cut midway between the two
                  baselines halves one line rather than separating two. */}
              {overlap.duplicate ? null : (
                <button
                  type="button"
                  className="pe-dd-btn"
                  disabled={pending === overlap.upper_id}
                  onClick={() =>
                    onApply({
                      kind: "trim",
                      upperId: overlap.upper_id,
                      lowerId: overlap.lower_id,
                    })
                  }
                >
                  {pending === overlap.upper_id ? "Trimming…" : "Trim apart"}
                </button>
              )}
            </div>
          ))}
        </>
      ) : null}

      {report.suspects.length > 0 ? (
        <>
          <div className="pe-dd-section">Might be noise</div>
          <p className="pe-dd-hint">
            Flagged, never deleted automatically. Some of this is real ink
            nobody has transcribed yet.
          </p>
          {report.suspects.map((suspect: SegmentSuspectResponse) => (
            <div
              key={`suspect-${suspect.line_id}`}
              className="pe-dd-row"
              onMouseEnter={() => onHighlight?.([suspect.line_id])}
            >
              <span className="pe-dd-check__label">
                {shortLineId(suspect.line_id)}: {suspect.reasons.join("; ")}.
              </span>
              {armedSuspect === suspect.line_id ? (
                <button
                  type="button"
                  className="pe-dd-btn pe-dd-btn--danger"
                  disabled={pending === suspect.line_id}
                  onClick={() => {
                    setArmedSuspect(null);
                    onApply({ kind: "delete", lineId: suspect.line_id });
                  }}
                >
                  {pending === suspect.line_id
                    ? "Deleting…"
                    : "Really delete it"}
                </button>
              ) : (
                <button
                  type="button"
                  className="pe-dd-btn"
                  onClick={() => setArmedSuspect(suspect.line_id)}
                >
                  Delete…
                </button>
              )}
            </div>
          ))}
        </>
      ) : null}
    </div>
  );
}
