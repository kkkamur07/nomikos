import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type LineResponse,
  type SegmentHealthResponse,
} from "../../../api/client";
import { isAbortError } from "../../../api/errors";
import type { SegmentHealthAction } from "../PageEditorSegmentHealthPanel";
import { errorMessage } from "./editorHookTypes";

type UseSegmentHealthArgs = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  /** True while the panel is on screen. Nothing is fetched until it is. */
  open: boolean;
  setLines: (lines: LineResponse[]) => void;
};

/** The id a finding's row spins on, which is the id its button is keyed by. */
function pendingIdFor(action: SegmentHealthAction): string {
  switch (action.kind) {
    case "split":
      return action.lineId;
    case "merge":
      return action.fragmentId;
    case "trim":
      return action.upperId;
    case "delete":
      return action.lineId;
  }
}

/**
 * The segment health panel's data, fetched only while the panel is open.
 *
 * The report is a measurement of the page as it stands, so it is thrown away
 * and re-read after every apply rather than patched: applying one fix routinely
 * changes the others (splitting a spanning segment can leave two new overlaps),
 * and a panel showing findings derived from geometry that has since changed
 * would offer fixes the server is about to refuse.
 *
 * Closing the panel clears the report for the same reason. Reopening it a
 * minute later must not show what the page looked like a minute ago.
 */
export function useSegmentHealth({
  projectId,
  documentId,
  partId,
  open,
  setLines,
}: UseSegmentHealthArgs) {
  const [report, setReport] = useState<SegmentHealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [readError, setReadError] = useState<string | null>(null);
  // Why the last fix was refused, kept apart from a failed read because a
  // refusal is always followed by a *successful* re-read. Folding the two into
  // one slot lets that re-read clear the sentence explaining what just went
  // wrong, and the reviewer watches the fix do nothing without being told why.
  const [applyError, setApplyError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  // Bumped to ask for a fresh read: by opening the panel, by "Try again", and
  // by every applied fix.
  const [refreshKey, setRefreshKey] = useState(0);
  // A response that arrives after the page has moved on is dropped rather than
  // rendered. Without this, a slow report from the previous part can land on
  // top of the current one's.
  const requestSeq = useRef(0);

  useEffect(() => {
    if (!open || !projectId || !documentId || !partId) return;
    const seq = ++requestSeq.current;
    let cancelled = false;
    setLoading(true);
    setReadError(null);
    api
      .getSegmentHealth(projectId, documentId, partId)
      .then((next) => {
        if (cancelled || seq !== requestSeq.current) return;
        setReport(next);
      })
      .catch((err: unknown) => {
        if (cancelled || seq !== requestSeq.current || isAbortError(err))
          return;
        setReadError(errorMessage(err, "Could not check this page."));
      })
      .finally(() => {
        if (cancelled || seq !== requestSeq.current) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, projectId, documentId, partId, refreshKey]);

  useEffect(() => {
    if (open) return;
    setReport(null);
    setReadError(null);
    setApplyError(null);
    setPending(null);
  }, [open]);

  const refresh = useCallback(() => setRefreshKey((key) => key + 1), []);

  const apply = useCallback(
    (action: SegmentHealthAction) => {
      if (!projectId || !documentId || !partId) return;
      setPending(pendingIdFor(action));
      setApplyError(null);
      const request = () => {
        switch (action.kind) {
          case "split":
            return api.splitSpanningSegment(
              projectId,
              documentId,
              partId,
              action.lineId,
            );
          case "merge":
            return api.mergeSegmentFragment(
              projectId,
              documentId,
              partId,
              action.primaryId,
              action.fragmentId,
            );
          case "trim":
            return api.trimSegmentOverlap(
              projectId,
              documentId,
              partId,
              action.upperId,
              action.lowerId,
            );
          case "delete":
            return api.deleteSegmentSuspect(
              projectId,
              documentId,
              partId,
              action.lineId,
            );
        }
      };
      void request()
        .then((lines) => {
          // Every apply route answers with the part's whole line list, so the
          // canvas is replaced rather than reconciled: the ids that changed are
          // not always the ones the fix named.
          setLines(lines);
          refresh();
        })
        .catch((err: unknown) => {
          if (isAbortError(err)) return;
          setApplyError(errorMessage(err, "Could not apply that fix."));
          // The server refuses a fix it no longer offers, which usually means
          // the page moved underneath the panel. Re-read so the reviewer is
          // looking at what is actually there now.
          refresh();
        })
        .finally(() => setPending(null));
    },
    [projectId, documentId, partId, setLines, refresh],
  );

  // A refusal outranks a read failure: it is the newer of the two, and it is
  // the one the reviewer's own click just caused.
  return {
    report,
    loading,
    error: applyError ?? readError,
    pending,
    apply,
    refresh,
  };
}
