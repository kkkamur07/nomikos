import { useCallback, useState } from "react";

import { api, type LineResponse } from "../../../api/client";
import { ApiError } from "../../../api/errors";
import { invalidateAfter } from "../../../api/resources";
import { MAX_MERGE_SEGMENTS } from "../segmentSelection";

/**
 * Merging the segments a human picked, and the one question the server asks back.
 *
 * The route answers with the part's whole line list, because a merge renumbers
 * everything below it. So this replaces the canvas's segments rather than
 * reconciling them, exactly as the segment-health panel does with the same
 * shape: there is no correct way to patch orders a client cannot recompute.
 *
 * It does not record an undo entry, and that is deliberate rather than
 * forgotten. The editor's undo stack holds one edit at a time - a polygon
 * moved, a segment drawn, a segment deleted - and a merge is a geometry change
 * plus several deletions at once. Replaying those one press at a time would
 * walk the page through states it was never in, and the lines it re-created
 * would come back with new ids and without the pairings the merge released. The
 * segment-health fixes sit outside the stack for the same reason. What does
 * cover it is the annotation history snapshot the editor takes on demand, which
 * restores the page by id.
 */

export type MergeTextConflict = {
  lineIds: string[];
  message: string;
};

function conflictFrom(error: unknown): MergeTextConflict | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const ids = error.detail?.line_ids;
  if (!Array.isArray(ids)) return null;
  const lineIds = ids.filter((id): id is string => typeof id === "string");
  return lineIds.length > 1 ? { lineIds, message: error.message } : null;
}

function mergeMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Only project members can edit layout.";
  }
  return error instanceof Error ? error.message : "Merge failed.";
}

type UseSegmentMergeArgs = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  setLines: (lines: LineResponse[]) => void;
  setTextLines: (
    textLines: { order: number; text: string; paired_line_id: string | null }[],
  ) => void;
  setPairingProgress: (progress: {
    paired_lines: number;
    total_lines: number;
    percent: number;
  }) => void;
  /** Called once the merge has committed, so the group can be dropped. */
  onMerged: (survivingLineId: string) => void;
};

export function useSegmentMerge({
  projectId,
  documentId,
  partId,
  setLines,
  setTextLines,
  setPairingProgress,
  onMerged,
}: UseSegmentMergeArgs) {
  const [merging, setMerging] = useState(false);
  const [mergeError, setMergeError] = useState<string | null>(null);
  // The ids the server refused over, held until the reviewer answers or gives
  // up. Kept apart from `mergeError`: a conflict is a question with an answer
  // on screen, not a failure to report.
  const [conflict, setConflict] = useState<MergeTextConflict | null>(null);
  const [pendingIds, setPendingIds] = useState<string[]>([]);

  const dismissConflict = useCallback(() => {
    setConflict(null);
    setPendingIds([]);
  }, []);

  const mergeSegments = useCallback(
    async (lineIds: string[], keepTextFrom?: string) => {
      if (!projectId || !documentId || !partId) return;
      if (lineIds.length < 2) {
        setMergeError("Select at least two segments to merge.");
        return;
      }
      if (lineIds.length > MAX_MERGE_SEGMENTS) {
        setMergeError(
          `Merging is limited to ${MAX_MERGE_SEGMENTS} segments at a time.`,
        );
        return;
      }
      setMerging(true);
      setMergeError(null);
      try {
        const lines = await api.mergePartLines(
          projectId,
          documentId,
          partId,
          lineIds,
          keepTextFrom,
        );
        setLines(lines);
        setConflict(null);
        setPendingIds([]);
        const survivor =
          lines.find((line) => lineIds.includes(line.id))?.id ?? lineIds[0];
        onMerged(survivor);
        invalidateAfter.partContentChanged(projectId, documentId);
        // The merge released the pairings of the lines it deleted, so the text
        // column is stale until this lands. Same follow-up the delete makes.
        const pairing = await api.getPagePairing(projectId, documentId, partId);
        setTextLines(pairing.text_lines);
        setPairingProgress(pairing.pairing_progress);
      } catch (err) {
        const textConflict = conflictFrom(err);
        if (textConflict) {
          setConflict(textConflict);
          setPendingIds(lineIds);
        } else {
          setMergeError(mergeMessage(err));
        }
      } finally {
        setMerging(false);
      }
    },
    [
      projectId,
      documentId,
      partId,
      setLines,
      setTextLines,
      setPairingProgress,
      onMerged,
    ],
  );

  const resolveConflict = useCallback(
    (keepTextFrom: string) => {
      if (pendingIds.length < 2) return;
      void mergeSegments(pendingIds, keepTextFrom);
    },
    [pendingIds, mergeSegments],
  );

  return {
    merging,
    mergeError,
    setMergeError,
    conflict,
    dismissConflict,
    resolveConflict,
    mergeSegments,
  };
}
