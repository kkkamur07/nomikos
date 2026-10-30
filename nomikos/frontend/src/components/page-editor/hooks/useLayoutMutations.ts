import { useCallback, useRef } from "react";
import {
  api,
  type LinePoint,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
import { invalidateAfter } from "../../../api/resources";
import { cleanPolygonPoints } from "../canvasGeometry";
import { withServerOrders } from "../segmentNumbering";
import {
  errorMessage,
  layoutMutationMessage,
  showPairing,
  type PairingProgressSetter,
  type SetState,
  type TextLinesSetter,
  type TrackJobAndWait,
} from "./editorHookTypes";
import { useAutoSegment } from "./useAutoSegment";
import { useBaselineEdit } from "./useBaselineEdit";
import { isRegionEdit, useCanvasUndo } from "./useCanvasUndo";
import { mergeSavedLine, syncLayoutLinesFromSegments } from "./utils";

export type { RegionEdit } from "./useCanvasUndo";

type LayoutMutationsInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  layout: PartLayoutResponse;
  setLayout: SetState<PartLayoutResponse>;
  lines: LineResponse[];
  setLines: SetState<LineResponse[]>;
  setLineError: SetState<string | null>;
  setTextLines: TextLinesSetter;
  setPairingProgress: PairingProgressSetter;
  setPairingError: SetState<string | null>;
  selectedSegmentId: string | null;
  setSelectedSegmentId: SetState<string | null>;
  setApprovedTextDraft: SetState<string>;
  /**
   * Null only while the segment catalog is empty or failed to load: then
   * `model_id` is not sent and the backend resolves its own default. Once
   * the catalog loads the picker always holds one of its ids.
   */
  selectedSegmentModelId?: string | null;
  onDrawComplete: () => void;
  /**
   * Where a refused submission is explained. It is a standing line rather than
   * the error toast, because "no inference host had capacity" is something the
   * researcher has to act on, and a toast is gone before they can.
   */
  setSubmissionRefusal: SetState<string | null>;
  subscribeToJobCompletion?: (
    listener: (event: JobCompletionEvent) => void,
  ) => () => void;
  trackJobAndWait: TrackJobAndWait;
};

function samePoints(before: LinePoint[], after: LinePoint[]): boolean {
  return (
    before.length === after.length &&
    before.every(
      (point, index) =>
        point[0] === after[index][0] && point[1] === after[index][1],
    )
  );
}

export function useLayoutMutations({
  projectId,
  documentId,
  partId,
  layout,
  setLayout,
  lines,
  setLines,
  setLineError,
  setTextLines,
  setPairingProgress,
  setPairingError,
  selectedSegmentId,
  setSelectedSegmentId,
  setApprovedTextDraft,
  selectedSegmentModelId = null,
  onDrawComplete,
  setSubmissionRefusal,
  subscribeToJobCompletion,
  trackJobAndWait,
}: LayoutMutationsInput) {
  const linesRef = useRef(lines);
  linesRef.current = lines;

  /**
   * Called once per committed write, after the server has taken it.
   *
   * The editor keeps its own copies of the Segments and layout, so a write here
   * is visible on the page without any cache doing anything - which is exactly
   * why the two reads that are *also* copies of it were being left behind.
   */
  const notePartContentChanged = useCallback(() => {
    if (!projectId || !documentId) return;
    invalidateAfter.partContentChanged(projectId, documentId);
  }, [projectId, documentId]);

  const applyLocalLines = useCallback(
    (nextLines: LineResponse[]) => {
      // Ahead of the re-render, so a second write in the same tick builds on
      // this one rather than on the render's snapshot.
      linesRef.current = nextLines;
      setLines(nextLines);
      setLayout((current) => syncLayoutLinesFromSegments(current, nextLines));
    },
    [setLines, setLayout],
  );

  const undo = useCanvasUndo({
    projectId,
    documentId,
    partId,
    linesRef,
    applyLocalLines,
    blocks: layout.blocks,
    selectedSegmentId,
    setSelectedSegmentId,
    subscribeToJobCompletion,
  });
  const { recordEdit } = undo;

  /**
   * Take the server's reading order after a create or delete. The server
   * renumbers every Segment after the one it inserted or removed (ADR 0008)
   * and returns only that one, so the shifted orders are read back. A failed
   * read leaves the local orders, which still number the Page in the same
   * sequence, just not densely.
   */
  async function syncServerOrders() {
    if (!projectId || !documentId || !partId) return;
    try {
      const serverLines = await api.listPartLines(
        projectId,
        documentId,
        partId,
      );
      applyLocalLines(withServerOrders(linesRef.current, serverLines));
    } catch {
      // Display numbering is a dense rank and does not depend on this read.
    }
  }

  /** Region writes renumber the Page, so the layout and Segments are re-read. */
  async function reloadAfterRegionWrite(
    projectId: string,
    documentId: string,
    partId: string,
  ) {
    const [nextLayout, nextLines] = await Promise.all([
      api.getPartLayout(projectId, documentId, partId),
      api.listPartLines(projectId, documentId, partId),
    ]);
    linesRef.current = nextLines;
    setLayout(nextLayout);
    setLines(nextLines);
  }

  const baseline = useBaselineEdit({
    projectId,
    documentId,
    partId,
    layout,
    setLayout,
    setLines,
    setLineError,
    notePartContentChanged,
  });

  const autoSegment = useAutoSegment({
    projectId,
    documentId,
    partId,
    lineCount: lines.length,
    setLines,
    setLayout,
    setTextLines,
    setPairingProgress,
    setPairingError,
    setSubmissionRefusal,
    selectedSegmentModelId,
    trackJobAndWait,
    notePartContentChanged,
    clearEditStacks: undo.clearEditStacks,
    onSegmentsReloaded: () => {
      baseline.setSelectedLineId(null);
      setSelectedSegmentId(null);
      baseline.setSelectedLineSnapshot(null);
      setApprovedTextDraft("");
    },
  });

  /**
   * The pairing read that follows a committed segment write. It runs after
   * the write, outside its rollback: a failure here is a stale pairing view,
   * not a failed write, so it only reports.
   */
  async function refreshPairing(
    projectId: string,
    documentId: string,
    partId: string,
  ) {
    try {
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      showPairing(pairing, setTextLines, setPairingProgress);
    } catch (err) {
      setPairingError(errorMessage(err, "Pairing could not be refreshed."));
    }
  }

  async function replaceWithManualLine(
    kind: "rectangle" | "polygon",
    points: LinePoint[],
  ) {
    if (!projectId || !documentId || !partId) return;
    try {
      // No `order`: the server places the Segment where it sits on the Page
      // and renumbers the ones after it (ADR 0008).
      const saved = await api.createPartLine(projectId, documentId, partId, {
        kind,
        points,
      });
      // Merge into the segments as they are *now*, not as they were when the
      // request went out: an edit that landed while it was in flight must
      // survive, the way undo/redo already reads `linesRef`.
      applyLocalLines(mergeSavedLine(linesRef.current, saved));
      recordEdit({ kind: "create", line: saved });
      notePartContentChanged();
      setLineError(null);
      onDrawComplete();
      await syncServerOrders();
    } catch (err) {
      setLineError(
        layoutMutationMessage(err, "Failed to save Segment geometry."),
      );
    }
  }

  async function updateSegmentPoints(segmentId: string, points: LinePoint[]) {
    if (!projectId || !documentId || !partId) return;
    const cleanedPoints = cleanPolygonPoints(points);
    if (cleanedPoints.length < 3) {
      setLineError("A segment needs at least 3 points.");
      return;
    }
    // The segments as they are now: a second commit in the same render must
    // build on the first, not on the render's snapshot.
    const previousSegment = linesRef.current.find(
      (line) => line.id === segmentId,
    );
    if (!previousSegment) return;
    const before = previousSegment.points;
    if (samePoints(before, cleanedPoints)) return;

    applyLocalLines(
      linesRef.current.map((line) =>
        line.id === segmentId
          ? { ...line, points: cleanedPoints, source: "manual" as const }
          : line,
      ),
    );
    try {
      const saved = await api.patchPartLine(
        projectId,
        documentId,
        partId,
        segmentId,
        { points: cleanedPoints },
      );
      // Merging into the optimistic snapshot would revert any edit made while
      // the patch was in flight.
      applyLocalLines(mergeSavedLine(linesRef.current, saved));
      recordEdit({ kind: "points", segmentId, before, after: cleanedPoints });
      notePartContentChanged();
      setLineError(null);
    } catch (err) {
      // Revert only this segment, against the segments as they are now: an edit to
      // a different segment that landed while the patch was in flight must survive.
      applyLocalLines(
        linesRef.current.map((line) =>
          line.id === segmentId
            ? { ...line, points: before, source: previousSegment.source }
            : line,
        ),
      );
      setLineError(layoutMutationMessage(err));
    }
  }

  async function deleteSelectedSegment() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (
      !window.confirm(
        "Delete this Segment? Its geometry and pairing on this Page will be removed. Undo brings the Segment back, but its text will not come back.",
      )
    ) {
      return;
    }
    const deletedId = selectedSegmentId;
    const deletedLine = linesRef.current.find((line) => line.id === deletedId);
    if (!deletedLine) return;
    applyLocalLines(linesRef.current.filter((line) => line.id !== deletedId));
    setSelectedSegmentId(null);
    try {
      await api.deletePartLine(projectId, documentId, partId, deletedId);
    } catch (err) {
      // Re-insert only the segment whose delete failed, into the segments as they
      // are now, so a concurrent edit to another segment is not rolled back too.
      applyLocalLines(mergeSavedLine(linesRef.current, deletedLine));
      setLineError(layoutMutationMessage(err));
      return;
    }
    recordEdit({ kind: "delete", line: deletedLine });
    notePartContentChanged();
    setLineError(null);
    await syncServerOrders();
    await refreshPairing(projectId, documentId, partId);
  }

  async function replayEdit(direction: "undo" | "redo") {
    if (!projectId || !documentId || !partId) return;
    let edit: Awaited<ReturnType<typeof undo.replay>>;
    try {
      edit = await undo.replay(direction);
      if (!edit) return;
      if (isRegionEdit(edit)) {
        await reloadAfterRegionWrite(projectId, documentId, partId);
      }
    } catch (err) {
      setLineError(layoutMutationMessage(err));
      return;
    }
    notePartContentChanged();
    setLineError(null);
    if (isRegionEdit(edit)) return;
    if (edit.kind !== "points") await syncServerOrders();
    await refreshPairing(projectId, documentId, partId);
  }

  return {
    selectedLineId: baseline.selectedLineId,
    setSelectedLineId: baseline.setSelectedLineId,
    selectedLineSnapshot: baseline.selectedLineSnapshot,
    setSelectedLineSnapshot: baseline.setSelectedLineSnapshot,
    saveMessage: baseline.saveMessage,
    setSaveMessage: baseline.setSaveMessage,
    mutationError: baseline.mutationError,
    segmenting: autoSegment.segmenting,
    segmentMessage: autoSegment.segmentMessage,
    moveSelectedBaseline: baseline.moveSelectedBaseline,
    saveSelectedLine: baseline.saveSelectedLine,
    resetSelectedLine: baseline.resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    deleteSelectedSegment,
    undoEdit: () => replayEdit("undo"),
    redoEdit: () => replayEdit("redo"),
    recordRegionEdit: recordEdit,
    canUndo: undo.canUndo,
    canRedo: undo.canRedo,
    editUndoRevision: undo.editUndoRevision,
    runAutoSegment: autoSegment.runAutoSegment,
  };
}
