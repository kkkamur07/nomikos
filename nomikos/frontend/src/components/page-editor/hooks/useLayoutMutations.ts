import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type EnqueueJobResponse,
  type JobResponse,
  type LayoutLineResponse,
  type LinePoint,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
import { ApiError, isAbortError } from "../../../api/errors";
import { invalidateAfter } from "../../../api/resources";
import { submissionRefusalExplanation } from "../../../inference";
import { cleanPolygonPoints, offsetGeometry } from "../canvasGeometry";
import {
  applyCanvasEdit,
  applyCanvasEditInverse,
  pushEditOntoStack,
  type CanvasEdit,
} from "../editUndo";
import {
  INFERENCE_JOB_WAIT_CEILING_MS,
  type PageEditorJobKind,
} from "../jobProgress";
import { withServerOrders } from "../segmentNumbering";
import { statusMessage, type StatusMessage } from "../statusMessage";
import {
  applyLayoutLineGeometryToSegments,
  mergeSavedLine,
  syncLayoutLinesFromSegments,
} from "./utils";

function layoutMutationMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Only project members can edit layout.";
  }
  return error instanceof Error ? error.message : "Layout update failed.";
}

type LayoutMutationsInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  layout: PartLayoutResponse;
  setLayout: Dispatch<SetStateAction<PartLayoutResponse>>;
  lines: LineResponse[];
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  setLineError: Dispatch<SetStateAction<string | null>>;
  setTextLines: Dispatch<
    SetStateAction<
      { order: number; text: string; paired_line_id: string | null }[]
    >
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{
      paired_lines: number;
      total_lines: number;
      percent: number;
    }>
  >;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  selectedSegmentId: string | null;
  setSelectedSegmentId: Dispatch<SetStateAction<string | null>>;
  setApprovedTextDraft: Dispatch<SetStateAction<string>>;
  /**
   * Null means "Default": `model_id` is not sent and the backend resolves
   * the binding or the worker's own default, exactly as before the picker
   * existed.
   */
  selectedSegmentModelId?: string | null;
  onDrawComplete: () => void;
  /**
   * Where a refused submission is explained. It is a standing line rather than
   * the error toast, because "no inference host had capacity" is something the
   * researcher has to act on, and a toast is gone before they can.
   */
  setSubmissionRefusal: Dispatch<SetStateAction<string | null>>;
  subscribeToJobCompletion?: (
    listener: (event: JobCompletionEvent) => void,
  ) => () => void;
  trackJobAndWait: (
    enqueued: EnqueueJobResponse,
    meta: { label: string; kind: PageEditorJobKind },
    options?: { timeoutMs?: number },
  ) => Promise<JobResponse>;
};

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
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedLineSnapshot, setSelectedLineSnapshot] = useState<{
    baseline?: LayoutLineResponse["baseline"];
    mask?: LayoutLineResponse["mask"];
  } | null>(null);
  const [saveMessage, setSaveMessage] = useState<StatusMessage | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);
  // A count, not a flag: a superseded run unwinds while its successor is still
  // going, and must not report the page as idle on the way out.
  const [segmentRunCount, setSegmentRunCount] = useState(0);
  const segmenting = segmentRunCount > 0;
  const [segmentMessage, setSegmentMessage] = useState<StatusMessage | null>(
    null,
  );
  const undoStackRef = useRef<CanvasEdit[]>([]);
  const redoStackRef = useRef<CanvasEdit[]>([]);
  const [editUndoRevision, setEditUndoRevision] = useState(0);
  const linesRef = useRef(lines);
  linesRef.current = lines;

  useEffect(() => {
    setSelectedLineId(null);
    setSelectedLineSnapshot(null);
    setSaveMessage(null);
    setMutationError(null);
    setSegmentMessage(null);
    undoStackRef.current = [];
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, [projectId, documentId, partId]);

  /**
   * A finished job replaces this page's Segments wherever the refresh comes
   * from, and only two of those routes run inside this hook.
   *
   * ``reloadAfterSegmentation`` clears the stacks for the run this hook
   * started. The background refresh in ``usePageEditorData`` reaches the same
   * lines from a sibling hook, so an undo left over from before a job that
   * finished while the tab was in the background would name a line id the
   * refresh has already replaced. Listening to the completion directly keeps
   * the stacks the responsibility of the hook that owns them.
   */
  useEffect(() => {
    if (!partId || !subscribeToJobCompletion) return;
    return subscribeToJobCompletion((event) => {
      if (event.documentPartId !== partId) return;
      // Only a segmentation that finished replaces the Segments, and with them
      // the line ids an undo entry names. Transcription patches text onto
      // lines that are already there, and a run that failed changed nothing at
      // all; clearing the stacks for either would throw away a geometry edit
      // the researcher can still legitimately take back.
      if (event.kind !== "segmentation" || event.status !== "done") return;
      undoStackRef.current = [];
      redoStackRef.current = [];
      setEditUndoRevision((value) => value + 1);
    });
  }, [partId, subscribeToJobCompletion]);

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

  const recordEdit = useCallback((edit: CanvasEdit) => {
    undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, []);

  const applyLocalLines = useCallback(
    (nextLines: LineResponse[]) => {
      // Ahead of the re-render, so a read that resolves before it (the order
      // sync after a create) merges into these lines, not the ones before.
      linesRef.current = nextLines;
      setLines(nextLines);
      setLayout((current) => syncLayoutLinesFromSegments(current, nextLines));
    },
    [setLines, setLayout],
  );

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

  function moveSelectedBaseline(deltaY: number) {
    if (!selectedLineId) return;
    setSaveMessage(null);
    setMutationError(null);
    setLayout((current) => {
      const nextLayoutLines = current.lines.map((line) =>
        line.id === selectedLineId
          ? {
              ...line,
              baseline: offsetGeometry(line.baseline, deltaY),
            }
          : line,
      );
      setLines((segments) =>
        applyLayoutLineGeometryToSegments(segments, nextLayoutLines),
      );
      return { ...current, lines: nextLayoutLines };
    });
  }

  async function saveSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const selectedLine = layout.lines.find(
      (line) => line.id === selectedLineId,
    );
    if (!selectedLine) return;

    try {
      await api.updateLineGeometry(
        projectId,
        documentId,
        partId,
        selectedLineId,
        {
          baseline: selectedLine.baseline,
          mask: selectedLine.mask,
        },
      );
      setLayout((current) => ({
        ...current,
        lines: current.lines.map((line) =>
          line.id === selectedLineId
            ? { ...line, manual_geometry: true }
            : line,
        ),
      }));
      setLines((current) =>
        current.map((line) =>
          line.id === selectedLineId
            ? {
                ...line,
                baseline: selectedLine.baseline ?? line.baseline,
                mask: selectedLine.mask ?? line.mask,
                manual_geometry: true,
              }
            : line,
        ),
      );
      notePartContentChanged();
      setMutationError(null);
      setSaveMessage(statusMessage("Manual geometry saved"));
      setSelectedLineSnapshot({
        baseline: selectedLine.baseline,
        mask: selectedLine.mask,
      });
    } catch (err) {
      if (selectedLineSnapshot) {
        setLayout((current) => ({
          ...current,
          lines: current.lines.map((line) =>
            line.id === selectedLineId
              ? {
                  ...line,
                  baseline: selectedLineSnapshot.baseline,
                  mask: selectedLineSnapshot.mask,
                }
              : line,
          ),
        }));
        setLines((current) =>
          applyLayoutLineGeometryToSegments(current, [
            {
              id: selectedLineId,
              baseline: selectedLineSnapshot.baseline,
              mask: selectedLineSnapshot.mask,
            },
          ]),
        );
      }
      setSaveMessage(null);
      setMutationError(layoutMutationMessage(err));
    }
  }

  async function resetSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    try {
      const resetLayout = await api.resetPartLayout(
        projectId,
        documentId,
        partId,
        {
          line_ids: [selectedLineId],
        },
      );
      const nextLayout = resetLayout ?? { blocks: [], lines: [] };
      setLayout(nextLayout);
      setLines((current) =>
        applyLayoutLineGeometryToSegments(current, nextLayout.lines),
      );
      setSelectedLineSnapshot(null);
      notePartContentChanged();
      setSaveMessage(statusMessage("Layout reset"));
    } catch (err) {
      // Both call sites - the Delete key and the Reset layout button - drop the
      // returned promise, so a rejection that got this far said nothing at all:
      // a reader without write access pressed Reset and the page did not move.
      setSaveMessage(null);
      setLineError(layoutMutationMessage(err));
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
      const nextLines = mergeSavedLine(linesRef.current, saved);
      applyLocalLines(nextLines);
      recordEdit({ kind: "create", line: saved });
      notePartContentChanged();
      setLineError(null);
      onDrawComplete();
      await syncServerOrders();
    } catch (err) {
      setLineError(
        err instanceof Error ? err.message : "Failed to save Segment geometry.",
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
    const previousSegment = lines.find((line) => line.id === segmentId);
    if (!previousSegment) return;
    const before = previousSegment.points;
    const pointsUnchanged =
      before.length === cleanedPoints.length &&
      before.every(
        (point, index) =>
          point[0] === cleanedPoints[index][0] &&
          point[1] === cleanedPoints[index][1],
      );
    if (pointsUnchanged) return;

    const optimisticLines = lines.map((line) =>
      line.id === segmentId
        ? { ...line, points: cleanedPoints, source: "manual" as const }
        : line,
    );
    applyLocalLines(optimisticLines);
    try {
      const saved = await api.patchPartLine(
        projectId,
        documentId,
        partId,
        segmentId,
        {
          points: cleanedPoints,
        },
      );
      // `optimisticLines` is the snapshot from before the request; merging into
      // it would revert any edit made while the patch was in flight.
      const nextLines = mergeSavedLine(linesRef.current, saved);
      applyLocalLines(nextLines);
      recordEdit({
        kind: "points",
        segmentId,
        before,
        after: cleanedPoints,
      });
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

  async function moveSelectedSegmentRight() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    const selectedLine = lines.find((line) => line.id === selectedSegmentId);
    if (!selectedLine) return;
    const nextPoints = selectedLine.points.map(
      ([x, y]) => [x + 5, y] as LinePoint,
    );
    await updateSegmentPoints(selectedSegmentId, nextPoints);
  }

  async function deleteSelectedSegment() {
    if (!projectId || !documentId || !partId || !selectedSegmentId) return;
    if (
      !window.confirm(
        "Delete this Segment? Its geometry and pairing on this Page will be removed.",
      )
    ) {
      return;
    }
    const deletedId = selectedSegmentId;
    const deletedLine = lines.find((line) => line.id === deletedId);
    if (!deletedLine) return;
    const optimisticLines = lines.filter((line) => line.id !== deletedId);
    applyLocalLines(optimisticLines);
    setSelectedSegmentId(null);
    try {
      await api.deletePartLine(projectId, documentId, partId, deletedId);
      recordEdit({ kind: "delete", line: deletedLine });
      notePartContentChanged();
      setLineError(null);
      await syncServerOrders();
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      // Re-insert only the segment whose delete failed, into the segments as they
      // are now, so a concurrent edit to another segment is not rolled back too.
      applyLocalLines(mergeSavedLine(linesRef.current, deletedLine));
      setLineError(layoutMutationMessage(err));
    }
  }

  /**
   * The region a restored Segment goes back into, when that region still
   * exists; otherwise nothing, and the server files it by where it sits.
   */
  function regionOf(line: { block_id?: string | null }) {
    const blockId = line.block_id;
    return blockId && layout.blocks.some((block) => block.id === blockId)
      ? { block_id: blockId }
      : {};
  }

  async function undoEdit() {
    const edit = undoStackRef.current.pop();
    if (!edit || !projectId || !documentId || !partId) {
      if (edit) undoStackRef.current.push(edit);
      return;
    }
    const previous = linesRef.current;
    try {
      if (edit.kind === "points") {
        applyLocalLines(applyCanvasEditInverse(previous, edit));
        await api.patchPartLine(projectId, documentId, partId, edit.segmentId, {
          points: edit.before,
        });
        redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      } else if (edit.kind === "create") {
        applyLocalLines(applyCanvasEditInverse(previous, edit));
        await api.deletePartLine(projectId, documentId, partId, edit.line.id);
        if (selectedSegmentId === edit.line.id) setSelectedSegmentId(null);
        redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      } else {
        const saved = await api.createPartLine(projectId, documentId, partId, {
          order: edit.line.order,
          kind: edit.line.kind,
          points: edit.line.points,
          ...regionOf(edit.line),
        });
        const restored: CanvasEdit = { kind: "delete", line: saved };
        applyLocalLines(applyCanvasEditInverse(previous, restored));
        redoStackRef.current = pushEditOntoStack(
          redoStackRef.current,
          restored,
        );
      }
      notePartContentChanged();
      setLineError(null);
      setEditUndoRevision((value) => value + 1);
      if (edit.kind !== "points") await syncServerOrders();
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      applyLocalLines(previous);
      setLineError(layoutMutationMessage(err));
      setEditUndoRevision((value) => value + 1);
    }
  }

  async function redoEdit() {
    const edit = redoStackRef.current.pop();
    if (!edit || !projectId || !documentId || !partId) {
      if (edit) redoStackRef.current.push(edit);
      return;
    }
    const previous = linesRef.current;
    try {
      if (edit.kind === "points") {
        applyLocalLines(applyCanvasEdit(previous, edit));
        await api.patchPartLine(projectId, documentId, partId, edit.segmentId, {
          points: edit.after,
        });
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      } else if (edit.kind === "create") {
        const saved = await api.createPartLine(projectId, documentId, partId, {
          order: edit.line.order,
          kind: edit.line.kind,
          points: edit.line.points,
          ...regionOf(edit.line),
        });
        const created: CanvasEdit = { kind: "create", line: saved };
        applyLocalLines(applyCanvasEdit(previous, created));
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, created);
      } else {
        applyLocalLines(applyCanvasEdit(previous, edit));
        await api.deletePartLine(projectId, documentId, partId, edit.line.id);
        if (selectedSegmentId === edit.line.id) setSelectedSegmentId(null);
        undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
      }
      notePartContentChanged();
      setLineError(null);
      setEditUndoRevision((value) => value + 1);
      if (edit.kind !== "points") await syncServerOrders();
      const pairing = await api.getPagePairing(projectId, documentId, partId);
      setTextLines(pairing.text_lines);
      setPairingProgress(pairing.pairing_progress);
    } catch (err) {
      redoStackRef.current = pushEditOntoStack(redoStackRef.current, edit);
      applyLocalLines(previous);
      setLineError(layoutMutationMessage(err));
      setEditUndoRevision((value) => value + 1);
    }
  }

  /**
   * The page state that follows a finished segmentation, local or cloud: both
   * paths replace the same Segments, layout and pairing, and only the sentence
   * they report differs. Returns the Segment count for that sentence.
   *
   * Callers reach this only after the `projectId`/`documentId`/`partId` guard in
   * `runAutoSegment`.
   */
  async function reloadAfterSegmentation(): Promise<number> {
    // The segmentation already replaced every Segment on the server, so any
    // edit still on the undo/redo stack now names a line id that reload is
    // about to make up. Left there, a later undo pops it, applyCanvasEditInverse
    // silently no-ops against the new lines array, and the paired
    // patchPartLine/deletePartLine 404s into setLineError. Cleared the same
    // way the route-change effect clears it, and unconditionally: the
    // segmentation is already stored by the time a caller reaches this
    // function, whether or not the reload below succeeds.
    undoStackRef.current = [];
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);

    const [reloadedLines, reloadedLayout, pairing] = await Promise.all([
      api.listPartLines(projectId!, documentId!, partId!),
      api.getPartLayout(projectId!, documentId!, partId!),
      api.getPagePairing(projectId!, documentId!, partId!),
    ]);
    setLines(reloadedLines);
    setLayout(reloadedLayout ?? { blocks: [], lines: [] });
    setSelectedLineId(null);
    setSelectedSegmentId(null);
    setSelectedLineSnapshot(null);
    setApprovedTextDraft("");
    setTextLines(pairing.text_lines);
    setPairingProgress(pairing.pairing_progress);
    return reloadedLines.length;
  }

  /**
   * The sentence a finished segmentation reports.
   *
   * It no longer names a host: the job does that itself, on the job, which is
   * the entire user interface for **execution target** (ADR 0002). Saying it
   * twice, in two places, with two sources, is how they come to disagree.
   */
  function segmentationMessage(segmentCount: number): string {
    return `Kraken segmentation completed using raw Kraken boundaries (${segmentCount} Segment(s)).`;
  }

  async function runAutoSegment(modelId?: string | null) {
    if (!projectId || !documentId || !partId) return;
    const chosenModelId =
      modelId === undefined ? selectedSegmentModelId : modelId;
    if (
      lines.length > 0 &&
      !window.confirm(
        "Run Kraken line segmentation? Existing machine Segments on this Page will be replaced.",
      )
    ) {
      return;
    }
    setSegmentRunCount((count) => count + 1);
    setSegmentMessage(null);
    setPairingError(null);
    setSubmissionRefusal(null);

    const jobMeta = {
      label: "Kraken line segmentation",
      kind: "segmentation" as const,
    };

    try {
      const enqueued = await api.segmentPart(
        projectId,
        documentId,
        partId,
        chosenModelId ? { model_id: chosenModelId } : {},
      );
      await trackJobAndWait(enqueued, jobMeta, {
        timeoutMs: INFERENCE_JOB_WAIT_CEILING_MS,
      });
      notePartContentChanged();

      // The segmentation is stored by now and only the reload is left. A
      // failure here is a stale view of saved Segments, so it surfaces as an
      // error banner and stops: the write is already finished, and there is
      // nothing left for it to re-run.
      setSegmentMessage(
        statusMessage(segmentationMessage(await reloadAfterSegmentation())),
      );
    } catch (err) {
      // The jobs panel already reports a user cancellation.
      if (isAbortError(err)) return;
      const refusal = submissionRefusalExplanation(err);
      if (refusal) {
        setSubmissionRefusal(refusal);
        return;
      }
      setPairingError(
        err instanceof Error ? err.message : "Auto segment failed.",
      );
    } finally {
      setSegmentRunCount((count) => Math.max(0, count - 1));
    }
  }

  return {
    selectedLineId,
    setSelectedLineId,
    selectedLineSnapshot,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    segmenting,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    moveSelectedSegmentRight,
    deleteSelectedSegment,
    undoEdit,
    redoEdit,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
    editUndoRevision,
    runAutoSegment,
  };
}
