import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { api, type LineResponse } from "../../../api/client";
import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
import {
  applyCanvasEdit,
  applyCanvasEditInverse,
  pushEditOntoStack,
  type CanvasEdit,
} from "../editUndo";
import type { SetState } from "./editorHookTypes";

type Direction = "undo" | "redo";

/** The Segment a replay takes off the page: undoing a create, redoing a delete. */
function removedSegmentId(
  edit: CanvasEdit,
  direction: Direction,
): string | null {
  if (edit.kind === "points") return null;
  return (edit.kind === "create") === (direction === "undo")
    ? edit.line.id
    : null;
}

export type CanvasUndoInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  linesRef: RefObject<LineResponse[]>;
  applyLocalLines: (lines: LineResponse[]) => void;
  selectedSegmentId: string | null;
  setSelectedSegmentId: SetState<string | null>;
  subscribeToJobCompletion?: (
    listener: (event: JobCompletionEvent) => void,
  ) => () => void;
};

type ReplayContext = {
  projectId: string;
  documentId: string;
  partId: string;
  previous: LineResponse[];
  applyLocalLines: (lines: LineResponse[]) => void;
};

/**
 * Writes one stack entry to the server in `direction` and shows it locally.
 * Returns the entry that goes onto the opposite stack: a re-created segment
 * comes back with a new id, so the entry names the line the server holds now.
 */
async function replayEdit(
  edit: CanvasEdit,
  direction: Direction,
  context: ReplayContext,
): Promise<CanvasEdit> {
  const { projectId, documentId, partId, previous, applyLocalLines } = context;
  const apply = direction === "undo" ? applyCanvasEditInverse : applyCanvasEdit;
  if (edit.kind === "points") {
    applyLocalLines(apply(previous, edit));
    await api.patchPartLine(projectId, documentId, partId, edit.segmentId, {
      points: direction === "undo" ? edit.before : edit.after,
    });
    return edit;
  }
  // Undoing a create and redoing a delete both remove the line.
  const removes = (edit.kind === "create") === (direction === "undo");
  if (removes) {
    applyLocalLines(apply(previous, edit));
    await api.deletePartLine(projectId, documentId, partId, edit.line.id);
    return edit;
  }
  const saved = await api.createPartLine(projectId, documentId, partId, {
    order: edit.line.order,
    kind: edit.line.kind,
    points: edit.line.points,
  });
  const restored: CanvasEdit = { kind: edit.kind, line: saved };
  applyLocalLines(apply(previous, restored));
  return restored;
}

/**
 * The canvas undo and redo stacks, and the server writes that replay them.
 *
 * A finished job replaces this page's Segments wherever the refresh comes
 * from. The background refresh in `usePageEditorData` reaches the same lines
 * from a sibling hook, so an undo left over from before a segmentation that
 * finished while the tab was in the background would name a line id the
 * refresh has already replaced. Listening to the completion directly keeps the
 * stacks the responsibility of the hook that owns them.
 */
export function useCanvasUndo({
  projectId,
  documentId,
  partId,
  linesRef,
  applyLocalLines,
  selectedSegmentId,
  setSelectedSegmentId,
  subscribeToJobCompletion,
}: CanvasUndoInput) {
  const undoStackRef = useRef<CanvasEdit[]>([]);
  const redoStackRef = useRef<CanvasEdit[]>([]);
  const [editUndoRevision, setEditUndoRevision] = useState(0);

  const clearEditStacks = useCallback(() => {
    undoStackRef.current = [];
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, []);

  useEffect(() => {
    clearEditStacks();
  }, [projectId, documentId, partId, clearEditStacks]);

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
      clearEditStacks();
    });
  }, [partId, subscribeToJobCompletion, clearEditStacks]);

  const recordEdit = useCallback((edit: CanvasEdit) => {
    undoStackRef.current = pushEditOntoStack(undoStackRef.current, edit);
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, []);

  /**
   * Pops one entry, writes it, and moves it to the other stack. Resolves true
   * once the server took the write; a failed write is put back and the page
   * shows the segments as they were before the replay.
   */
  async function replay(direction: Direction): Promise<boolean> {
    const [from, to] =
      direction === "undo"
        ? [undoStackRef, redoStackRef]
        : [redoStackRef, undoStackRef];
    const edit = from.current.pop();
    if (!edit) return false;
    if (!projectId || !documentId || !partId) {
      from.current.push(edit);
      return false;
    }
    const previous = linesRef.current;
    try {
      const done = await replayEdit(edit, direction, {
        projectId,
        documentId,
        partId,
        previous,
        applyLocalLines,
      });
      const removed = removedSegmentId(edit, direction);
      if (removed !== null && removed === selectedSegmentId) {
        setSelectedSegmentId(null);
      }
      to.current = pushEditOntoStack(to.current, done);
      return true;
    } catch (err) {
      from.current = pushEditOntoStack(from.current, edit);
      applyLocalLines(previous);
      throw err;
    } finally {
      setEditUndoRevision((value) => value + 1);
    }
  }

  return {
    recordEdit,
    clearEditStacks,
    replay,
    canUndo: undoStackRef.current.length > 0,
    canRedo: redoStackRef.current.length > 0,
    editUndoRevision,
  };
}
