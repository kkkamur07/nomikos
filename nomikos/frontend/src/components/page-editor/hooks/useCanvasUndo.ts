import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  api,
  type LayoutBlockResponse,
  type LineResponse,
} from "../../../api/client";
import type { JobCompletionEvent } from "../../../context/BackgroundJobsContext";
import {
  applyCanvasEdit,
  applyCanvasEditInverse,
  pushEditOntoStack,
  type CanvasEdit,
} from "../editUndo";
import type { SetState } from "./editorHookTypes";

type Direction = "undo" | "redo";

/**
 * A region write on the same undo stack as the Segment edits (ADR 0011). The
 * id is the region as it exists now: undo and redo re-create it under a new
 * one and record that.
 */
type RegionBox = NonNullable<LayoutBlockResponse["box"]>;

export type RegionEdit =
  | {
      kind: "region-create";
      region: {
        id: string;
        box: RegionBox;
        label: string | null;
      };
    }
  | {
      kind: "region-delete";
      region: {
        id: string;
        box: RegionBox;
        label: string | null;
      };
      /** The Segments it held, moved back into it when the delete is undone. */
      lineIds: string[];
    };

export type UndoEntry = CanvasEdit | RegionEdit;

export function isRegionEdit(entry: UndoEntry): entry is RegionEdit {
  return entry.kind === "region-create" || entry.kind === "region-delete";
}

function pushEntry(stack: UndoEntry[], entry: UndoEntry): UndoEntry[] {
  return pushEditOntoStack(stack as CanvasEdit[], entry as CanvasEdit);
}

/**
 * The region a restored Segment goes back into, when that region still
 * exists; otherwise nothing, and the server files it by where it sits.
 */
function regionOf(
  line: { block_id?: string | null },
  blocks: LayoutBlockResponse[],
) {
  const blockId = line.block_id;
  return blockId && blocks.some((block) => block.id === blockId)
    ? { block_id: blockId }
    : {};
}

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

type CanvasUndoInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  linesRef: RefObject<LineResponse[]>;
  applyLocalLines: (lines: LineResponse[]) => void;
  /** The page's regions now, so a restored Segment can go back into its own. */
  blocks: LayoutBlockResponse[];
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
  blocks: LayoutBlockResponse[];
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
  const { projectId, documentId, partId, previous, applyLocalLines, blocks } =
    context;
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
    ...regionOf(edit.line, blocks),
  });
  const restored: CanvasEdit = { kind: edit.kind, line: saved };
  applyLocalLines(apply(previous, restored));
  return restored;
}

/**
 * Take back (or redo) a region write. Undo of a create and redo of a delete
 * remove the region; the other two draw it again with the same box and
 * label, and undo of a delete puts its recorded Segments back into it.
 */
async function replayRegionEdit(
  edit: RegionEdit,
  direction: Direction,
  context: ReplayContext,
): Promise<RegionEdit> {
  const { projectId, documentId, partId, previous } = context;
  const removes = (edit.kind === "region-create") === (direction === "undo");
  if (removes) {
    await api.deletePartBlock(projectId, documentId, partId, edit.region.id);
    return edit;
  }
  const created = await api.createPartBlock(projectId, documentId, partId, {
    box: edit.region.box,
    label: edit.region.label,
    assign_contained: edit.kind === "region-create",
  });
  const region = { ...edit.region, id: created.id };
  if (edit.kind === "region-create") return { ...edit, region };
  const present = new Set(previous.map((line) => line.id));
  for (const lineId of edit.lineIds) {
    if (!present.has(lineId)) continue;
    await api.patchPartLine(projectId, documentId, partId, lineId, {
      block_id: created.id,
    });
  }
  return { ...edit, region };
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
  blocks,
  selectedSegmentId,
  setSelectedSegmentId,
  subscribeToJobCompletion,
}: CanvasUndoInput) {
  const undoStackRef = useRef<UndoEntry[]>([]);
  const redoStackRef = useRef<UndoEntry[]>([]);
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

  const recordEdit = useCallback((edit: UndoEntry) => {
    undoStackRef.current = pushEntry(undoStackRef.current, edit);
    redoStackRef.current = [];
    setEditUndoRevision((value) => value + 1);
  }, []);

  /**
   * Pops one entry, writes it, and moves it to the other stack. Resolves the
   * entry it replayed once the server took the write, or null when there was
   * none; a failed write is put back and the page shows the segments as they
   * were before the replay.
   */
  async function replay(direction: Direction): Promise<UndoEntry | null> {
    const [from, to] =
      direction === "undo"
        ? [undoStackRef, redoStackRef]
        : [redoStackRef, undoStackRef];
    const edit = from.current.pop();
    if (!edit) return null;
    if (!projectId || !documentId || !partId) {
      from.current.push(edit);
      return null;
    }
    const previous = linesRef.current;
    const context = {
      projectId,
      documentId,
      partId,
      previous,
      applyLocalLines,
      blocks,
    };
    try {
      if (isRegionEdit(edit)) {
        const done = await replayRegionEdit(edit, direction, context);
        to.current = pushEntry(to.current, done);
        return edit;
      }
      const done = await replayEdit(edit, direction, context);
      const removed = removedSegmentId(edit, direction);
      if (removed !== null && removed === selectedSegmentId) {
        setSelectedSegmentId(null);
      }
      to.current = pushEntry(to.current, done);
      return edit;
    } catch (err) {
      from.current = pushEntry(from.current, edit);
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
