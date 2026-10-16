import { useEffect, useRef, useState, type PointerEvent } from "react";
import type { LinePoint, LineResponse } from "../../api/client";
import { useStableCallback } from "./useStableCallback";

/** Screen pixels a pressed vertex must travel before it starts to follow. */
const VERTEX_DRAG_THRESHOLD_PX = 3;

type VertexEdit = {
  segmentId: string;
  points: LinePoint[];
  draggedIndex: number | null;
  pendingVertexIndex: number | null;
};

type CommitOptions = {
  selectedVertex?: number | null;
  clearInteracting?: boolean;
};

export type VertexEditInput = {
  lines: LineResponse[];
  selectedSegmentId: string | null;
  segmentVertexEditEnabled: boolean;
  zoomLevel: number;
  /** Bumped by Esc and the Select tool: finish or drop the edit in progress. */
  commitSignal: number;
  onSelectedVertexChange: (vertexIndex: number | null) => void;
  onSegmentPointsChange: (
    segmentId: string,
    points: LinePoint[],
  ) => void | Promise<void>;
};

/** One pointer move during a press: arm the drag past the threshold, or move. */
function movedVertexEdit(
  current: VertexEdit,
  point: LinePoint,
  zoomLevel: number,
): VertexEdit {
  if (current.pendingVertexIndex !== null && current.draggedIndex === null) {
    const anchor = current.points[current.pendingVertexIndex];
    // The threshold is in screen pixels: in image units it would grow into a
    // dead zone as the page is zoomed in.
    const moved =
      Math.hypot(point[0] - anchor[0], point[1] - anchor[1]) >
      VERTEX_DRAG_THRESHOLD_PX / Math.max(zoomLevel, 0.05);
    if (!moved) return current;
    return {
      ...current,
      draggedIndex: current.pendingVertexIndex,
      pendingVertexIndex: null,
    };
  }
  if (current.draggedIndex === null) return current;
  const nextPoints = [...current.points];
  nextPoints[current.draggedIndex] = point;
  return { ...current, points: nextPoints };
}

/**
 * The selected segment's vertex handles: press, drag past a threshold, release
 * to commit; click an edge to insert a vertex; Delete to remove one.
 *
 * Every handler it returns is stable, and the effects depend only on the
 * signals that should re-run them. A parent re-render that hands in a fresh
 * `onSegmentPointsChange` therefore neither re-runs the commit effect (which
 * would cancel a press in progress) nor re-renders the memoised canvas.
 */
/** The selected segment's points: the edit in progress, else as saved. */
function pointsOnScreen(
  vertexEdit: VertexEdit | null,
  selectedSegmentId: string | null,
  lines: LineResponse[],
): LinePoint[] | null {
  if (vertexEdit?.segmentId === selectedSegmentId) return vertexEdit.points;
  return lines.find((line) => line.id === selectedSegmentId)?.points ?? null;
}

export function useVertexEdit({
  lines,
  selectedSegmentId,
  segmentVertexEditEnabled,
  zoomLevel,
  commitSignal,
  onSelectedVertexChange,
  onSegmentPointsChange,
}: VertexEditInput) {
  const [vertexEdit, setVertexEdit] = useState<VertexEdit | null>(null);
  const vertexInteractingRef = useRef(false);
  const vertexEditRef = useRef(vertexEdit);
  vertexEditRef.current = vertexEdit;
  const vertexEditPoints = pointsOnScreen(vertexEdit, selectedSegmentId, lines);
  const draggedVertexIndex = vertexEdit?.draggedIndex ?? null;
  const pendingVertexIndex = vertexEdit?.pendingVertexIndex ?? null;
  const isVertexInteracting =
    pendingVertexIndex !== null || draggedVertexIndex !== null;

  useEffect(() => {
    if (!segmentVertexEditEnabled || !selectedSegmentId) {
      setVertexEdit(null);
      vertexInteractingRef.current = false;
    }
  }, [segmentVertexEditEnabled, selectedSegmentId]);

  const selectVertex = useStableCallback(onSelectedVertexChange);

  const commitVertexEdit = useStableCallback(
    (segmentId: string, points: LinePoint[], options?: CommitOptions) => {
      setVertexEdit({
        segmentId,
        points,
        draggedIndex: null,
        pendingVertexIndex: null,
      });
      if (options && "selectedVertex" in options) {
        selectVertex(options.selectedVertex ?? null);
      }
      void Promise.resolve(onSegmentPointsChange(segmentId, points)).finally(
        () => {
          setVertexEdit((latest) =>
            latest?.segmentId === segmentId ? null : latest,
          );
          if (options?.clearInteracting) {
            vertexInteractingRef.current = false;
          }
        },
      );
    },
  );

  useEffect(() => {
    if (commitSignal === 0) return;
    const current = vertexEditRef.current;
    if (!current) return;
    if (current.draggedIndex === null) {
      vertexInteractingRef.current = false;
      setVertexEdit(null);
      return;
    }
    commitVertexEdit(current.segmentId, current.points, {
      clearInteracting: true,
    });
  }, [commitSignal, commitVertexEdit]);

  useEffect(() => {
    if (!isVertexInteracting) return;
    const finishInteraction = () => {
      const current = vertexEditRef.current;
      if (!current || current.draggedIndex === null) {
        // Click without drag: select the vertex (Delete removes it later).
        if (current && current.pendingVertexIndex !== null) {
          selectVertex(current.pendingVertexIndex);
          setVertexEdit(null);
        }
        vertexInteractingRef.current = false;
        return;
      }
      commitVertexEdit(current.segmentId, current.points, {
        selectedVertex: current.draggedIndex,
        clearInteracting: true,
      });
    };
    window.addEventListener("pointerup", finishInteraction);
    window.addEventListener("mouseup", finishInteraction);
    return () => {
      window.removeEventListener("pointerup", finishInteraction);
      window.removeEventListener("mouseup", finishInteraction);
    };
  }, [isVertexInteracting, commitVertexEdit, selectVertex]);

  const onVertexPointerDown = useStableCallback(
    (vertexIndex: number, event: PointerEvent<SVGCircleElement>) => {
      if (!selectedSegmentId) return;
      if (!vertexEditPoints || vertexEditPoints.length < 3) return;
      vertexInteractingRef.current = true;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      selectVertex(vertexIndex);
      setVertexEdit({
        segmentId: selectedSegmentId,
        points: [...vertexEditPoints],
        draggedIndex: null,
        pendingVertexIndex: vertexIndex,
      });
    },
  );

  const onVertexPointerMove = useStableCallback((point: LinePoint) => {
    setVertexEdit((current) =>
      current ? movedVertexEdit(current, point, zoomLevel) : current,
    );
  });

  const onInsertVertexOnEdge = useStableCallback((nextPoints: LinePoint[]) => {
    if (!selectedSegmentId) return;
    commitVertexEdit(selectedSegmentId, nextPoints, { selectedVertex: null });
  });

  return {
    editingSegmentId: vertexEdit?.segmentId ?? null,
    vertexEditPoints,
    draggedVertexIndex,
    pendingVertexIndex,
    isVertexInteracting,
    vertexInteractingRef,
    selectVertex,
    onVertexPointerDown,
    onVertexPointerMove,
    onInsertVertexOnEdge,
  };
}
