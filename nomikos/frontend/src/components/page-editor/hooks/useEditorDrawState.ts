import { useCallback, useState } from "react";
import type { LayoutPoint, LinePoint } from "../../../api/client";
import { rectanglePoints } from "../canvasGeometry";
import type { DrawMode } from "../canvasHint";

type CommitShape = (
  kind: "rectangle" | "polygon",
  points: LinePoint[],
) => Promise<void>;

/**
 * The drawing tools: which one is active, and the shape being drawn with it.
 * Committing a finished shape is the caller's, passed in where it happens.
 */
export function useEditorDrawState() {
  const [drawMode, setDrawMode] = useState<DrawMode>("none");
  const [draftStart, setDraftStart] = useState<LayoutPoint | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<LinePoint[]>([]);

  /** Back to the Select tool with nothing half-drawn. */
  const resetDrawing = useCallback(() => {
    setDrawMode("none");
    setDraftPolygon([]);
    setDraftStart(null);
  }, []);

  /** Picking the active tool again turns it off. */
  const toggleDrawMode = useCallback((nextMode: "rectangle" | "polygon") => {
    setDrawMode((mode) => (mode === nextMode ? "none" : nextMode));
    setDraftPolygon([]);
    setDraftStart(null);
  }, []);

  const addPolygonPoint = useCallback((point: LinePoint) => {
    setDraftPolygon((current) => [...current, point]);
  }, []);

  function completeDraftPolygon(commit: CommitShape) {
    if (draftPolygon.length >= 3) void commit("polygon", draftPolygon);
    setDraftPolygon([]);
  }

  async function finishRectangle(end: LayoutPoint, commit: CommitShape) {
    if (!draftStart) return;
    await commit("rectangle", rectanglePoints(draftStart, end));
    setDraftStart(null);
  }

  return {
    drawMode,
    setDrawMode,
    draftStart,
    setDraftStart,
    draftPolygon,
    resetDrawing,
    toggleDrawMode,
    addPolygonPoint,
    completeDraftPolygon,
    finishRectangle,
    canCompletePolygon: drawMode === "polygon" && draftPolygon.length >= 3,
  };
}
