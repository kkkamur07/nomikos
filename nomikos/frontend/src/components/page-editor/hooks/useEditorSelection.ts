import { useCallback, useRef, useState } from "react";
import type { LayoutLineResponse, LineResponse } from "../../../api/client";
import { removePolygonVertex } from "../canvasGeometry";
import type { SetState } from "./editorHookTypes";
import type { useLayoutMutations } from "./useLayoutMutations";
import type { usePairingState } from "./usePairingState";

type EditorSelectionInput = {
  lines: LineResponse[];
  layoutLines: LayoutLineResponse[];
  pairing: ReturnType<typeof usePairingState>;
  mutations: ReturnType<typeof useLayoutMutations>;
  selectedVertexIndex: number | null;
  setSelectedVertexIndex: SetState<number | null>;
  setStripDismissed: SetState<boolean>;
  setSelectedRegionId: SetState<string | null>;
  /** Back to the Select tool, with no menu left open. */
  resetTools: () => void;
};

/**
 * What a click, Esc or Delete does to the selection: one Segment (with
 * perhaps one of its vertices) or one baseline, never both.
 */
export function useEditorSelection({
  lines,
  layoutLines,
  pairing,
  mutations,
  selectedVertexIndex,
  setSelectedVertexIndex,
  setStripDismissed,
  setSelectedRegionId,
  resetTools,
}: EditorSelectionInput) {
  const {
    selectedSegmentId,
    setSelectedSegmentId,
    setSelectedSegmentIds,
    setApprovedTextDraft,
  } = pairing;
  const { selectedLineId, setSelectedLineId } = mutations;
  const [vertexCommitSignal, setVertexCommitSignal] = useState(0);

  // Latest selection, readable after an await: a save may resolve after the user
  // has switched segments, and the strip must not be dismissed on the new one.
  const selectedSegmentIdRef = useRef(selectedSegmentId);
  selectedSegmentIdRef.current = selectedSegmentId;

  /** ``additive`` is a shift-click or cmd/ctrl-click: toggle, do not replace. */
  function selectSegment(lineId: string, additive = false) {
    setSelectedRegionId(null);
    setSelectedLineId(null);
    mutations.setSaveMessage(null);
    setStripDismissed(false);
    setSelectedVertexIndex(null);
    pairing.selectSegment(lineId, additive);
  }

  function selectRegion(regionId: string) {
    setSelectedSegmentId(null);
    setSelectedLineId(null);
    setSelectedVertexIndex(null);
    setSelectedRegionId(regionId);
  }

  function selectLine(lineId: string) {
    const selectedLine = layoutLines.find((line) => line.id === lineId);
    setSelectedLineId(lineId);
    setSelectedSegmentId(null);
    setSelectedVertexIndex(null);
    mutations.setSelectedLineSnapshot({
      baseline: selectedLine?.baseline,
      mask: selectedLine?.mask,
    });
  }

  /** Esc and the Select tool: drop every selection and half-drawn shape. */
  const clearSelection = useCallback(() => {
    resetTools();
    setVertexCommitSignal((value) => value + 1);
    setSelectedVertexIndex(null);
    setSelectedSegmentId(null);
    setSelectedSegmentIds([]);
    setSelectedLineId(null);
    setSelectedRegionId(null);
    setApprovedTextDraft("");
    setStripDismissed(false);
  }, [
    resetTools,
    setSelectedVertexIndex,
    setSelectedSegmentId,
    setSelectedSegmentIds,
    setSelectedLineId,
    setSelectedRegionId,
    setApprovedTextDraft,
    setStripDismissed,
  ]);

  function removeSelectedVertex() {
    if (!selectedSegmentId || selectedVertexIndex === null) return;
    const segment = lines.find((line) => line.id === selectedSegmentId);
    if (!segment) return;
    const nextPoints = removePolygonVertex(segment.points, selectedVertexIndex);
    if (!nextPoints) return;
    setSelectedVertexIndex(null);
    void mutations.updateSegmentPoints(selectedSegmentId, nextPoints);
  }

  function deleteSelection() {
    if (selectedSegmentId) void mutations.deleteSelectedSegment();
    if (selectedLineId) void mutations.resetSelectedLine();
  }

  /** Close the strip after a save, unless the researcher has moved on. */
  async function dismissAfterSave(save: () => Promise<boolean>) {
    const savedSegmentId = selectedSegmentId;
    if ((await save()) && selectedSegmentIdRef.current === savedSegmentId) {
      setStripDismissed(true);
    }
  }

  return {
    hasSelection: Boolean(selectedSegmentId || selectedLineId),
    vertexCommitSignal,
    selectSegment,
    selectLine,
    selectRegion,
    clearSelection,
    removeSelectedVertex,
    deleteSelection,
    dismissAfterSave,
  };
}
