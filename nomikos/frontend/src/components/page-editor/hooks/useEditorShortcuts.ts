import { useKeyboardShortcuts } from "../../../hooks/useKeyboardShortcuts";
import type { DrawTool } from "../PageEditorCanvasIsland";

type EditorShortcutsInput = {
  selectedSegmentId: string | null;
  selectedLineId: string | null;
  selectedVertexIndex: number | null;
  onPickDrawMode: (mode: DrawTool) => void;
  onRemoveSelectedVertex: () => void;
  onDeleteSelection: () => void;
  onEscape: () => void;
  onUndo: () => void;
  onRedo: () => void;
  /** Whether a drawn polygon has enough corners for Enter to finish it. */
  canCompletePolygon: boolean;
  onCompletePolygon: () => void;
  onPreviousPage: (() => void) | undefined;
  onNextPage: (() => void) | undefined;
};

/**
 * What Delete removes: the selected vertex of the selected Segment first,
 * then the selected Segment or baseline, else nothing (the key stays the
 * browser's).
 */
function deleteHandler({
  selectedSegmentId,
  selectedLineId,
  selectedVertexIndex,
  onRemoveSelectedVertex,
  onDeleteSelection,
}: EditorShortcutsInput): (() => void) | undefined {
  if (selectedSegmentId && selectedVertexIndex !== null) {
    return onRemoveSelectedVertex;
  }
  if (selectedSegmentId || selectedLineId) return onDeleteSelection;
  return undefined;
}

/** The page editor's keyboard map. */
export function useEditorShortcuts(input: EditorShortcutsInput) {
  useKeyboardShortcuts({
    onDrawBox: () => input.onPickDrawMode("rectangle"),
    onDrawPolygon: () => input.onPickDrawMode("polygon"),
    onDelete: deleteHandler(input),
    onEscape: input.onEscape,
    onUndo: input.onUndo,
    onRedo: input.onRedo,
    onEnter: input.canCompletePolygon ? input.onCompletePolygon : undefined,
    onPreviousPage: input.onPreviousPage,
    onNextPage: input.onNextPage,
  });
}
