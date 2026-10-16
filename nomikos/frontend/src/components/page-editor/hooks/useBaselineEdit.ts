import { useEffect, useState } from "react";
import {
  api,
  type LayoutLineResponse,
  type LineResponse,
  type PartLayoutResponse,
} from "../../../api/client";
import { offsetGeometry } from "../canvasGeometry";
import { statusMessage, type StatusMessage } from "../statusMessage";
import { layoutMutationMessage, type SetState } from "./editorHookTypes";
import { applyLayoutLineGeometryToSegments } from "./utils";

type LineGeometry = {
  baseline?: LayoutLineResponse["baseline"];
  mask?: LayoutLineResponse["mask"];
};

export type BaselineEditInput = {
  projectId: string | undefined;
  documentId: string | undefined;
  partId: string | undefined;
  layout: PartLayoutResponse;
  setLayout: SetState<PartLayoutResponse>;
  setLines: SetState<LineResponse[]>;
  setLineError: SetState<string | null>;
  notePartContentChanged: () => void;
};

/** `lines` with the line `lineId` changed by `change`; the rest untouched. */
function withLine<T extends { id: string }>(
  lines: T[],
  lineId: string,
  change: (line: T) => T,
): T[] {
  return lines.map((line) => (line.id === lineId ? change(line) : line));
}

/**
 * The selected layout line's baseline and mask: nudging it, saving it as
 * manual geometry, and resetting it to the machine layout.
 */
export function useBaselineEdit({
  projectId,
  documentId,
  partId,
  layout,
  setLayout,
  setLines,
  setLineError,
  notePartContentChanged,
}: BaselineEditInput) {
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [selectedLineSnapshot, setSelectedLineSnapshot] =
    useState<LineGeometry | null>(null);
  const [saveMessage, setSaveMessage] = useState<StatusMessage | null>(null);
  const [mutationError, setMutationError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedLineId(null);
    setSelectedLineSnapshot(null);
    setSaveMessage(null);
    setMutationError(null);
  }, [projectId, documentId, partId]);

  function moveSelectedBaseline(deltaY: number) {
    if (!selectedLineId) return;
    setSaveMessage(null);
    setMutationError(null);
    setLayout((current) => {
      const nextLayoutLines = withLine(
        current.lines,
        selectedLineId,
        (line) => ({
          ...line,
          baseline: offsetGeometry(line.baseline, deltaY),
        }),
      );
      setLines((segments) =>
        applyLayoutLineGeometryToSegments(segments, nextLayoutLines),
      );
      return { ...current, lines: nextLayoutLines };
    });
  }

  function showLineGeometry(lineId: string, geometry: LineGeometry) {
    setLayout((current) => ({
      ...current,
      lines: withLine(current.lines, lineId, (line) => ({
        ...line,
        baseline: geometry.baseline,
        mask: geometry.mask,
      })),
    }));
    setLines((current) =>
      applyLayoutLineGeometryToSegments(current, [{ id: lineId, ...geometry }]),
    );
  }

  async function saveSelectedLine() {
    if (!projectId || !documentId || !partId || !selectedLineId) return;
    const selectedLine = layout.lines.find(
      (line) => line.id === selectedLineId,
    );
    if (!selectedLine) return;
    const geometry = {
      baseline: selectedLine.baseline,
      mask: selectedLine.mask,
    };

    try {
      await api.updateLineGeometry(
        projectId,
        documentId,
        partId,
        selectedLineId,
        geometry,
      );
      setLayout((current) => ({
        ...current,
        lines: withLine(current.lines, selectedLineId, (line) => ({
          ...line,
          manual_geometry: true,
        })),
      }));
      setLines((current) =>
        withLine(current, selectedLineId, (line) => ({
          ...line,
          baseline: selectedLine.baseline ?? line.baseline,
          mask: selectedLine.mask ?? line.mask,
          manual_geometry: true,
        })),
      );
      notePartContentChanged();
      setMutationError(null);
      setSaveMessage(statusMessage("Manual geometry saved"));
      setSelectedLineSnapshot(geometry);
    } catch (err) {
      if (selectedLineSnapshot) {
        showLineGeometry(selectedLineId, selectedLineSnapshot);
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
        { line_ids: [selectedLineId] },
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

  return {
    selectedLineId,
    setSelectedLineId,
    selectedLineSnapshot,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
  };
}
