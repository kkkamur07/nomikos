import type { LineResponse } from "../../api/client";
import { segmentHasGroundTruth } from "./hooks/utils";
import type { DrawTool } from "./PageEditorCanvasIsland";

export type DrawMode = "none" | DrawTool;

export type CanvasHintState = {
  partLoading: boolean;
  /** 1-based page number, or null before the page is known. */
  partIndex: number | null;
  processingLabel: string | null;
  drawMode: DrawMode;
  draftPointCount: number;
  selectedSegment: LineResponse | null;
  selectedSegmentNumber: number | null;
  selectedVertexIndex: number | null;
};

function polygonHint(pointCount: number): string {
  if (pointCount === 0) {
    return "Polygon: click to place the first corner · hold Space to pan";
  }
  return `Polygon: ${pointCount} point${pointCount === 1 ? "" : "s"} · click to add · double-click or Enter to finish`;
}

/**
 * The one line under the canvas that says what a click will do now. A page
 * turn keeps the editor on screen, so this is also where it says the next
 * page's Segments are still on their way.
 */
export function canvasHintFor(state: CanvasHintState): string {
  if (state.partLoading) {
    return state.partIndex
      ? `Loading page ${state.partIndex}…`
      : "Loading page…";
  }
  if (state.processingLabel) return `${state.processingLabel}…`;
  if (state.drawMode === "polygon") return polygonHint(state.draftPointCount);
  if (state.drawMode === "rectangle") {
    return "Rectangle: drag to draw a segment · hold Space to pan";
  }
  if (state.drawMode === "region") {
    return "Region: drag around a folio or column · the segments inside join it";
  }
  const segment = state.selectedSegment;
  if (!segment) {
    return "Drag to pan · scroll to zoom · Page Up/Down for pages · select a segment to edit it";
  }
  if (state.selectedVertexIndex !== null) {
    return `Segment ${state.selectedSegmentNumber} · vertex ${state.selectedVertexIndex + 1} selected · Delete removes point · Esc deselects`;
  }
  const pairing = segmentHasGroundTruth(segment) ? "paired" : "unpaired";
  return `Segment ${state.selectedSegmentNumber} ${pairing} · click edge to add · click handle to select · Esc deselects`;
}
