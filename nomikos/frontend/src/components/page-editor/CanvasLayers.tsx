import {
  memo,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
} from "react";
import type { LinePoint, LineResponse } from "../../api/client";
import { normalizeGeometryPoints, points } from "./canvasGeometry";

/** Enter or Space on a focused shape does what a click does. */
function activateWithKeyboard(
  event: KeyboardEvent<SVGElement>,
  activate: () => void,
) {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  event.stopPropagation();
  activate();
}

export type SegmentLook = {
  fill: string;
  stroke: string;
  strokeWidth: number;
};

type SegmentShapeProps = {
  line: LineResponse;
  segmentNumber: number | undefined;
  selected: boolean;
  paired: boolean;
  /** The points drawn: the vertex edit in progress, or the stored outline. */
  segmentPoints: LinePoint[];
  look: SegmentLook;
  /** Selected with vertex editing on: a click may insert a vertex. */
  editable: boolean;
  onClick: (
    lineId: string,
    segmentPoints: LinePoint[],
    event: MouseEvent<SVGPolygonElement>,
  ) => void;
  onActivate: (lineId: string) => void;
};

/**
 * One segment polygon. Memoised on its own props, so a drag that moves the
 * selected outline re-renders that one polygon, not every segment on the page.
 */
export const SegmentShape = memo(function SegmentShape({
  line,
  segmentNumber,
  selected,
  paired,
  segmentPoints,
  look,
  editable,
  onClick,
  onActivate,
}: SegmentShapeProps) {
  return (
    <polygon
      className="pe-segment-shape"
      role="button"
      tabIndex={0}
      aria-label={`Segment ${segmentNumber}${paired ? ", paired" : ""}`}
      aria-current={selected ? "true" : undefined}
      onClick={(event) => onClick(line.id, segmentPoints, event)}
      onKeyDown={(event) =>
        activateWithKeyboard(event, () => onActivate(line.id))
      }
      points={points(segmentPoints)}
      fill={look.fill}
      stroke={look.stroke}
      strokeWidth={look.strokeWidth}
      style={editable ? { pointerEvents: "all", cursor: "copy" } : undefined}
    />
  );
});

type BaselineLayerProps = {
  orderedLines: LineResponse[];
  suppressSegmentId: string | null;
  manualStroke: number;
  machineStroke: number;
  onSelectLine: (lineId: string) => void;
};

/** The baselines, drawn once per change of the lines or the zoom. */
export const BaselineLayer = memo(function BaselineLayer({
  orderedLines,
  suppressSegmentId,
  manualStroke,
  machineStroke,
  onSelectLine,
}: BaselineLayerProps) {
  return orderedLines.map((line) => {
    if (line.id === suppressSegmentId) return null;
    const baseline = normalizeGeometryPoints(line.baseline);
    if (baseline.length < 2) return null;
    const manual = Boolean(line.manual_geometry);
    return (
      <polyline
        key={`baseline-${line.id}`}
        role="button"
        tabIndex={0}
        aria-label={`Line ${line.id} baseline`}
        onClick={(event) => {
          event.stopPropagation();
          onSelectLine(line.id);
        }}
        onKeyDown={(event) =>
          activateWithKeyboard(event, () => onSelectLine(line.id))
        }
        points={baseline.map(([x, y]) => `${x},${y}`).join(" ")}
        fill="none"
        stroke={manual ? "#059669" : "#0d9488"}
        strokeLinecap="round"
        strokeWidth={manual ? manualStroke : machineStroke}
        strokeDasharray={
          manual ? undefined : `${machineStroke * 2.5},${machineStroke * 1.25}`
        }
        style={{ pointerEvents: "stroke" }}
      />
    );
  });
});

type DraftPolygonLayerProps = {
  draftPolygon: LinePoint[];
  cursor: LinePoint | null;
  stroke: (base: number) => number;
  handleRadius: number;
};

/** The polygon being drawn: its sides, the rubber band to the pointer, its vertices. */
export function DraftPolygonLayer({
  draftPolygon,
  cursor,
  stroke,
  handleRadius,
}: DraftPolygonLayerProps) {
  const last = draftPolygon[draftPolygon.length - 1];
  return (
    <>
      {draftPolygon.slice(1).map(([x, y], index) => (
        <line
          key={`draft-segment-${index + 1}`}
          x1={draftPolygon[index][0]}
          y1={draftPolygon[index][1]}
          x2={x}
          y2={y}
          stroke="rgba(13, 31, 60, 0.85)"
          strokeWidth={stroke(2)}
          strokeLinecap="round"
        />
      ))}
      {cursor && (
        <line
          aria-hidden="true"
          x1={last[0]}
          y1={last[1]}
          x2={cursor[0]}
          y2={cursor[1]}
          stroke="rgba(13, 31, 60, 0.65)"
          strokeWidth={stroke(1.8)}
          strokeDasharray={`${stroke(6)},${stroke(3)}`}
          strokeLinecap="round"
        />
      )}
      {cursor && draftPolygon.length >= 2 && (
        <line
          aria-hidden="true"
          x1={cursor[0]}
          y1={cursor[1]}
          x2={draftPolygon[0][0]}
          y2={draftPolygon[0][1]}
          stroke="rgba(4, 120, 87, 0.65)"
          strokeWidth={stroke(1.4)}
          strokeDasharray={`${stroke(5)},${stroke(3)}`}
          strokeLinecap="round"
        />
      )}
      {draftPolygon.map(([x, y], index) => (
        <circle
          key={`draft-vertex-${index}`}
          cx={x}
          cy={y}
          r={handleRadius}
          fill="#fff"
          stroke="rgba(13, 31, 60, 0.85)"
          strokeWidth={stroke(0.8)}
        />
      ))}
    </>
  );
}

type VertexHandlesProps = {
  segmentId: string;
  vertexPoints: LinePoint[];
  selectedVertexIndex: number | null;
  dragging: boolean;
  stroke: (base: number) => number;
  handleRadius: (base: number) => number;
  onPointerDown: (
    vertexIndex: number,
    event: PointerEvent<SVGCircleElement>,
  ) => void;
  onSelectVertex: (vertexIndex: number) => void;
};

/**
 * The selected segment's vertex handles. Enter and Space select a vertex, the
 * same as a click; Delete removes the selected one.
 */
export function VertexHandles({
  segmentId,
  vertexPoints,
  selectedVertexIndex,
  dragging,
  stroke,
  handleRadius,
  onPointerDown,
  onSelectVertex,
}: VertexHandlesProps) {
  return vertexPoints.map(([x, y], index) => {
    const selectedVertex = selectedVertexIndex === index;
    return (
      <circle
        key={`segment-vertex-${segmentId}-${index}`}
        className="pe-vertex-handle"
        cx={x}
        cy={y}
        r={handleRadius(selectedVertex ? 3.1 : 2.4)}
        fill={selectedVertex ? "var(--red, #b40000)" : "#fff"}
        stroke="var(--red, #b40000)"
        strokeWidth={stroke(selectedVertex ? 1.2 : 0.9)}
        style={{
          cursor: dragging ? "grabbing" : "pointer",
          pointerEvents: "all",
        }}
        role="button"
        tabIndex={0}
        aria-label={`Segment vertex ${index + 1}${selectedVertex ? ", selected" : ""} · drag to move · Delete to remove`}
        aria-current={selectedVertex ? "true" : undefined}
        onPointerDown={(event) => {
          event.stopPropagation();
          onPointerDown(index, event);
        }}
        onKeyDown={(event) =>
          activateWithKeyboard(event, () => onSelectVertex(index))
        }
      />
    );
  });
}
