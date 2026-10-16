import {
  memo,
  useCallback,
  useMemo,
  useState,
  type MouseEvent,
  type PointerEvent,
} from "react";
import type {
  LinePoint,
  LineResponse,
  PartLayoutResponse,
} from "../../api/client";
import { AuthenticatedImage } from "../AuthenticatedImage";
import {
  canvasHandleRadius,
  canvasStrokeWidth,
  insertPolygonVertexAtClick,
  points,
  rectanglePoints,
} from "./canvasGeometry";
import {
  BaselineLayer,
  DraftPolygonLayer,
  SegmentShape,
  VertexHandles,
  type SegmentLook,
} from "./CanvasLayers";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";
import { segmentNumbersById, segmentsInNumberOrder } from "./segmentNumbering";
import { useStableCallback } from "./useStableCallback";

export type CanvasSurfaceProps = {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  selectedSegmentId: string | null;
  pairedSegmentIds: Set<string>;
  drawingRectangle: boolean;
  drawingPolygon: boolean;
  draftStart: LinePoint | null;
  draftEnd: LinePoint | null;
  draftPolygon: LinePoint[];
  draftPolygonCursor: LinePoint | null;
  onDraftPolygonCursor: (point: LinePoint | null) => void;
  settings: PageEditorCanvasSettings;
  zoomLevel: number;
  onDraftStart: (point: LinePoint) => void;
  onDraftMove: (point: LinePoint) => void;
  onRectangleDrawn: (point: LinePoint) => void;
  onPolygonPoint: (point: LinePoint) => void;
  onPolygonComplete: () => void;
  onSelectLine: (lineId: string) => void;
  onSelectSegment: (lineId: string) => void;
  segmentVertexEditEnabled: boolean;
  /**
   * Space is held, so the pointer belongs to the pan gesture no matter which
   * tool is armed. Drawing handlers stand down rather than the tool switching,
   * which is what makes the pan temporary instead of a mode change.
   */
  panOverride: boolean;
  suppressBaselineSegmentId: string | null;
  vertexEditPoints: LinePoint[] | null;
  draggedVertexIndex: number | null;
  pendingVertexIndex: number | null;
  selectedVertexIndex: number | null;
  onVertexPointerDown: (
    vertexIndex: number,
    event: PointerEvent<SVGCircleElement>,
  ) => void;
  onVertexPointerMove: (point: LinePoint) => void;
  onInsertVertexOnEdge: (nextPoints: LinePoint[]) => void;
  onSelectVertex: (vertexIndex: number | null) => void;
};

type StrokeSettings = Pick<
  PageEditorCanvasSettings,
  "overlayStrokeWidth" | "baselineStrokeWidth" | "handleSize"
>;

/** Stroke and handle sizes for one zoom level, in image units. */
function canvasScale(
  settings: StrokeSettings,
  zoomLevel: number,
  canvasMax: number,
) {
  const stroke = (base: number) =>
    canvasStrokeWidth(base, zoomLevel, settings.overlayStrokeWidth, canvasMax);
  const handle = (base: number) =>
    canvasHandleRadius(
      base,
      zoomLevel,
      settings.handleSize,
      settings.overlayStrokeWidth,
      canvasMax,
    );
  const baseline = (manual: boolean) =>
    stroke((manual ? 1.0 : 1.2) * settings.baselineStrokeWidth);
  return { stroke, handle, baseline };
}

function segmentLook(
  selected: boolean,
  paired: boolean,
  fillOpacity: number,
  stroke: (base: number) => number,
): SegmentLook {
  const fill = (r: number, g: number, b: number) =>
    `rgba(${r}, ${g}, ${b}, ${fillOpacity})`;
  if (selected) {
    return {
      fill: fill(180, 0, 0),
      stroke: "var(--red, #b40000)",
      strokeWidth: stroke(2.2),
    };
  }
  if (paired) {
    return {
      fill: fill(4, 120, 87),
      stroke: "rgba(4, 120, 87, 0.75)",
      strokeWidth: stroke(1.8),
    };
  }
  return {
    fill: fill(180, 83, 9),
    stroke: "rgba(180, 83, 9, 0.55)",
    strokeWidth: stroke(1.6),
  };
}

/** The three looks a segment can have, rebuilt only when the zoom or settings change. */
function useSegmentLooks(
  fillOpacity: number,
  stroke: (base: number) => number,
) {
  return useMemo(
    () => ({
      selected: segmentLook(true, false, fillOpacity, stroke),
      paired: segmentLook(false, true, fillOpacity, stroke),
      plain: segmentLook(false, false, fillOpacity, stroke),
    }),
    [fillOpacity, stroke],
  );
}

function surfaceCursor(
  props: Pick<
    CanvasSurfaceProps,
    | "panOverride"
    | "drawingRectangle"
    | "drawingPolygon"
    | "segmentVertexEditEnabled"
  >,
  vertexInteracting: boolean,
): string {
  if (props.panOverride) return "grab";
  if (props.drawingRectangle || props.drawingPolygon) return "crosshair";
  if (vertexInteracting) return "grabbing";
  return props.segmentVertexEditEnabled ? "default" : "grab";
}

/** Image coordinates of a pointer event on the canvas. */
function eventPoint(
  event: MouseEvent<SVGElement> | PointerEvent<SVGElement>,
  canvasWidth: number,
  canvasHeight: number,
): LinePoint {
  const svg = event.currentTarget.ownerSVGElement ?? event.currentTarget;
  const rect = svg.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return [event.clientX, event.clientY];
  }
  return [
    Math.round(((event.clientX - rect.left) / rect.width) * canvasWidth),
    Math.round(((event.clientY - rect.top) / rect.height) * canvasHeight),
  ];
}

/**
 * The SVG drawing handlers. Each stands down while Space pans, and only acts
 * for the tool that is armed.
 */
function useDrawingHandlers(
  props: CanvasSurfaceProps,
  toPoint: (
    event: MouseEvent<SVGElement> | PointerEvent<SVGElement>,
  ) => LinePoint,
  vertexInteracting: boolean,
) {
  const { panOverride, drawingRectangle, drawingPolygon, draftStart } = props;
  const drawsRectangle = !panOverride && drawingRectangle;
  const drawsPolygon = !panOverride && drawingPolygon;
  return {
    onPointerDown: (event: PointerEvent<SVGSVGElement>) => {
      if (!drawsRectangle) return;
      event.stopPropagation();
      props.onDraftStart(toPoint(event));
    },
    onPointerMove: (event: PointerEvent<SVGSVGElement>) => {
      if (panOverride) return;
      if (drawingPolygon) props.onDraftPolygonCursor(toPoint(event));
      if (vertexInteracting) {
        event.stopPropagation();
        props.onVertexPointerMove(toPoint(event));
        return;
      }
      if (!drawsRectangle || !draftStart) return;
      event.stopPropagation();
      props.onDraftMove(toPoint(event));
    },
    onPointerUp: (event: PointerEvent<SVGSVGElement>) => {
      if (!drawsRectangle || !draftStart) return;
      event.stopPropagation();
      props.onRectangleDrawn(toPoint(event));
    },
    onClick: (event: MouseEvent<SVGSVGElement>) => {
      if (!drawsPolygon) return;
      event.stopPropagation();
      props.onPolygonPoint(toPoint(event));
    },
    onDoubleClick: (event: MouseEvent<SVGSVGElement>) => {
      if (!drawsPolygon) return;
      event.stopPropagation();
      props.onPolygonComplete();
    },
  };
}

function CanvasSurfaceInner(props: CanvasSurfaceProps) {
  const {
    imageWidth,
    imageHeight,
    lines,
    selectedSegmentId,
    pairedSegmentIds,
    settings,
    zoomLevel,
    segmentVertexEditEnabled,
    vertexEditPoints,
    draggedVertexIndex,
  } = props;
  const [naturalSize, setNaturalSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const canvasWidth = naturalSize?.width ?? imageWidth;
  const canvasHeight = naturalSize?.height ?? imageHeight;
  const canvasMax = Math.max(canvasWidth, canvasHeight);

  const scale = useMemo(
    () => canvasScale(settings, zoomLevel, canvasMax),
    [settings, zoomLevel, canvasMax],
  );
  const looks = useSegmentLooks(settings.segmentFillOpacity, scale.stroke);
  const toPoint = useCallback(
    (event: MouseEvent<SVGElement> | PointerEvent<SVGElement>) =>
      eventPoint(event, canvasWidth, canvasHeight),
    [canvasWidth, canvasHeight],
  );
  const edgeHitDistance = Math.max(scale.handle(3.5), scale.stroke(10));
  const vertexInteracting =
    props.pendingVertexIndex !== null || draggedVertexIndex !== null;
  const orderedLines = useMemo(() => segmentsInNumberOrder(lines), [lines]);
  const segmentNumbers = useMemo(() => segmentNumbersById(lines), [lines]);

  const onSegmentClick = useStableCallback(
    (
      lineId: string,
      segmentPoints: LinePoint[],
      event: MouseEvent<SVGPolygonElement>,
    ) => {
      event.stopPropagation();
      const editable = lineId === selectedSegmentId && segmentVertexEditEnabled;
      if (editable && segmentPoints.length >= 3) {
        const nextPoints = insertPolygonVertexAtClick(
          segmentPoints,
          toPoint(event),
          edgeHitDistance,
        );
        if (nextPoints) {
          props.onInsertVertexOnEdge(nextPoints);
          return;
        }
      }
      props.onSelectVertex(null);
      props.onSelectSegment(lineId);
    },
  );
  const handlers = useDrawingHandlers(props, toPoint, vertexInteracting);
  const draftRectangle =
    props.drawingRectangle && props.draftStart && props.draftEnd
      ? rectanglePoints(props.draftStart, props.draftEnd)
      : null;

  return (
    <div
      style={{
        position: "relative",
        width: canvasWidth,
        height: canvasHeight,
        background: "#fff",
        boxShadow: "0 8px 40px rgba(0, 0, 0, 0.5)",
      }}
    >
      <AuthenticatedImage
        compact
        src={props.imageUrl}
        alt={props.imageAlt}
        onLoad={(event) => {
          const { naturalWidth, naturalHeight } = event.currentTarget;
          if (naturalWidth > 0 && naturalHeight > 0) {
            setNaturalSize({ width: naturalWidth, height: naturalHeight });
          }
        }}
        style={{
          display: "block",
          width: canvasWidth,
          height: canvasHeight,
          userSelect: "none",
          pointerEvents: "none",
        }}
      />
      <svg
        role="group"
        aria-label="Page geometry canvas"
        viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
        {...handlers}
        style={{
          position: "absolute",
          inset: 0,
          width: canvasWidth,
          height: canvasHeight,
          cursor: surfaceCursor(props, vertexInteracting),
          touchAction: "none",
        }}
      >
        {settings.showLayoutBlocks &&
          props.layout.blocks.map((block) => (
            <polygon
              key={block.id}
              aria-label={`Block ${block.id}`}
              points={points(block.box)}
              fill="rgba(216, 199, 161, 0.08)"
              stroke="#d8c7a1"
              strokeWidth={scale.stroke(1.2)}
            />
          ))}
        {orderedLines.map((line) => {
          const selected = line.id === selectedSegmentId;
          const paired = pairedSegmentIds.has(line.id);
          return (
            <SegmentShape
              key={line.id}
              line={line}
              segmentNumber={segmentNumbers.get(line.id)}
              selected={selected}
              paired={paired}
              segmentPoints={
                selected && vertexEditPoints && vertexEditPoints.length >= 3
                  ? vertexEditPoints
                  : line.points
              }
              look={
                selected ? looks.selected : paired ? looks.paired : looks.plain
              }
              editable={selected && segmentVertexEditEnabled}
              onClick={onSegmentClick}
              onActivate={props.onSelectSegment}
            />
          );
        })}
        {settings.showBaselines && (
          <BaselineLayer
            orderedLines={orderedLines}
            suppressSegmentId={props.suppressBaselineSegmentId}
            manualStroke={scale.baseline(true)}
            machineStroke={scale.baseline(false)}
            onSelectLine={props.onSelectLine}
          />
        )}
        {draftRectangle && (
          <polygon
            aria-label="Draft rectangle segment"
            points={points(draftRectangle)}
            fill="rgba(13, 31, 60, 0.12)"
            stroke="rgba(13, 31, 60, 0.75)"
            strokeWidth={scale.stroke(2)}
            strokeDasharray={`${scale.stroke(6)},${scale.stroke(3)}`}
          />
        )}
        {props.drawingPolygon && props.draftPolygon.length > 0 && (
          <DraftPolygonLayer
            draftPolygon={props.draftPolygon}
            cursor={props.draftPolygonCursor}
            stroke={scale.stroke}
            handleRadius={scale.handle(2.2)}
          />
        )}
        {segmentVertexEditEnabled && selectedSegmentId && vertexEditPoints && (
          <VertexHandles
            segmentId={selectedSegmentId}
            vertexPoints={vertexEditPoints}
            selectedVertexIndex={props.selectedVertexIndex}
            dragging={draggedVertexIndex !== null}
            stroke={scale.stroke}
            handleRadius={scale.handle}
            onPointerDown={props.onVertexPointerDown}
            onSelectVertex={props.onSelectVertex}
          />
        )}
      </svg>
    </div>
  );
}

export const CanvasSurface = memo(CanvasSurfaceInner);
