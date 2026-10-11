import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type Ref,
  type RefObject,
} from "react";
import {
  TransformComponent,
  TransformWrapper,
  type ReactZoomPanPinchContentRef,
  type ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch";
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
  normalizeGeometryPoints,
  points,
  rectanglePoints,
  removePolygonVertex,
} from "./canvasGeometry";
import type { PageEditorCanvasSettings } from "./pageEditorSettings";
import { segmentNumbersById, segmentsInNumberOrder } from "./segmentNumbering";
import { resetPanVelocityTracking } from "../../utils/zoomPanVelocity";
import { useZoomLayerHint } from "../../utils/zoomLayerHint";
import { PageEditorCanvasIsland } from "./PageEditorCanvasIsland";
import { useSmoothWheelZoom } from "./useSmoothWheelZoom";

const ZOOM_ANIMATION_MS = 220;
const ZOOM_BUTTON_STEP = 0.12;
const MIN_SCALE = 0.15;
const MAX_SCALE = 8;
/**
 * The island is 40px tall and sits 14px off the bottom edge. Fitting has to
 * treat that strip as gone, or a fitted page parks its final lines behind it.
 */
const ISLAND_RESERVED_PX = 70;
const FIT_PADDING_PX = 24;
/** Screen pixels a pressed vertex must travel before it starts to follow. */
const VERTEX_DRAG_THRESHOLD_PX = 3;
/** A press that travels this far (screen px) is a pan, not a click. */
const PAN_CLICK_THRESHOLD_PX = 4;
/** Animation for a programmatic "bring this segment into view" request. */
const FOCUS_ANIMATION_MS = 200;

/**
 * The typings claim the ref carries `state`, but at runtime it is undefined.
 * The live transform lives at `instance.transformState`.
 */
function currentScale(viewport: ReactZoomPanPinchRef): number {
  const live = viewport as unknown as {
    instance?: { transformState?: { scale?: number } };
    state?: { scale?: number };
  };
  return live.instance?.transformState?.scale ?? live.state?.scale ?? 1;
}

/** Write through to an optional forwarded ref without disturbing the local one. */
function assignRef<T>(ref: Ref<T> | undefined, value: T): void {
  if (!ref) return;
  if (typeof ref === "function") {
    ref(value);
  } else {
    (ref as RefObject<T | null>).current = value;
  }
}

type CanvasSurfaceProps = {
  imageUrl: string;
  imageAlt: string;
  imageWidth: number;
  imageHeight: number;
  layout: PartLayoutResponse;
  lines: LineResponse[];
  selectedSegmentId: string | null;
  pairedSegmentIds: Set<string>;
  hoveredSegmentId?: string | null;
  onHoverSegment?: (lineId: string | null) => void;
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
  onRemoveVertex: (vertexIndex: number) => void;
  onSelectVertex: (vertexIndex: number | null) => void;
};

function CanvasSurfaceInner({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  selectedSegmentId,
  pairedSegmentIds,
  hoveredSegmentId,
  onHoverSegment,
  drawingRectangle,
  drawingPolygon,
  draftStart,
  draftEnd,
  draftPolygon,
  draftPolygonCursor,
  onDraftPolygonCursor,
  settings,
  zoomLevel,
  panOverride,
  onDraftStart,
  onDraftMove,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
  segmentVertexEditEnabled,
  suppressBaselineSegmentId,
  vertexEditPoints,
  draggedVertexIndex,
  pendingVertexIndex,
  selectedVertexIndex,
  onVertexPointerDown,
  onVertexPointerMove,
  onInsertVertexOnEdge,
  onRemoveVertex,
  onSelectVertex,
}: CanvasSurfaceProps) {
  const [naturalSize, setNaturalSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const canvasWidth = naturalSize?.width ?? imageWidth;
  const canvasHeight = naturalSize?.height ?? imageHeight;
  const canvasMax = Math.max(canvasWidth, canvasHeight);

  const strokeWidth = (base: number) =>
    canvasStrokeWidth(base, zoomLevel, settings.overlayStrokeWidth, canvasMax);
  const baselineStroke = (manual: boolean) =>
    canvasStrokeWidth(
      (manual ? 1.0 : 1.2) * settings.baselineStrokeWidth,
      zoomLevel,
      settings.overlayStrokeWidth,
      canvasMax,
    );
  const handleRadius = (base: number) =>
    canvasHandleRadius(
      base,
      zoomLevel,
      settings.handleSize,
      settings.overlayStrokeWidth,
      canvasMax,
    );
  const segmentFill = (r: number, g: number, b: number) =>
    `rgba(${r}, ${g}, ${b}, ${settings.segmentFillOpacity})`;

  const eventPoint = (
    event: MouseEvent<SVGElement> | PointerEvent<SVGElement>,
  ): LinePoint => {
    const svg = event.currentTarget.ownerSVGElement ?? event.currentTarget;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return [event.clientX, event.clientY];
    }
    return [
      Math.round(((event.clientX - rect.left) / rect.width) * canvasWidth),
      Math.round(((event.clientY - rect.top) / rect.height) * canvasHeight),
    ];
  };

  const edgeHitDistance = Math.max(handleRadius(3.5), strokeWidth(10));

  const draftRectangle =
    drawingRectangle && draftStart && draftEnd
      ? rectanglePoints(draftStart, draftEnd)
      : null;
  const isDraggingVertex = draggedVertexIndex !== null;
  const isVertexInteracting = pendingVertexIndex !== null || isDraggingVertex;
  const orderedLines = useMemo(() => segmentsInNumberOrder(lines), [lines]);
  const segmentNumbers = useMemo(() => segmentNumbersById(lines), [lines]);
  const activateWithKeyboard = (
    event: KeyboardEvent<SVGElement>,
    activate: () => void,
  ) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    activate();
  };

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
        src={imageUrl}
        alt={imageAlt}
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
        onPointerDown={(event) => {
          if (panOverride) return;
          if (!drawingRectangle) return;
          event.stopPropagation();
          onDraftStart(eventPoint(event));
        }}
        onPointerMove={(event) => {
          if (panOverride) return;
          if (drawingPolygon) {
            onDraftPolygonCursor(eventPoint(event));
          }
          if (isVertexInteracting) {
            event.stopPropagation();
            onVertexPointerMove(eventPoint(event));
            return;
          }
          if (!drawingRectangle || !draftStart) return;
          event.stopPropagation();
          onDraftMove(eventPoint(event));
        }}
        onPointerUp={(event) => {
          if (panOverride) return;
          if (!drawingRectangle || !draftStart) return;
          event.stopPropagation();
          onRectangleDrawn(eventPoint(event));
        }}
        onClick={(event) => {
          if (panOverride) return;
          if (!drawingPolygon) return;
          event.stopPropagation();
          onPolygonPoint(eventPoint(event));
        }}
        onDoubleClick={(event) => {
          if (panOverride) return;
          if (!drawingPolygon) return;
          event.stopPropagation();
          onPolygonComplete();
        }}
        style={{
          position: "absolute",
          inset: 0,
          width: canvasWidth,
          height: canvasHeight,
          cursor: panOverride
            ? "grab"
            : drawingRectangle || drawingPolygon
              ? "crosshair"
              : isVertexInteracting
                ? "grabbing"
                : segmentVertexEditEnabled
                  ? "default"
                  : "grab",
          touchAction: "none",
        }}
      >
        {settings.showLayoutBlocks &&
          layout.blocks.map((block) => (
            <polygon
              key={block.id}
              aria-label={`Block ${block.id}`}
              points={points(block.box)}
              fill="rgba(216, 199, 161, 0.08)"
              stroke="#d8c7a1"
              strokeWidth={strokeWidth(1.2)}
            />
          ))}
        {orderedLines.map((line) => {
          const selected = line.id === selectedSegmentId;
          const paired = pairedSegmentIds.has(line.id);
          const hovered = line.id === hoveredSegmentId;
          const fill = selected
            ? segmentFill(180, 0, 0)
            : paired
              ? segmentFill(4, 120, 87)
              : segmentFill(180, 83, 9);
          const strokeColor = selected
            ? "var(--red, #b40000)"
            : paired
              ? "rgba(4, 120, 87, 0.75)"
              : "rgba(180, 83, 9, 0.55)";
          const segmentPoints =
            selected && vertexEditPoints && vertexEditPoints.length >= 3
              ? vertexEditPoints
              : line.points;
          const baseWidth = selected ? 2.2 : paired ? 1.8 : 1.6;
          return (
            <polygon
              key={line.id}
              className={`pe-segment-shape${hovered ? " is-hovered" : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`Segment ${segmentNumbers.get(line.id)}${paired ? ", paired" : ""}`}
              aria-current={selected ? "true" : undefined}
              onMouseEnter={() => onHoverSegment?.(line.id)}
              onMouseLeave={() => onHoverSegment?.(null)}
              onClick={(event) => {
                event.stopPropagation();
                if (
                  selected &&
                  segmentVertexEditEnabled &&
                  segmentPoints.length >= 3
                ) {
                  const click = eventPoint(event);
                  const nextPoints = insertPolygonVertexAtClick(
                    segmentPoints,
                    click,
                    edgeHitDistance,
                  );
                  if (nextPoints) {
                    onInsertVertexOnEdge(nextPoints);
                    return;
                  }
                }
                onSelectVertex(null);
                onSelectSegment(line.id);
              }}
              onKeyDown={(event) =>
                activateWithKeyboard(event, () => onSelectSegment(line.id))
              }
              points={points(segmentPoints)}
              fill={fill}
              stroke={strokeColor}
              strokeWidth={strokeWidth(
                hovered ? Math.max(baseWidth, 2.4) : baseWidth,
              )}
              style={
                selected && segmentVertexEditEnabled
                  ? { pointerEvents: "all", cursor: "copy" }
                  : undefined
              }
            />
          );
        })}
        {settings.showBaselines &&
          orderedLines.map((line) => {
            if (line.id === suppressBaselineSegmentId) return null;
            if (normalizeGeometryPoints(line.baseline).length < 2) return null;
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
                points={points(line.baseline)}
                fill="none"
                stroke={line.manual_geometry ? "#059669" : "#0d9488"}
                strokeLinecap="round"
                strokeWidth={baselineStroke(Boolean(line.manual_geometry))}
                strokeDasharray={
                  line.manual_geometry
                    ? undefined
                    : `${baselineStroke(false) * 2.5},${baselineStroke(false) * 1.25}`
                }
                style={{ pointerEvents: "stroke" }}
              />
            );
          })}
        {draftRectangle && (
          <polygon
            aria-label="Draft rectangle segment"
            points={points(draftRectangle)}
            fill="rgba(13, 31, 60, 0.12)"
            stroke="rgba(13, 31, 60, 0.75)"
            strokeWidth={strokeWidth(2)}
            strokeDasharray={`${strokeWidth(6)},${strokeWidth(3)}`}
          />
        )}
        {drawingPolygon && draftPolygon.length > 0 && (
          <>
            {draftPolygon.map(([x, y], index) => {
              if (index === 0) return null;
              const [prevX, prevY] = draftPolygon[index - 1];
              return (
                <line
                  key={`draft-segment-${index}`}
                  x1={prevX}
                  y1={prevY}
                  x2={x}
                  y2={y}
                  stroke="rgba(13, 31, 60, 0.85)"
                  strokeWidth={strokeWidth(2)}
                  strokeLinecap="round"
                />
              );
            })}
            {draftPolygonCursor && draftPolygon.length > 0 && (
              <>
                <line
                  aria-hidden="true"
                  x1={draftPolygon[draftPolygon.length - 1][0]}
                  y1={draftPolygon[draftPolygon.length - 1][1]}
                  x2={draftPolygonCursor[0]}
                  y2={draftPolygonCursor[1]}
                  stroke="rgba(13, 31, 60, 0.65)"
                  strokeWidth={strokeWidth(1.8)}
                  strokeDasharray={`${strokeWidth(6)},${strokeWidth(3)}`}
                  strokeLinecap="round"
                />
                {draftPolygon.length >= 2 && (
                  <line
                    aria-hidden="true"
                    x1={draftPolygonCursor[0]}
                    y1={draftPolygonCursor[1]}
                    x2={draftPolygon[0][0]}
                    y2={draftPolygon[0][1]}
                    stroke="rgba(4, 120, 87, 0.65)"
                    strokeWidth={strokeWidth(1.4)}
                    strokeDasharray={`${strokeWidth(5)},${strokeWidth(3)}`}
                    strokeLinecap="round"
                  />
                )}
              </>
            )}
            {draftPolygon.map(([x, y], index) => (
              <circle
                key={`draft-vertex-${index}`}
                cx={x}
                cy={y}
                r={handleRadius(2.2)}
                fill="#fff"
                stroke="rgba(13, 31, 60, 0.85)"
                strokeWidth={strokeWidth(0.8)}
              />
            ))}
          </>
        )}
        {segmentVertexEditEnabled &&
          selectedSegmentId &&
          vertexEditPoints &&
          vertexEditPoints.map(([x, y], index) => {
            const selectedVertex = selectedVertexIndex === index;
            return (
              <circle
                key={`segment-vertex-${selectedSegmentId}-${index}`}
                className="pe-vertex-handle"
                cx={x}
                cy={y}
                r={handleRadius(selectedVertex ? 3.1 : 2.4)}
                fill={selectedVertex ? "var(--red, #b40000)" : "#fff"}
                stroke="var(--red, #b40000)"
                strokeWidth={strokeWidth(selectedVertex ? 1.2 : 0.9)}
                style={{
                  cursor: isDraggingVertex ? "grabbing" : "pointer",
                  pointerEvents: "all",
                }}
                role="button"
                tabIndex={0}
                aria-label={`Segment vertex ${index + 1}${selectedVertex ? ", selected" : ""} · drag to move · Delete to remove`}
                aria-current={selectedVertex ? "true" : undefined}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onVertexPointerDown(index, event);
                }}
                onKeyDown={(event) =>
                  activateWithKeyboard(event, () => onRemoveVertex(index))
                }
              />
            );
          })}
      </svg>
    </div>
  );
}

const CanvasSurface = memo(CanvasSurfaceInner);

type PageEditorCanvasProps = Omit<
  CanvasSurfaceProps,
  | "draftEnd"
  | "onDraftMove"
  | "zoomLevel"
  | "suppressBaselineSegmentId"
  | "vertexEditPoints"
  | "draggedVertexIndex"
  | "pendingVertexIndex"
  | "onVertexPointerDown"
  | "onVertexPointerMove"
  | "onInsertVertexOnEdge"
  | "onRemoveVertex"
  | "draftPolygonCursor"
  | "onDraftPolygonCursor"
  | "onSelectVertex"
  | "panOverride"
> & {
  /** Tool controls for the on-canvas island. */
  onSelectTool: () => void;
  onPickDrawMode: (mode: "rectangle" | "polygon") => void;
  canDelete: boolean;
  onDeleteSelected: () => void;
  selectedVertexIndex: number | null;
  onSelectedVertexChange: (vertexIndex: number | null) => void;
  commitSignal: number;
  onSegmentPointsChange: (
    segmentId: string,
    points: LinePoint[],
  ) => void | Promise<void>;
  /** Exposes the internal zoom/pan viewport so a sibling pane can link to it. */
  viewportRef?: Ref<ReactZoomPanPinchRef>;
  /** Reports zoom/pan state after every transform, for linked viewports. */
  onTransformed?: (state: {
    scale: number;
    positionX: number;
    positionY: number;
  }) => void;
  /** Bring this segment into view. Handled once per nonce, then ignored. */
  focusRequest?: { segmentId: string; nonce: number } | null;
};

export function PageEditorCanvas({
  imageUrl,
  imageAlt,
  imageWidth,
  imageHeight,
  layout,
  lines,
  selectedSegmentId,
  pairedSegmentIds,
  hoveredSegmentId,
  onHoverSegment,
  viewportRef,
  onTransformed,
  focusRequest,
  drawingRectangle,
  drawingPolygon,
  draftStart,
  draftPolygon,
  settings,
  segmentVertexEditEnabled,
  onSelectTool,
  onPickDrawMode,
  canDelete,
  onDeleteSelected,
  selectedVertexIndex,
  onSelectedVertexChange,
  commitSignal,
  onDraftStart,
  onRectangleDrawn,
  onPolygonPoint,
  onPolygonComplete,
  onSelectLine,
  onSelectSegment,
  onSegmentPointsChange,
}: PageEditorCanvasProps) {
  const [zoomLevel, setZoomLevel] = useState(1);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const { transforming, markTransforming } = useZoomLayerHint();
  const [draftEnd, setDraftEnd] = useState<LinePoint | null>(null);
  const [draftPolygonCursor, setDraftPolygonCursor] =
    useState<LinePoint | null>(null);
  const [vertexEdit, setVertexEdit] = useState<{
    segmentId: string;
    points: LinePoint[];
    draggedIndex: number | null;
    pendingVertexIndex: number | null;
  } | null>(null);
  const transformRef = useRef<ReactZoomPanPinchRef>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const panStartRef = useRef<{ x: number; y: number } | null>(null);
  const panMovedRef = useRef(false);
  const vertexInteractingRef = useRef(false);
  const vertexEditRef = useRef(vertexEdit);
  vertexEditRef.current = vertexEdit;
  const canvasWidth = imageWidth;
  const canvasHeight = imageHeight;
  const isDrawing = drawingRectangle || drawingPolygon;
  const isDraggingVertex =
    vertexEdit?.draggedIndex !== null && vertexEdit?.draggedIndex !== undefined;
  const isVertexInteracting =
    vertexEdit?.pendingVertexIndex !== null &&
    vertexEdit?.pendingVertexIndex !== undefined
      ? true
      : isDraggingVertex;
  const selectedSegment = lines.find((line) => line.id === selectedSegmentId);
  const vertexEditPoints =
    vertexEdit?.segmentId === selectedSegmentId
      ? vertexEdit.points
      : (selectedSegment?.points ?? null);

  useEffect(() => {
    if (!drawingRectangle) setDraftEnd(null);
  }, [drawingRectangle]);

  useEffect(() => {
    if (!drawingPolygon) setDraftPolygonCursor(null);
  }, [drawingPolygon]);

  useEffect(() => {
    if (!segmentVertexEditEnabled || !selectedSegmentId) {
      setVertexEdit(null);
      vertexInteractingRef.current = false;
    }
  }, [segmentVertexEditEnabled, selectedSegmentId]);

  const commitPendingVertexEdit = (
    pending: {
      segmentId: string;
      points: LinePoint[];
      draggedIndex: null;
      pendingVertexIndex: null;
    },
    options?: {
      selectedVertex?: number | null;
      clearInteracting?: boolean;
    },
  ) => {
    setVertexEdit(pending);
    if (options && "selectedVertex" in options) {
      onSelectedVertexChange(options.selectedVertex ?? null);
    }
    void Promise.resolve(
      onSegmentPointsChange(pending.segmentId, pending.points),
    ).finally(() => {
      setVertexEdit((latest) =>
        latest?.segmentId === pending.segmentId ? null : latest,
      );
      if (options?.clearInteracting) {
        vertexInteractingRef.current = false;
      }
    });
  };

  useEffect(() => {
    if (commitSignal === 0) return;
    const current = vertexEditRef.current;
    if (!current) return;
    if (current.draggedIndex === null) {
      vertexInteractingRef.current = false;
      setVertexEdit(null);
      return;
    }
    commitPendingVertexEdit(
      {
        segmentId: current.segmentId,
        points: current.points,
        draggedIndex: null,
        pendingVertexIndex: null,
      },
      { clearInteracting: true },
    );
  }, [commitSignal, onSegmentPointsChange]);

  useEffect(() => {
    if (!isVertexInteracting) return;
    const finishInteraction = () => {
      const current = vertexEditRef.current;
      if (!current) {
        vertexInteractingRef.current = false;
        return;
      }

      // Click without drag: select the vertex (Delete removes it later).
      if (
        current.pendingVertexIndex !== null &&
        current.draggedIndex === null
      ) {
        onSelectedVertexChange(current.pendingVertexIndex);
        vertexInteractingRef.current = false;
        setVertexEdit(null);
        return;
      }

      if (current.draggedIndex === null) {
        vertexInteractingRef.current = false;
        return;
      }
      commitPendingVertexEdit(
        {
          segmentId: current.segmentId,
          points: current.points,
          draggedIndex: null,
          pendingVertexIndex: null,
        },
        {
          selectedVertex: current.draggedIndex,
          clearInteracting: true,
        },
      );
    };
    window.addEventListener("pointerup", finishInteraction);
    window.addEventListener("mouseup", finishInteraction);
    return () => {
      window.removeEventListener("pointerup", finishInteraction);
      window.removeEventListener("mouseup", finishInteraction);
    };
  }, [isVertexInteracting, onSegmentPointsChange, onSelectedVertexChange]);

  /**
   * Hold Space to pan from any tool.
   *
   * Without this the only way to reach another part of the page mid-drawing is
   * to disarm the tool, pan, and arm it again, which is why a drawing tool and
   * a pan gesture kept fighting over the left button. Every canvas tool a
   * researcher already uses (Figma, Photoshop, Excalidraw, tldraw) binds Space
   * to exactly this, so it costs nothing to learn.
   *
   * Typing a space into the transcription strip must not pan the page, hence
   * the guard below. preventDefault stops the browser scrolling the pane out
   * from under the gesture.
   */
  useEffect(() => {
    /**
     * Whatever has focus gets Space before the canvas does.
     *
     * A field types it, and a focused button is activated by it - Space is how
     * half the keyboard world presses a button, and swallowing it here would
     * make the page rail and the pager reachable by Tab but dead on arrival.
     * Space only means "pan" when nothing else has a claim on it.
     */
    function ownsSpaceKey(target: EventTarget | null): boolean {
      if (!(target instanceof HTMLElement)) return false;
      if (target.isContentEditable) return true;
      const tag = target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT")
        return true;
      if (tag === "BUTTON" || tag === "SUMMARY") return true;
      return target.getAttribute("role") === "button";
    }
    function handleKeyDown(event: globalThis.KeyboardEvent) {
      if (event.code !== "Space" || event.repeat) return;
      if (ownsSpaceKey(event.target)) return;
      event.preventDefault();
      setSpaceHeld(true);
    }
    function handleKeyUp(event: globalThis.KeyboardEvent) {
      if (event.code !== "Space") return;
      setSpaceHeld(false);
    }
    // Alt-tabbing away with Space down would otherwise leave the canvas stuck
    // in pan override, because the keyup lands on another window.
    function releaseOnBlur() {
      setSpaceHeld(false);
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", releaseOnBlur);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", releaseOnBlur);
    };
  }, []);
  useSmoothWheelZoom(hostRef, transformRef, {
    minScale: MIN_SCALE,
    maxScale: MAX_SCALE,
    speed: settings.wheelZoomSpeed,
  });

  const zoomAnimated = (direction: "in" | "out") => {
    const ref = transformRef.current;
    if (!ref) return;
    if (direction === "in") ref.zoomIn(ZOOM_BUTTON_STEP, ZOOM_ANIMATION_MS);
    else ref.zoomOut(ZOOM_BUTTON_STEP, ZOOM_ANIMATION_MS);
  };

  /**
   * Fit the whole folio into the visible canvas.
   *
   * centerView only re-centres at the current scale, which is not what a
   * button called "Fit page to view" promises. This scales to the pane and
   * keeps the island's own height out of the usable area, because the last
   * line of a folio matters as much as the first and must not land under the
   * toolbar the moment someone presses fit.
   */
  const fitToView = () => {
    const ref = transformRef.current;
    const wrapper = ref?.instance.wrapperComponent;
    if (!ref || !wrapper) return;
    const viewWidth = wrapper.offsetWidth;
    const viewHeight = wrapper.offsetHeight;
    if (!viewWidth || !viewHeight || !canvasWidth || !canvasHeight) return;

    const usableWidth = Math.max(1, viewWidth - FIT_PADDING_PX * 2);
    const usableHeight = Math.max(
      1,
      viewHeight - ISLAND_RESERVED_PX - FIT_PADDING_PX,
    );
    const scale = Math.min(
      MAX_SCALE,
      Math.max(
        MIN_SCALE,
        Math.min(usableWidth / canvasWidth, usableHeight / canvasHeight),
      ),
    );
    ref.setTransform(
      (viewWidth - canvasWidth * scale) / 2,
      (viewHeight - ISLAND_RESERVED_PX - canvasHeight * scale) / 2,
      scale,
      ZOOM_ANIMATION_MS,
    );
  };

  /**
   * Keep the internal viewport ref working while also exposing it to a
   * sibling pane through `viewportRef`.
   */
  const handleViewportRef = (instance: ReactZoomPanPinchContentRef | null) => {
    const viewport = instance as unknown as ReactZoomPanPinchRef | null;
    transformRef.current = viewport;
    assignRef(viewportRef, viewport);
  };

  /**
   * Centre the requested segment in the visible canvas at the current scale.
   * Keyed on the nonce so each request pans once; a missing segment or
   * viewport is silently ignored.
   */
  const focusNonce = focusRequest?.nonce;
  useEffect(() => {
    if (!focusRequest) return;
    const segment = lines.find((line) => line.id === focusRequest.segmentId);
    if (!segment) return;
    let segmentPoints = normalizeGeometryPoints(segment.points);
    if (segmentPoints.length === 0) {
      segmentPoints = normalizeGeometryPoints(segment.mask);
    }
    if (segmentPoints.length === 0) return;
    const viewport = transformRef.current;
    const host = hostRef.current;
    if (!viewport || !host) return;
    const viewWidth = host.clientWidth;
    const viewHeight = host.clientHeight;
    if (!viewWidth || !viewHeight) return;
    const xs = segmentPoints.map(([x]) => x);
    const ys = segmentPoints.map(([, y]) => y);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const scale = currentScale(viewport);
    viewport.setTransform(
      viewWidth / 2 - centerX * scale,
      viewHeight / 2 - centerY * scale,
      scale,
      FOCUS_ANIMATION_MS,
    );
    // Keyed on the nonce by design: lines and refs are read, not watched.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusNonce]);

  return (
    <div
      className={`pe-canvas-host${spaceHeld ? " pe-canvas-host--panning" : ""}${
        transforming ? " pe-canvas-host--transforming" : ""
      }`}
      ref={hostRef}
      onContextMenu={(event) => event.preventDefault()}
      // Every new press starts out as a click; only panning past the threshold
      // below turns it into a drag.
      onPointerDownCapture={() => {
        panMovedRef.current = false;
      }}
      // A drag that panned the page ends in a click on whatever is under the
      // pointer; swallow it so releasing over a segment does not select it.
      onClickCapture={(event) => {
        if (!panMovedRef.current) return;
        panMovedRef.current = false;
        event.stopPropagation();
      }}
    >
      <TransformWrapper
        ref={handleViewportRef}
        initialScale={1}
        minScale={MIN_SCALE}
        maxScale={MAX_SCALE}
        centerOnInit={false}
        limitToBounds={false}
        // Wheel zoom is handled by useSmoothWheelZoom (cursor-anchored glide).
        wheel={{ disabled: true }}
        pinch={{ step: 5, disabled: isDrawing }}
        doubleClick={{
          disabled: isDrawing,
          step: 0.65,
          mode: "zoomIn",
          animationTime: ZOOM_ANIMATION_MS,
        }}
        panning={{
          // A held Space outranks the drawing lockout: that is the whole point
          // of the override. Vertex dragging still wins over both, or the page
          // would slide while a handle is being moved.
          disabled:
            (isDrawing && !spaceHeld) ||
            isVertexInteracting ||
            vertexInteractingRef.current,
          velocityDisabled: false,
          wheelPanning: false,
          allowLeftClickPan: true,
          // Middle-drag is the mouse convention; right-drag is what an
          // eScriptorium user reaches for. Both leave the left button to the
          // tool, so neither can be mistaken for drawing.
          allowMiddleClickPan: true,
          allowRightClickPan: true,
          // Only the vertex handles own their drags. A page under two hundred
          // segments is nearly all segment, so excluding them too would leave
          // almost nowhere to grab; the click that ends a pan is swallowed
          // above instead, so releasing over a segment does not also select it.
          excluded: ["pe-vertex-handle"],
        }}
        zoomAnimation={{
          disabled: false,
          size: ZOOM_BUTTON_STEP,
          animationTime: ZOOM_ANIMATION_MS,
        }}
        velocityAnimation={{
          disabled: false,
          sensitivity: 1,
          animationTime: 350,
        }}
        alignmentAnimation={{
          disabled: false,
          sizeX: 0,
          sizeY: 0,
          animationTime: ZOOM_ANIMATION_MS,
        }}
        // A click after a wheel zoom must not fling the page: see zoomPanVelocity.
        onPanningStart={(ref, event) => {
          resetPanVelocityTracking(ref);
          panStartRef.current =
            "clientX" in event
              ? { x: event.clientX, y: event.clientY }
              : event.touches.length > 0
                ? { x: event.touches[0].clientX, y: event.touches[0].clientY }
                : null;
        }}
        onPanning={(_ref, event) => {
          const start = panStartRef.current;
          if (!start || panMovedRef.current) return;
          const point =
            "clientX" in event
              ? event
              : event.touches.length > 0
                ? event.touches[0]
                : null;
          if (!point) return;
          if (
            Math.hypot(point.clientX - start.x, point.clientY - start.y) >
            PAN_CLICK_THRESHOLD_PX
          ) {
            panMovedRef.current = true;
          }
        }}
        onTransformed={(ref, state) => {
          setZoomLevel(ref.state.scale);
          markTransforming();
          onTransformed?.({
            scale: state.scale,
            positionX: state.positionX,
            positionY: state.positionY,
          });
        }}
      >
        {({ resetTransform }) => (
          <>
            <PageEditorCanvasIsland
              tool={
                drawingRectangle
                  ? "rectangle"
                  : drawingPolygon
                    ? "polygon"
                    : "none"
              }
              onSelectTool={onSelectTool}
              onPickDrawMode={onPickDrawMode}
              canDelete={canDelete}
              onDeleteSelected={onDeleteSelected}
              zoomPercent={Math.round(zoomLevel * 100)}
              onZoomIn={() => zoomAnimated("in")}
              onZoomOut={() => zoomAnimated("out")}
              onFitToView={fitToView}
              onResetZoom={() => resetTransform()}
              panOverride={spaceHeld}
            />
            <TransformComponent
              wrapperStyle={{
                width: "100%",
                height: "100%",
                touchAction: "none",
              }}
              contentStyle={{ width: canvasWidth, height: canvasHeight }}
            >
              <CanvasSurface
                key={imageUrl}
                imageUrl={imageUrl}
                imageAlt={imageAlt}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                layout={layout}
                lines={lines}
                selectedSegmentId={selectedSegmentId}
                pairedSegmentIds={pairedSegmentIds}
                hoveredSegmentId={hoveredSegmentId}
                onHoverSegment={onHoverSegment}
                drawingRectangle={drawingRectangle}
                drawingPolygon={drawingPolygon}
                draftStart={draftStart}
                draftEnd={draftEnd}
                draftPolygon={draftPolygon}
                draftPolygonCursor={draftPolygonCursor}
                onDraftPolygonCursor={setDraftPolygonCursor}
                settings={settings}
                zoomLevel={zoomLevel}
                onDraftStart={(point) => {
                  setDraftEnd(point);
                  onDraftStart(point);
                }}
                onDraftMove={setDraftEnd}
                onRectangleDrawn={(point) => {
                  onRectangleDrawn(point);
                  setDraftEnd(null);
                }}
                onPolygonPoint={onPolygonPoint}
                onPolygonComplete={onPolygonComplete}
                onSelectLine={onSelectLine}
                onSelectSegment={onSelectSegment}
                segmentVertexEditEnabled={segmentVertexEditEnabled}
                panOverride={spaceHeld}
                suppressBaselineSegmentId={vertexEdit?.segmentId ?? null}
                vertexEditPoints={vertexEditPoints}
                draggedVertexIndex={vertexEdit?.draggedIndex ?? null}
                pendingVertexIndex={vertexEdit?.pendingVertexIndex ?? null}
                selectedVertexIndex={selectedVertexIndex}
                onSelectVertex={onSelectedVertexChange}
                onVertexPointerDown={(vertexIndex, event) => {
                  if (!selectedSegmentId) return;
                  const basePoints =
                    vertexEdit?.segmentId === selectedSegmentId
                      ? vertexEdit.points
                      : selectedSegment?.points;
                  if (!basePoints || basePoints.length < 3) return;
                  vertexInteractingRef.current = true;
                  event.currentTarget.setPointerCapture?.(event.pointerId);
                  onSelectedVertexChange(vertexIndex);
                  setVertexEdit({
                    segmentId: selectedSegmentId,
                    points: [...basePoints],
                    draggedIndex: null,
                    pendingVertexIndex: vertexIndex,
                  });
                }}
                onVertexPointerMove={(point) => {
                  setVertexEdit((current) => {
                    if (!current) return current;
                    if (
                      current.pendingVertexIndex !== null &&
                      current.draggedIndex === null
                    ) {
                      const anchor = current.points[current.pendingVertexIndex];
                      // The threshold is in screen pixels: in image units it
                      // would grow into a dead zone as the page is zoomed in.
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
                  });
                }}
                onInsertVertexOnEdge={(nextPoints) => {
                  if (!selectedSegmentId) return;
                  commitPendingVertexEdit(
                    {
                      segmentId: selectedSegmentId,
                      points: nextPoints,
                      draggedIndex: null,
                      pendingVertexIndex: null,
                    },
                    { selectedVertex: null },
                  );
                }}
                onRemoveVertex={(vertexIndex) => {
                  if (!selectedSegmentId || !vertexEditPoints) return;
                  const nextPoints = removePolygonVertex(
                    vertexEditPoints,
                    vertexIndex,
                  );
                  if (!nextPoints) return;
                  commitPendingVertexEdit(
                    {
                      segmentId: selectedSegmentId,
                      points: nextPoints,
                      draggedIndex: null,
                      pendingVertexIndex: null,
                    },
                    { selectedVertex: null },
                  );
                }}
              />
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
