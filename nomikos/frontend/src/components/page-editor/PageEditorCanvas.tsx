import { useEffect, useRef, useState, type Ref, type RefObject } from "react";
import {
  TransformComponent,
  TransformWrapper,
  type ReactZoomPanPinchContentRef,
  type ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch";
import type { LinePoint } from "../../api/client";
import { resetPanVelocityTracking } from "../../utils/zoomPanVelocity";
import { useZoomLayerHint } from "../../utils/zoomLayerHint";
import { CanvasSurface, type CanvasSurfaceProps } from "./CanvasSurface";
import { normalizeGeometryPoints } from "./canvasGeometry";
import { fitTransform } from "./canvasZoom";
import {
  PageEditorCanvasIsland,
  type DrawTool,
} from "./PageEditorCanvasIsland";
import { PageEditorMergeBar } from "./PageEditorMergeBar";
import { useSmoothWheelZoom } from "./useSmoothWheelZoom";
import { useSpacePan } from "./useSpacePan";
import { useStableCallback } from "./useStableCallback";
import { useVertexEdit } from "./useVertexEdit";

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
  | "draftPolygonCursor"
  | "onDraftPolygonCursor"
  | "onSelectVertex"
  | "panOverride"
> & {
  /** Tool controls for the on-canvas island. */
  onSelectTool: () => void;
  onPickDrawMode: (mode: DrawTool) => void;
  /** The Region tool is armed: the rectangle drag draws a region, not a Segment. */
  drawingRegion?: boolean;
  canDelete: boolean;
  onDeleteSelected: () => void;
  /** Merge bar, shown only while two or more segments are picked. */
  mergeBusy: boolean;
  mergeError: string | null;
  onMergeSelected: () => void;
  onClearMergeSelection: () => void;
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

export function PageEditorCanvas(props: PageEditorCanvasProps) {
  const {
    imageUrl,
    imageWidth,
    imageHeight,
    lines,
    selectedSegmentId,
    drawingRectangle: drawingSegmentRectangle,
    drawingRegion = false,
    drawingPolygon,
    settings,
    segmentVertexEditEnabled,
    viewportRef,
    onTransformed,
    focusRequest,
  } = props;
  // The Region tool draws with the same drag as a rectangle Segment; only
  // what the page does with the finished rectangle differs.
  const drawingRectangle = drawingSegmentRectangle || drawingRegion;
  const [zoomLevel, setZoomLevel] = useState(1);
  const spaceHeld = useSpacePan();
  const { transforming, markTransforming } = useZoomLayerHint();
  const [draftEnd, setDraftEnd] = useState<LinePoint | null>(null);
  const [draftPolygonCursor, setDraftPolygonCursor] =
    useState<LinePoint | null>(null);
  const transformRef = useRef<ReactZoomPanPinchRef>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const panStartRef = useRef<{ x: number; y: number } | null>(null);
  const panMovedRef = useRef(false);
  const canvasWidth = imageWidth;
  const canvasHeight = imageHeight;
  const isDrawing = drawingRectangle || drawingPolygon;
  const vertex = useVertexEdit({
    lines,
    selectedSegmentId,
    segmentVertexEditEnabled,
    zoomLevel,
    commitSignal: props.commitSignal,
    onSelectedVertexChange: props.onSelectedVertexChange,
    onSegmentPointsChange: props.onSegmentPointsChange,
  });

  // The memoised surface only skips a render when every callback it gets
  // keeps its identity, whatever the parent hands in.
  const onDraftStart = useStableCallback((point: LinePoint) => {
    setDraftEnd(point);
    props.onDraftStart(point);
  });
  const onRectangleDrawn = useStableCallback((point: LinePoint) => {
    props.onRectangleDrawn(point);
    setDraftEnd(null);
  });
  const onPolygonPoint = useStableCallback(props.onPolygonPoint);
  const onPolygonComplete = useStableCallback(props.onPolygonComplete);
  const onSelectLine = useStableCallback(props.onSelectLine);
  const onSelectSegment = useStableCallback(props.onSelectSegment);
  const onHoverSegment = useStableCallback((lineId: string | null) =>
    props.onHoverSegment?.(lineId),
  );
  const onSelectRegion = useStableCallback((regionId: string) =>
    props.onSelectRegion?.(regionId),
  );

  useEffect(() => {
    if (!drawingRectangle) setDraftEnd(null);
  }, [drawingRectangle]);

  useEffect(() => {
    if (!drawingPolygon) setDraftPolygonCursor(null);
  }, [drawingPolygon]);

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
   * button called "Fit page to view" promises.
   */
  const fitToView = () => {
    const ref = transformRef.current;
    const wrapper = ref?.instance.wrapperComponent;
    if (!ref || !wrapper) return;
    const fit = fitTransform(
      { width: wrapper.offsetWidth, height: wrapper.offsetHeight },
      { width: canvasWidth, height: canvasHeight },
      {
        padding: FIT_PADDING_PX,
        reservedBottom: ISLAND_RESERVED_PX,
        minScale: MIN_SCALE,
        maxScale: MAX_SCALE,
      },
    );
    if (fit) ref.setTransform(fit.x, fit.y, fit.scale, ZOOM_ANIMATION_MS);
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
            vertex.isVertexInteracting ||
            vertex.vertexInteractingRef.current,
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
                drawingRegion
                  ? "region"
                  : drawingSegmentRectangle
                    ? "rectangle"
                    : drawingPolygon
                      ? "polygon"
                      : "none"
              }
              onSelectTool={props.onSelectTool}
              onPickDrawMode={props.onPickDrawMode}
              canDelete={props.canDelete}
              onDeleteSelected={props.onDeleteSelected}
              zoomPercent={Math.round(zoomLevel * 100)}
              onZoomIn={() => zoomAnimated("in")}
              onZoomOut={() => zoomAnimated("out")}
              onFitToView={fitToView}
              onResetZoom={() => resetTransform()}
              panOverride={spaceHeld}
            />
            <PageEditorMergeBar
              count={props.selectedSegmentIds.length}
              busy={props.mergeBusy}
              error={props.mergeError}
              onMerge={props.onMergeSelected}
              onClear={props.onClearMergeSelection}
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
                imageAlt={props.imageAlt}
                imageWidth={imageWidth}
                imageHeight={imageHeight}
                layout={props.layout}
                lines={lines}
                selectedSegmentId={selectedSegmentId}
                selectedSegmentIds={props.selectedSegmentIds}
                pairedSegmentIds={props.pairedSegmentIds}
                hoveredSegmentId={props.hoveredSegmentId}
                onHoverSegment={onHoverSegment}
                drawingRectangle={drawingRectangle}
                drawingPolygon={drawingPolygon}
                draftStart={props.draftStart}
                draftEnd={draftEnd}
                draftPolygon={props.draftPolygon}
                draftPolygonCursor={draftPolygonCursor}
                onDraftPolygonCursor={setDraftPolygonCursor}
                settings={settings}
                zoomLevel={zoomLevel}
                onDraftStart={onDraftStart}
                onDraftMove={setDraftEnd}
                onRectangleDrawn={onRectangleDrawn}
                onPolygonPoint={onPolygonPoint}
                onPolygonComplete={onPolygonComplete}
                onSelectLine={onSelectLine}
                onSelectSegment={onSelectSegment}
                selectedRegionId={props.selectedRegionId}
                onSelectRegion={onSelectRegion}
                segmentVertexEditEnabled={segmentVertexEditEnabled}
                panOverride={spaceHeld}
                suppressBaselineSegmentId={vertex.editingSegmentId}
                vertexEditPoints={vertex.vertexEditPoints}
                draggedVertexIndex={vertex.draggedVertexIndex}
                pendingVertexIndex={vertex.pendingVertexIndex}
                selectedVertexIndex={props.selectedVertexIndex}
                onSelectVertex={vertex.selectVertex}
                onVertexPointerDown={vertex.onVertexPointerDown}
                onVertexPointerMove={vertex.onVertexPointerMove}
                onInsertVertexOnEdge={vertex.onInsertVertexOnEdge}
              />
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
