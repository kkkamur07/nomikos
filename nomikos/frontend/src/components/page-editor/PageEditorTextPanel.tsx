import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react";
import {
  TransformComponent,
  TransformWrapper,
  type ReactZoomPanPinchRef,
} from "react-zoom-pan-pinch";
import type { LineResponse } from "../../api/client";
import {
  normalizeGeometryPoints,
  points as pointsAttribute,
} from "./canvasGeometry";
import { segmentsInNumberOrder } from "./segmentNumbering";
import {
  baselinePath,
  baselinePoints,
  displayText,
  lineFontSize,
  polygonBounds,
  polylineLength,
} from "./textPanelGeometry";
import { useSmoothWheelZoom } from "./useSmoothWheelZoom";

const MIN_SCALE = 0.15;
const MAX_SCALE = 8;
/** Baseline teal for lines without text, as on the canvas. */
const EMPTY_BASELINE_STROKE = "#0d9488";
const MIN_EDITOR_HEIGHT = 24;

export type TextPanelTransformState = {
  scale: number;
  positionX: number;
  positionY: number;
};

type PageEditorTextPanelProps = {
  lines: LineResponse[];
  imageWidth: number;
  imageHeight: number;
  selectedSegmentId: string | null;
  hoveredSegmentId: string | null;
  editingSegmentId: string | null;
  textDirection: "ltr" | "rtl";
  preferredLayerId?: string | null;
  fontScale?: number;
  wheelZoomSpeed?: number;
  onSelectSegment: (lineId: string) => void;
  onHoverSegment: (lineId: string | null) => void;
  onRequestEdit: (lineId: string | null) => void;
  onCommitText: (lineId: string, text: string) => Promise<void> | void;
  onCommitted?: (lineId: string) => void;
  viewportRef?: Ref<ReactZoomPanPinchRef>;
  onTransformed?: (state: TextPanelTransformState) => void;
};

function mergeViewportRef(
  viewportRef: Ref<ReactZoomPanPinchRef> | undefined,
  innerRef: ReactZoomPanPinchRef | null,
): void {
  if (!viewportRef) return;
  if (typeof viewportRef === "function") {
    viewportRef(innerRef);
  } else {
    viewportRef.current = innerRef;
  }
}

function TextLineEditor({
  lineId,
  initialValue,
  fontSize,
  dir,
  onCommitText,
  onRequestEdit,
  onCommitted,
}: {
  lineId: string;
  initialValue: string;
  fontSize: number;
  dir: "ltr" | "rtl";
  onCommitText: (lineId: string, text: string) => Promise<void> | void;
  onRequestEdit: (lineId: string | null) => void;
  onCommitted?: (lineId: string) => void;
}) {
  const [value, setValue] = useState(initialValue);
  const [saving, setSaving] = useState(false);
  const [hasError, setHasError] = useState(false);
  const settledRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.focus();
    element.setSelectionRange(element.value.length, element.value.length);
  }, []);

  const commit = async () => {
    if (saving || settledRef.current) return;
    settledRef.current = true;
    setSaving(true);
    try {
      await onCommitText(lineId, value.trim());
      onRequestEdit(null);
      onCommitted?.(lineId);
    } catch (error) {
      console.error(error);
      settledRef.current = false;
      setHasError(true);
      setSaving(false);
    }
  };

  const cancel = () => {
    settledRef.current = true;
    onRequestEdit(null);
  };

  return (
    <textarea
      className={`pe-text-line-editor${saving ? " is-saving" : ""}${hasError ? " has-error" : ""}`}
      dir={dir}
      value={value}
      style={{ fontSize }}
      ref={textareaRef}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          void commit();
        } else if (event.key === "Escape") {
          event.preventDefault();
          cancel();
        }
      }}
      onBlur={() => {
        if (settledRef.current) return;
        if (value !== initialValue) void commit();
      }}
    />
  );
}

export function PageEditorTextPanel({
  lines,
  imageWidth,
  imageHeight,
  selectedSegmentId,
  hoveredSegmentId,
  editingSegmentId,
  textDirection,
  preferredLayerId,
  fontScale = 1,
  wheelZoomSpeed = 1,
  onSelectSegment,
  onHoverSegment,
  onRequestEdit,
  onCommitText,
  onCommitted,
  viewportRef,
  onTransformed,
}: PageEditorTextPanelProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const transformRef = useRef<ReactZoomPanPinchRef>(null);
  useSmoothWheelZoom(hostRef, transformRef, {
    minScale: MIN_SCALE,
    maxScale: MAX_SCALE,
    speed: wheelZoomSpeed,
  });
  useEffect(() => {
    mergeViewportRef(viewportRef, transformRef.current);
  }, [viewportRef]);

  const orderedLines = useMemo(() => segmentsInNumberOrder(lines), [lines]);
  const isRtl = textDirection === "rtl";

  return (
    <div className="pe-text-panel" ref={hostRef}>
      <TransformWrapper
        ref={transformRef}
        initialScale={1}
        minScale={MIN_SCALE}
        maxScale={MAX_SCALE}
        centerOnInit={false}
        limitToBounds={false}
        wheel={{ disabled: true }}
        onTransformed={(ref) => {
          onTransformed?.({
            scale: ref.state.scale,
            positionX: ref.state.positionX,
            positionY: ref.state.positionY,
          });
        }}
      >
        <TransformComponent
          wrapperStyle={{ width: "100%", height: "100%" }}
          contentStyle={{ width: imageWidth, height: imageHeight }}
        >
          <div
            className="pe-text-panel__surface"
            style={{ width: imageWidth, height: imageHeight }}
          >
            <svg
              viewBox={`0 0 ${imageWidth} ${imageHeight}`}
              width={imageWidth}
              height={imageHeight}
              role="group"
              aria-label="Page text panel"
            >
              {orderedLines.map((line) => {
                const baseline = baselinePoints(line);
                const display = displayText(line, preferredLayerId);
                const isEmpty = display.text.trim() === "";
                const fontSize = lineFontSize(line, fontScale);
                const maskPoints = normalizeGeometryPoints(line.mask);
                const polygon =
                  maskPoints.length > 0
                    ? maskPoints
                    : normalizeGeometryPoints(line.points);
                const className =
                  `pe-text-line${line.id === selectedSegmentId ? " is-selected" : ""}` +
                  `${line.id === hoveredSegmentId ? " is-hovered" : ""}` +
                  `${isEmpty ? " is-empty" : display.source === "model" ? " is-model" : " is-ground-truth"}`;
                const editing = editingSegmentId === line.id;
                const bounds = editing ? polygonBounds(polygon) : null;
                return (
                  <g
                    key={line.id}
                    data-line-id={line.id}
                    className={className}
                    role="button"
                    tabIndex={0}
                    onMouseEnter={() => onHoverSegment(line.id)}
                    onMouseLeave={() => onHoverSegment(null)}
                    onClick={() => onSelectSegment(line.id)}
                    onDoubleClick={() => onRequestEdit(line.id)}
                    onKeyDown={(event: KeyboardEvent<SVGGElement>) => {
                      if (event.key !== "Enter") return;
                      const target = event.target as Element | null;
                      if (target && target.closest("textarea")) return;
                      event.preventDefault();
                      onRequestEdit(line.id);
                    }}
                  >
                    {polygon.length > 0 && (
                      <polygon
                        className="pe-text-line-mask"
                        points={pointsAttribute(polygon)}
                      />
                    )}
                    {baseline.length >= 2 && (
                      <path
                        id={`pe-text-baseline-${line.id}`}
                        className="pe-text-line-baseline"
                        d={baselinePath(baseline)}
                        stroke={isEmpty ? EMPTY_BASELINE_STROKE : "transparent"}
                      />
                    )}
                    {!isEmpty && baseline.length >= 2 && (
                      <text
                        className={
                          display.source === "model"
                            ? "is-model"
                            : "is-ground-truth"
                        }
                        fontSize={fontSize}
                        lengthAdjust="spacingAndGlyphs"
                        textLength={polylineLength(baseline)}
                        direction={isRtl ? "rtl" : "ltr"}
                        textAnchor={isRtl ? "end" : "start"}
                      >
                        <textPath
                          href={`#pe-text-baseline-${line.id}`}
                          startOffset={isRtl ? "100%" : undefined}
                        >
                          {display.text}
                        </textPath>
                      </text>
                    )}
                    {editing && bounds && (
                      <foreignObject
                        x={bounds.x}
                        y={bounds.y}
                        width={Math.max(bounds.width, 10)}
                        height={Math.max(bounds.height, MIN_EDITOR_HEIGHT)}
                      >
                        <TextLineEditor
                          lineId={line.id}
                          initialValue={display.text}
                          fontSize={fontSize}
                          dir={textDirection}
                          onCommitText={onCommitText}
                          onRequestEdit={onRequestEdit}
                          onCommitted={onCommitted}
                        />
                      </foreignObject>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </TransformComponent>
      </TransformWrapper>
    </div>
  );
}
