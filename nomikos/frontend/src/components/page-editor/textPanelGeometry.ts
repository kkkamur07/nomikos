import type { LinePoint, LineResponse } from "../../api/client";

import { normalizeGeometryPoints } from "./canvasGeometry";
import { approvedText, modelTranscriptionForLine } from "./hooks/utils";

export const FONT_SIZE_RATIO = 0.65;
export const MIN_FONT_SIZE = 8;
export const MAX_FONT_SIZE = 400;
export const DEFAULT_STRIP_HEIGHT = 30;

export type TextSource = "ground_truth" | "model" | "none";

/**
 * Usable baseline points for drawing a line's transcription.
 *
 * A stored baseline with at least two points is used only when it is not
 * degenerate: its polyline length must be at least half the polygon
 * bounding-box width (mask if present, else points), or greater than zero
 * when there is no polygon. Otherwise a fake baseline is derived from the
 * polygon: from the leftmost polygon point to the rightmost one, both
 * placed at the polygon's lower quarter. Empty when no geometry exists.
 */
export function baselinePoints(line: LineResponse): LinePoint[] {
  const baseline = normalizeGeometryPoints(line.baseline);
  const maskPoints = normalizeGeometryPoints(line.mask);
  const polygon =
    maskPoints.length > 0 ? maskPoints : normalizeGeometryPoints(line.points);
  if (baseline.length >= 2) {
    const length = polylineLength(baseline);
    if (polygon.length === 0) {
      if (length > 0) return baseline;
    } else {
      const bounds = polygonBounds(polygon);
      if (length > 0 && length >= bounds.width * 0.5) return baseline;
    }
  }
  if (polygon.length === 0) return [];
  const bounds = polygonBounds(polygon);
  const lowerQuarterY = bounds.y + bounds.height * 0.75;
  let leftmost = polygon[0];
  let rightmost = polygon[0];
  for (const point of polygon) {
    if (point[0] < leftmost[0]) leftmost = point;
    if (point[0] > rightmost[0]) rightmost = point;
  }
  return [
    [leftmost[0], lowerQuarterY],
    [rightmost[0], lowerQuarterY],
  ];
}

/** Baseline points as an SVG path (`M x y L x y ...`, integers). */
export function baselinePath(points: LinePoint[]): string {
  return points
    .map(([x, y], index) =>
      index === 0
        ? `M ${Math.round(x)} ${Math.round(y)}`
        : `L ${Math.round(x)} ${Math.round(y)}`,
    )
    .join(" ");
}

/** Length of a polyline in image px. */
export function polylineLength(points: LinePoint[]): number {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    total += Math.hypot(
      points[index][0] - points[index - 1][0],
      points[index][1] - points[index - 1][1],
    );
  }
  return total;
}

/** Absolute polygon area (shoelace) in square image px. */
export function polygonArea(points: LinePoint[]): number {
  let total = 0;
  for (let index = 0; index < points.length; index += 1) {
    const [x1, y1] = points[index];
    const [x2, y2] = points[(index + 1) % points.length];
    total += x1 * y2 - x2 * y1;
  }
  return Math.abs(total) / 2;
}

export function polygonBounds(points: LinePoint[]): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  if (points.length === 0) return { x: 0, y: 0, width: 0, height: 0 };
  let minX = points[0][0];
  let maxX = points[0][0];
  let minY = points[0][1];
  let maxY = points[0][1];
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * Font size for a line's transcription: the mask strip height (polygon area
 * over baseline length) scaled to about two thirds of a line height,
 * clamped, then multiplied by the user's font scale.
 */
export function lineFontSize(line: LineResponse, fontScale = 1): number {
  const maskPoints = normalizeGeometryPoints(line.mask);
  const polygon =
    maskPoints.length > 0 ? maskPoints : normalizeGeometryPoints(line.points);
  const baseline = baselinePoints(line);
  const length = polylineLength(baseline);
  const stripHeight =
    polygon.length > 0 && length > 0
      ? polygonArea(polygon) / length
      : DEFAULT_STRIP_HEIGHT;
  const raw = Math.round(FONT_SIZE_RATIO * stripHeight);
  const heightCapped =
    polygon.length > 0
      ? Math.min(raw, 0.9 * polygonBounds(polygon).height)
      : raw;
  const clamped = Math.min(
    MAX_FONT_SIZE,
    Math.max(MIN_FONT_SIZE, heightCapped),
  );
  return clamped * fontScale;
}

/**
 * Text to draw for a line: ground truth when present and non-blank, else
 * the model transcription for the preferred layer, else empty.
 */
export function displayText(
  line: LineResponse,
  preferredLayerId?: string | null,
): { text: string; source: TextSource } {
  const groundTruth = approvedText(line);
  if (groundTruth && groundTruth.trim() !== "") {
    return { text: groundTruth, source: "ground_truth" };
  }
  const model = modelTranscriptionForLine(line, preferredLayerId);
  if (model) {
    return { text: model.text ?? "", source: "model" };
  }
  return { text: "", source: "none" };
}
