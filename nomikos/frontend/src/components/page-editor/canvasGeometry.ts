import type { LayoutPoint, LinePoint } from "../../api/client";

type GeometryInput =
  | LinePoint[]
  | LayoutPoint[]
  | number[][]
  | {
      points?: number[][];
      type?: string;
      coordinates?: number[][];
    }
  | null
  | undefined;

type GeometryObject = {
  points?: LinePoint[];
  type?: string;
  coordinates?: LinePoint[];
};

function asPointPair(value: LinePoint | number[]): LinePoint | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const [x, y] = value;
  if (typeof x !== "number" || typeof y !== "number") return null;
  return [x, y];
}

export function normalizeGeometryPoints(input: GeometryInput): LinePoint[] {
  if (!input) return [];
  if (Array.isArray(input)) {
    return input
      .map(asPointPair)
      .filter((point): point is LinePoint => point !== null);
  }
  if (typeof input === "object") {
    if (Array.isArray(input.points)) {
      return normalizeGeometryPoints(input.points);
    }
    if (Array.isArray(input.coordinates)) {
      return normalizeGeometryPoints(input.coordinates);
    }
  }
  return [];
}

export function points(input: GeometryInput): string {
  return normalizeGeometryPoints(input)
    .map(([x, y]) => `${x},${y}`)
    .join(" ");
}

export function withGeometryPoints(
  geometry: GeometryInput,
  nextPoints: LinePoint[],
): GeometryObject {
  if (geometry && typeof geometry === "object" && !Array.isArray(geometry)) {
    const type = geometry.type;
    if ("coordinates" in geometry) {
      return { ...(type ? { type } : {}), coordinates: nextPoints };
    }
    return { ...(type ? { type } : {}), points: nextPoints };
  }
  return { points: nextPoints };
}

export function offsetGeometry(
  geometry: GeometryInput,
  deltaY: number,
): GeometryObject {
  const shifted = normalizeGeometryPoints(geometry).map(
    ([x, y]) => [x, y + deltaY] as LinePoint,
  );
  return withGeometryPoints(geometry, shifted);
}

export function rectanglePoints(start: LinePoint, end: LinePoint): LinePoint[] {
  const [startX, startY] = start;
  const [endX, endY] = end;
  return [
    [Math.min(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.min(startY, endY)],
    [Math.max(startX, endX), Math.max(startY, endY)],
    [Math.min(startX, endX), Math.max(startY, endY)],
  ];
}

/** Screen-consistent SVG stroke width in image coordinates. */
export function canvasStrokeWidth(
  base: number,
  zoomLevel: number,
  overlayStrokeWidth: number,
  canvasMaxDimension: number,
): number {
  const safeZoom = Math.max(zoomLevel, 0.05);
  const imageNorm = Math.max(canvasMaxDimension / 1400, 0.35);
  return Math.max((base * overlayStrokeWidth * imageNorm) / safeZoom, 0.35);
}

/** Screen-consistent vertex / pointer handle radius in image coordinates. */
export function canvasHandleRadius(
  base: number,
  zoomLevel: number,
  handleSize: number,
  overlayStrokeWidth: number,
  canvasMaxDimension: number,
): number {
  return canvasStrokeWidth(
    base * handleSize,
    zoomLevel,
    overlayStrokeWidth,
    canvasMaxDimension,
  );
}

/** Minimum spacing between consecutive polygon vertices (image px). */
const MIN_VERTEX_SPACING = 3;

function pointDistance(a: LinePoint, b: LinePoint): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function cleanPolygonPoints(
  points: LinePoint[],
  minDistance = MIN_VERTEX_SPACING,
  minVertices = 3,
): LinePoint[] {
  if (points.length < 2) return points;

  const working = points.map(([x, y]) => [x, y] as LinePoint);
  if (
    working.length >= 2 &&
    pointDistance(working[0], working[working.length - 1]) <= minDistance
  ) {
    working.pop();
  }
  if (working.length < 2) return points;

  const cleaned: LinePoint[] = [working[0]];
  for (let index = 1; index < working.length; index += 1) {
    if (
      pointDistance(working[index], cleaned[cleaned.length - 1]) > minDistance
    ) {
      cleaned.push(working[index]);
    }
  }
  if (
    cleaned.length >= 2 &&
    pointDistance(cleaned[0], cleaned[cleaned.length - 1]) <= minDistance
  ) {
    cleaned.pop();
  }

  return cleaned.length >= minVertices ? cleaned : points;
}

export function removePolygonVertex(
  polygon: LinePoint[],
  index: number,
): LinePoint[] | null {
  if (index < 0 || index >= polygon.length || polygon.length <= 3) return null;
  return polygon.filter((_, vertexIndex) => vertexIndex !== index);
}

function distanceSquaredPointToSegment(
  point: LinePoint,
  start: LinePoint,
  end: LinePoint,
): number {
  const [px, py] = point;
  const [ax, ay] = start;
  const [bx, by] = end;
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    const deltaX = px - ax;
    const deltaY = py - ay;
    return deltaX * deltaX + deltaY * deltaY;
  }
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  t = Math.max(0, Math.min(1, t));
  const closestX = ax + t * dx;
  const closestY = ay + t * dy;
  const deltaX = px - closestX;
  const deltaY = py - closestY;
  return deltaX * deltaX + deltaY * deltaY;
}

function closestPointOnSegment(
  point: LinePoint,
  start: LinePoint,
  end: LinePoint,
): LinePoint {
  const [px, py] = point;
  const [ax, ay] = start;
  const [bx, by] = end;
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    return [Math.round(ax), Math.round(ay)];
  }
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  t = Math.max(0, Math.min(1, t));
  return [Math.round(ax + t * dx), Math.round(ay + t * dy)];
}

function isNearExistingVertex(
  polygon: LinePoint[],
  click: LinePoint,
  minDistance: number,
): boolean {
  const minDistanceSquared = minDistance * minDistance;
  return polygon.some(([x, y]) => {
    const deltaX = click[0] - x;
    const deltaY = click[1] - y;
    return deltaX * deltaX + deltaY * deltaY <= minDistanceSquared;
  });
}

export function findPolygonEdgeInsert(
  polygon: LinePoint[],
  click: LinePoint,
  maxDistance: number,
): { insertIndex: number; point: LinePoint } | null {
  if (polygon.length < 3) return null;
  if (isNearExistingVertex(polygon, click, maxDistance * 0.9)) return null;

  const maxDistanceSquared = maxDistance * maxDistance;
  let best: {
    insertIndex: number;
    point: LinePoint;
    distanceSquared: number;
  } | null = null;

  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    const distanceSquared = distanceSquaredPointToSegment(click, start, end);
    if (distanceSquared > maxDistanceSquared) continue;

    const candidate = {
      insertIndex: index + 1,
      point: closestPointOnSegment(click, start, end),
      distanceSquared,
    };
    if (!best || candidate.distanceSquared < best.distanceSquared) {
      best = candidate;
    }
  }

  return best ? { insertIndex: best.insertIndex, point: best.point } : null;
}

export function insertPolygonVertexAtClick(
  polygon: LinePoint[],
  click: LinePoint,
  maxDistance: number,
): LinePoint[] | null {
  const hit = findPolygonEdgeInsert(polygon, click, maxDistance);
  if (!hit) return null;
  const next = [...polygon];
  next.splice(hit.insertIndex, 0, hit.point);
  return next;
}
