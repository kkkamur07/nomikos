import type { GeometryValue, LinePoint } from "../../api/client";

import { normalizeGeometryPoints } from "./canvasGeometry";

export type ReadingDirection = "ltr" | "rtl";

export type OrderableSegment = {
  id: string;
  baseline?: GeometryValue | null;
  mask?: GeometryValue | null;
  points?: LinePoint[] | null;
};

export type ReadingOrderOptions = {
  direction?: ReadingDirection;
  columnGapRatio?: number;
};

const DEFAULT_DIRECTION: ReadingDirection = "ltr";
const DEFAULT_COLUMN_GAP_RATIO = 0.6;

function compareIds(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function boundingBox(points: LinePoint[]): {
  minX: number;
  maxX: number;
  minY: number;
} {
  let minX = points[0][0];
  let maxX = points[0][0];
  let minY = points[0][1];
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
  }
  return { minX, maxX, minY };
}

/**
 * The point a reader starts from on this Segment: the left end of the
 * baseline, or the right end for RTL scripts. Segments whose baseline is
 * too short to have ends (fewer than 2 points) fall back to the corner of
 * their polygon bounding box the reading starts from. Null when the
 * Segment has no geometry at all.
 */
export function segmentOrigin(
  segment: OrderableSegment,
  direction: ReadingDirection = DEFAULT_DIRECTION,
): LinePoint | null {
  const baseline = normalizeGeometryPoints(segment.baseline);
  if (baseline.length >= 2) {
    let best = baseline[0];
    for (const point of baseline) {
      if (direction === "rtl") {
        if (
          point[0] > best[0] ||
          (point[0] === best[0] && point[1] < best[1])
        ) {
          best = point;
        }
      } else if (
        point[0] < best[0] ||
        (point[0] === best[0] && point[1] < best[1])
      ) {
        best = point;
      }
    }
    return [best[0], best[1]];
  }
  const mask = normalizeGeometryPoints(segment.mask);
  const fallback =
    mask.length > 0 ? mask : normalizeGeometryPoints(segment.points ?? null);
  if (fallback.length === 0) return null;
  const box = boundingBox(fallback);
  return direction === "rtl" ? [box.maxX, box.minY] : [box.minX, box.minY];
}

function horizontalExtent(segment: OrderableSegment): {
  left: number;
  right: number;
} | null {
  const mask = normalizeGeometryPoints(segment.mask);
  const outline =
    mask.length > 0 ? mask : normalizeGeometryPoints(segment.points ?? null);
  const polygon =
    outline.length > 0 ? outline : normalizeGeometryPoints(segment.baseline);
  if (polygon.length === 0) return null;
  const box = boundingBox(polygon);
  return { left: box.minX, right: box.maxX };
}

type PlacedSegment<T> = {
  segment: T;
  origin: LinePoint;
  left: number;
  right: number;
};

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * The Segments in the order a reader would read them: columns run left to
 * right (mirrored for RTL scripts) and Segments within a column run top to
 * bottom from their reading origin. Segments without geometry go last, in
 * id order. The input array is left untouched.
 */
export function readingOrder<T extends OrderableSegment>(
  segments: T[],
  options: ReadingOrderOptions = {},
): T[] {
  const direction = options.direction ?? DEFAULT_DIRECTION;
  const columnGapRatio = options.columnGapRatio ?? DEFAULT_COLUMN_GAP_RATIO;

  const placed: PlacedSegment<T>[] = [];
  const withoutGeometry: T[] = [];
  for (const segment of segments) {
    const origin = segmentOrigin(segment, direction);
    const extent = horizontalExtent(segment);
    if (origin === null || extent === null) {
      withoutGeometry.push(segment);
      continue;
    }
    placed.push({
      segment,
      origin,
      left: extent.left,
      right: extent.right,
    });
  }
  withoutGeometry.sort((left, right) => compareIds(left.id, right.id));
  if (placed.length === 0) return withoutGeometry;

  const medianWidth = median(placed.map((entry) => entry.right - entry.left));
  const byOriginX = [...placed].sort(
    (left, right) =>
      left.origin[0] - right.origin[0] ||
      left.origin[1] - right.origin[1] ||
      compareIds(left.segment.id, right.segment.id),
  );

  const columns: {
    left: number;
    right: number;
    members: PlacedSegment<T>[];
  }[] = [];
  for (const entry of byOriginX) {
    const current = columns[columns.length - 1];
    if (!current || entry.left > current.right - columnGapRatio * medianWidth) {
      columns.push({ left: entry.left, right: entry.right, members: [entry] });
    } else {
      current.members.push(entry);
      current.left = Math.min(current.left, entry.left);
      current.right = Math.max(current.right, entry.right);
    }
  }
  columns.sort((left, right) =>
    direction === "rtl" ? right.left - left.left : left.left - right.left,
  );

  const ordered: T[] = [];
  for (const column of columns) {
    column.members.sort(
      (left, right) =>
        left.origin[1] - right.origin[1] ||
        (direction === "rtl"
          ? right.origin[0] - left.origin[0]
          : left.origin[0] - right.origin[0]) ||
        compareIds(left.segment.id, right.segment.id),
    );
    for (const entry of column.members) ordered.push(entry.segment);
  }
  return [...ordered, ...withoutGeometry];
}

/**
 * The Segment `step` away from `currentId` in reading order, or null past
 * either end (nothing wraps). A null `currentId` starts at the first
 * Segment for step 1 or the last for step -1. An unknown id returns null.
 */
export function nextInReadingOrder<T extends OrderableSegment>(
  segments: T[],
  currentId: string | null,
  step: -1 | 1,
  options: ReadingOrderOptions = {},
): T | null {
  const ordered = readingOrder(segments, options);
  if (ordered.length === 0) return null;
  if (currentId === null) {
    return step === 1 ? ordered[0] : ordered[ordered.length - 1];
  }
  const index = ordered.findIndex((segment) => segment.id === currentId);
  if (index === -1) return null;
  const next = index + step;
  if (next < 0 || next >= ordered.length) return null;
  return ordered[next];
}
