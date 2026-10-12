import type { LineResponse } from "../../api/client";

/**
 * What numbering needs from a Segment: which one it is, and the `order` the
 * server stored for it. Anything Segment-shaped can be numbered.
 */
export type NumberableSegment = Pick<LineResponse, "id" | "order">;

/**
 * The Segments of a Page in Segment number order.
 *
 * The server keeps `order` dense (ADR 0008), but a Page written before that
 * can still carry gaps and duplicates until it is renumbered, and the local
 * copy is briefly ahead of the server after an edit. Ties break on id so every
 * caller walks the same sequence whatever order the API listed the Segments in.
 */
export function segmentsInNumberOrder<T extends NumberableSegment>(
  segments: T[],
): T[] {
  return [...segments].sort((left, right) => {
    if (left.order !== right.order) return left.order - right.order;
    return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
  });
}

/**
 * Segment number by Segment id: the 1-based position of each Segment on its
 * Page, dense even when the stored orders are not.
 */
export function segmentNumbersById(
  segments: NumberableSegment[],
): Map<string, number> {
  return new Map(
    segmentsInNumberOrder(segments).map((segment, index) => [
      segment.id,
      index + 1,
    ]),
  );
}

/** The Segment number of one Segment, or null if it is not on this Page. */
export function segmentNumberFor(
  segments: NumberableSegment[],
  segmentId: string | null,
): number | null {
  if (segmentId === null) return null;
  return segmentNumbersById(segments).get(segmentId) ?? null;
}

/**
 * `segments` with each Segment's `order` taken from the server's copy of it.
 *
 * The server owns reading order (ADR 0008): creating or deleting one Segment
 * renumbers every Segment after it, and only the one row the request touched
 * comes back. Only `order` is taken, so an edit still in flight on another
 * Segment is not overwritten with the server's older geometry, and a Segment
 * the server does not list is left as it is.
 */
export function withServerOrders<T extends NumberableSegment>(
  segments: T[],
  serverSegments: NumberableSegment[],
): T[] {
  const serverOrders = new Map(
    serverSegments.map((segment) => [segment.id, segment.order]),
  );
  return segments.map((segment) => {
    const order = serverOrders.get(segment.id);
    return order === undefined || order === segment.order
      ? segment
      : { ...segment, order };
  });
}
