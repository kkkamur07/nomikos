import type { LineResponse } from "../../api/client";

/**
 * What numbering needs from a Segment: which one it is, and the `order` the
 * server stored for it. Anything Segment-shaped can be numbered.
 */
type NumberableSegment = Pick<LineResponse, "id" | "order">;

/**
 * The Segments of a Page in Segment number order.
 *
 * `order` alone is not a total order: the backend does not renumber on delete,
 * so stored orders carry gaps, and older Pages can hold duplicates. Ties break
 * on id so every caller walks the same sequence whatever order the API listed
 * the Segments in.
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
 * The `order` to persist for a newly drawn Segment: after every Segment already
 * on the Page, and never colliding with one, however gapped the stored orders
 * are. The Segment count cannot answer this: deleting a Segment leaves a hole,
 * not a shorter tail.
 */
export function nextSegmentOrder(segments: NumberableSegment[]): number {
  return (
    segments.reduce(
      (highest, segment) => Math.max(highest, segment.order),
      -1,
    ) + 1
  );
}
