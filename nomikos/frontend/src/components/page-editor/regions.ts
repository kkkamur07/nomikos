import type { LayoutBlockResponse } from "../../api/client";

/**
 * Regions (ADR 0011): the blocks of a Page, typically one per folio of a
 * spread. The server numbers Segments region by region, so the editor only
 * has to show where one region's Segments end and the next one's begin.
 */
export type Region = LayoutBlockResponse;

/** The Page's regions in the order the server numbers their Segments. */
export function regionsInOrder<T extends Region>(regions: T[]): T[] {
  // Stable: the layout lists blocks by (order, created_at) already.
  return [...regions].sort(
    (left, right) => (left.order ?? 0) - (right.order ?? 0),
  );
}

/** What a region is called: its label, or "Region n" by position. */
export function regionName(region: Region, regions: Region[]): string {
  const label = region.label?.trim();
  if (label) return label;
  const index = regionsInOrder(regions).findIndex(
    (other) => other.id === region.id,
  );
  return `Region ${index + 1}`;
}

/**
 * For each Segment that opens a group, the heading shown above it: a
 * region's name before its first Segment, and "Unassigned" before the first
 * Segment in no region. Nothing when the Page has no regions.
 */
export function regionSeparators(
  orderedSegments: { id: string; block_id?: string | null }[],
  regions: Region[],
): Map<string, string> {
  const separators = new Map<string, string>();
  if (regions.length === 0) return separators;
  const known = new Map(regions.map((region) => [region.id, region]));
  let previous: string | null | undefined;
  for (const segment of orderedSegments) {
    const region = segment.block_id ? known.get(segment.block_id) : undefined;
    const group = region?.id ?? null;
    if (group !== previous) {
      separators.set(
        segment.id,
        region ? regionName(region, regions) : "Unassigned",
      );
      previous = group;
    }
  }
  return separators;
}
