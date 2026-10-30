/**
 * Picking several segments at once, which the canvas could not do before.
 *
 * The editor already has exactly one selected segment and a great deal hangs
 * off it: the transcription strip, pairing, per-segment OCR, delete, vertex
 * editing. None of that means anything for a group, so this does not replace
 * it. A plain click still selects one segment and clears the group; a
 * shift-click or cmd/ctrl-click builds the group instead, and the anchor
 * follows the last segment added so the strip keeps showing something real.
 *
 * A group of one is not a group. Collapsing it back to an empty list is what
 * makes "is a merge on offer?" a length test rather than a length test plus a
 * special case, and it means shift-clicking a segment twice leaves the editor
 * exactly as an ordinary click would have.
 */

/** The cap the server enforces; the canvas refuses earlier so it can say why. */
export const MAX_MERGE_SEGMENTS = 20;

export type SegmentSelection = {
  /** The one segment everything single-selection already keys off. */
  anchorId: string | null;
  /** Two or more segments picked for a group action, or empty. */
  groupIds: string[];
};

export const EMPTY_SELECTION: SegmentSelection = {
  anchorId: null,
  groupIds: [],
};

/** Shift, cmd or ctrl: whichever the platform's muscle memory reaches for. */
export function isAdditiveClick(event: {
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
}): boolean {
  return event.shiftKey || event.metaKey || event.ctrlKey;
}

export function selectionAfterClick(
  current: SegmentSelection,
  lineId: string,
  additive: boolean,
): SegmentSelection {
  if (!additive) return { anchorId: lineId, groupIds: [] };

  // The first additive click has to pull the already-selected segment in with
  // it, or shift-clicking a neighbour would silently start a group of one and
  // throw away the segment the user was looking at.
  const base =
    current.groupIds.length > 0
      ? current.groupIds
      : current.anchorId
        ? [current.anchorId]
        : [];
  const next = base.includes(lineId)
    ? base.filter((id) => id !== lineId)
    : [...base, lineId];

  if (next.length <= 1) {
    return { anchorId: next[0] ?? null, groupIds: [] };
  }
  return { anchorId: next[next.length - 1], groupIds: next };
}

/**
 * Drop group members that are no longer on the page.
 *
 * A merge, a delete or a re-segment replaces the line list, and a group still
 * naming rows that are gone would send the server ids it answers 404 to. What
 * is left of a group of one is not a group, so it comes back empty.
 */
export function prunedGroup(
  groupIds: string[],
  availableIds: Iterable<string>,
): string[] {
  const available = new Set(availableIds);
  const kept = groupIds.filter((id) => available.has(id));
  if (kept.length === groupIds.length) return groupIds;
  return kept.length > 1 ? kept : [];
}

/** Group ids back in the order they sit on the page, which is how a merge reads. */
export function inPageOrder(
  groupIds: string[],
  pageIds: readonly string[],
): string[] {
  const chosen = new Set(groupIds);
  return pageIds.filter((id) => chosen.has(id));
}
