/**
 * The registry id of the segment model the old "Default" picker entry
 * resolved to. Display names are free text, so the match below reads the
 * registry id inside `artifact_ref` (`registry://blla-segment?tag=stable`)
 * and never the name.
 */
export const SEGMENT_REGISTRY_MODEL_ID = "blla-segment";

type SegmentModelCandidate = {
  id: string;
  artifact_ref?: string | null;
};

/** The registry id inside `artifact_ref`, or null when it is not one. */
export function segmentRegistryIdOf(
  model: SegmentModelCandidate,
): string | null {
  const match = /^registry:\/\/([^/?#]+)/.exec(model.artifact_ref ?? "");
  return match ? match[1] : null;
}

/**
 * Which segment model a picker selects once the catalog is loaded.
 *
 * Order: the persisted choice when it still exists, else the catalog row
 * whose `artifact_ref` registry id is `blla-segment` (what Default resolved
 * to), else the first row. Null only when the list is empty, in which case
 * the request omits `model_id` and the backend resolves its own default.
 */
export function resolveSegmentModelId(
  models: SegmentModelCandidate[],
  persistedId: string | null = null,
  bindingId: string | null = null,
): string | null {
  if (persistedId && models.some((model) => model.id === persistedId)) {
    return persistedId;
  }
  if (bindingId && models.some((model) => model.id === bindingId)) {
    return bindingId;
  }
  const canonical = models.find(
    (model) => segmentRegistryIdOf(model) === SEGMENT_REGISTRY_MODEL_ID,
  );
  return canonical?.id ?? models[0]?.id ?? null;
}
