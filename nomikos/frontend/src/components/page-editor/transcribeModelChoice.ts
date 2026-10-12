type TranscribeModelCandidate = {
  id: string;
};

/**
 * Which transcribe model a picker selects once the catalog is loaded.
 *
 * Order: the explicit choice of this session when it still exists, else the
 * resolved project binding, else the first row. Null only when the list is
 * empty, in which case the request omits `model_id` and the backend resolves
 * its own default.
 */
export function resolveTranscribeModelId(
  models: TranscribeModelCandidate[],
  explicitId: string | null = null,
  bindingId: string | null = null,
): string | null {
  if (explicitId && models.some((model) => model.id === explicitId)) {
    return explicitId;
  }
  if (bindingId && models.some((model) => model.id === bindingId)) {
    return bindingId;
  }
  return models[0]?.id ?? null;
}
