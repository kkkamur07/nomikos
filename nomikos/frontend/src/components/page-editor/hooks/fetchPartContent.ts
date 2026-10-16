import {
  api,
  type InferenceModelResponse,
  type LineResponse,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { ApiError } from "../../../api/errors";
import { isUnauthorized, redirectToLogin } from "../../../auth/session";
import {
  errorMessage,
  type PairingProgressSetter,
  type SetState,
  type TextLinesSetter,
} from "./editorHookTypes";

/** The caller's cancelled/stale guard around every write. */
export type ApplyIfCurrent = <T>(setter: (value: T) => void, value: T) => void;

export type PartContentSetters = {
  setLayout: SetState<PartLayoutResponse>;
  setLayoutError: SetState<string | null>;
  setLines: SetState<LineResponse[]>;
  setLineError: SetState<string | null>;
  setTranscriptionLayers: SetState<TranscriptionLayerResponse[]>;
  setGroundTruthTranscriptionId: SetState<string | null>;
  setSelectedTranscriptionLayerId: SetState<string | null>;
  setPairingError: SetState<string | null>;
  setTextLines: TextLinesSetter;
  setPairingProgress: PairingProgressSetter;
  setTranscribeModels: SetState<InferenceModelResponse[]>;
  setSelectedTranscribeModelId: SetState<string | null>;
  setSegmentModels: SetState<InferenceModelResponse[]>;
  setSelectedSegmentModelId: SetState<string | null>;
};

type EditorModels = {
  transcribeModels: InferenceModelResponse[] | null;
  resolvedTranscribeModel: InferenceModelResponse | null;
  segmentModels: InferenceModelResponse[] | null;
  resolvedSegmentModel: InferenceModelResponse | null;
};

/**
 * The banner for one part of the page that failed to load while the rest of it
 * succeeded. 403 and 404 both mean "not yours to see", which reads better as the
 * feature-specific sentence than as the raw API message.
 */
function partialLoadMessage(
  error: unknown,
  unavailable: string,
  fallback: string,
): string {
  if (
    error instanceof ApiError &&
    (error.status === 403 || error.status === 404)
  ) {
    return unavailable;
  }
  return errorMessage(error, fallback);
}

/**
 * The model pickers' contents for one page. The catalog is the
 * document-level half and is only refetched when the caller passes it; the
 * bindings are resolved for every page, since they can be bound per part.
 *
 * Both pickers read the one `listInferenceModels()` call: the catalog is
 * filtered by task, never fetched twice.
 */
async function loadEditorModels(
  projectId: string,
  documentId: string,
  partId: string,
  catalog: Promise<InferenceModelResponse[]> | null,
): Promise<EditorModels> {
  const catalogModels = catalog ? await catalog.catch(() => []) : null;
  const forTask = (task: InferenceModelResponse["task"]) =>
    catalogModels?.filter((model) => model.task === task) ?? null;

  const [transcribeResult, segmentResult] = await Promise.allSettled([
    api.resolvePartModelBinding(projectId, documentId, partId, "transcribe"),
    api.resolvePartModelBinding(projectId, documentId, partId, "segment"),
  ]);
  return {
    transcribeModels: forTask("transcribe"),
    resolvedTranscribeModel:
      transcribeResult.status === "fulfilled"
        ? transcribeResult.value.model
        : null,
    segmentModels: forTask("segment"),
    resolvedSegmentModel:
      segmentResult.status === "fulfilled" ? segmentResult.value.model : null,
  };
}

/** A bound model joins its catalog when the catalog does not list it. */
function withBoundModel(
  catalog: InferenceModelResponse[],
  bound: InferenceModelResponse | null,
): InferenceModelResponse[] {
  if (!bound || catalog.some((model) => model.id === bound.id)) return catalog;
  return [bound, ...catalog];
}

function applyEditorModels(
  models: EditorModels,
  apply: ApplyIfCurrent,
  setters: PartContentSetters,
  segmentChoiceIsExplicit: () => boolean,
) {
  const {
    transcribeModels,
    resolvedTranscribeModel,
    segmentModels,
    resolvedSegmentModel,
  } = models;
  // On a page turn the catalogs are null and the catalogs already on screen
  // are kept.
  apply(setters.setTranscribeModels, (current: InferenceModelResponse[]) =>
    withBoundModel(transcribeModels ?? current, resolvedTranscribeModel),
  );
  apply(setters.setSelectedTranscribeModelId, (current: string | null) => {
    if (resolvedTranscribeModel) return resolvedTranscribeModel.id;
    return transcribeModels ? (transcribeModels[0]?.id ?? null) : current;
  });
  apply(setters.setSegmentModels, (current: InferenceModelResponse[]) =>
    withBoundModel(segmentModels ?? current, resolvedSegmentModel),
  );
  // Unlike the HTR picker, "Default" (null) is a real choice here, so a
  // missing binding selects it rather than the first catalog row: a new
  // catalog row must not silently change what runs. An explicit choice wins
  // over bindings for the rest of the document: on a document-level load
  // the flag was cleared before this read, so the binding if any else null
  // applies; on a page turn an explicit choice keeps the current value,
  // else the binding if any else the current value is kept.
  apply(setters.setSelectedSegmentModelId, (current: string | null) => {
    if (segmentChoiceIsExplicit()) return current;
    if (resolvedSegmentModel) return resolvedSegmentModel.id;
    return segmentModels ? null : current;
  });
}

function applyNoModels(
  apply: ApplyIfCurrent,
  setters: PartContentSetters,
  segmentChoiceIsExplicit: () => boolean,
) {
  apply(setters.setTranscribeModels, []);
  apply(setters.setSelectedTranscribeModelId, null);
  apply(setters.setSegmentModels, []);
  apply(setters.setSelectedSegmentModelId, (current: string | null) =>
    segmentChoiceIsExplicit() ? current : null,
  );
}

function applyTranscriptionLayers(
  layers: TranscriptionLayerResponse[] | null,
  apply: ApplyIfCurrent,
  setters: PartContentSetters,
) {
  // Null on a page turn: the document's layers on screen are kept.
  if (layers === null) return;
  const groundTruth = layers.find((layer) => layer.kind === "ground_truth");
  apply(setters.setTranscriptionLayers, layers);
  apply(setters.setGroundTruthTranscriptionId, groundTruth?.id ?? null);
  apply(
    setters.setSelectedTranscriptionLayerId,
    groundTruth?.id ?? layers[0]?.id ?? null,
  );
}

type PartialRead<T> = {
  result: PromiseSettledResult<T>;
  onValue: (value: T) => void;
  setError: (message: string) => void;
  unavailable: string;
  fallback: string;
};

/**
 * Applies one of the page's reads. A failure shows its own banner and leaves
 * the other reads alone, except a 401, which sends the researcher to log in
 * and stops the rest. Returns false when it did that.
 */
function applyPartialRead<T>({
  result,
  onValue,
  setError,
  unavailable,
  fallback,
}: PartialRead<T>): boolean {
  if (result.status === "fulfilled") {
    onValue(result.value);
    return true;
  }
  if (isUnauthorized(result.reason)) {
    redirectToLogin();
    return false;
  }
  setError(partialLoadMessage(result.reason, unavailable, fallback));
  return true;
}

/**
 * The layout/lines/transcriptions/pairing/models read for one part.
 *
 * Shared by the route-keyed mount effect and the job-completion refresh
 * effect of `usePageEditorData`: the first runs it once resolving a fresh
 * part, the second re-runs it when a segmentation or OCR job finishes for the
 * part already on screen. `apply` is the caller's own cancelled/stale guard -
 * this function does not know or care which one it was given.
 *
 * `documentLevel` is what separates a page turn from a load. The transcription
 * layers and the model catalog belong to the document and do not change when
 * the page does, so a page turn fetches only what is the page's own: its
 * layout, its Segments, its pairing and its model binding. A cold load and a
 * finished job (which may have written a new layer) fetch everything.
 *
 * `segmentChoiceIsExplicit` is what keeps an explicit picker choice for the
 * rest of the document. The picker's change handler sets it, a
 * document-level load clears it, and the selection reads it at apply time so
 * a choice made while the read is in flight still wins.
 */
export async function fetchPartContent(
  projectId: string,
  documentId: string,
  partId: string,
  apply: ApplyIfCurrent,
  setters: PartContentSetters,
  {
    documentLevel,
    segmentChoiceIsExplicit,
  }: {
    documentLevel: boolean;
    segmentChoiceIsExplicit: () => boolean;
  },
): Promise<void> {
  const [layout, lines, transcriptions, pairing, models] =
    await Promise.allSettled([
      api.getPartLayout(projectId, documentId, partId),
      api.listPartLines(projectId, documentId, partId),
      documentLevel
        ? api.listTranscriptions(projectId, documentId)
        : Promise.resolve(null),
      api.getPagePairing(projectId, documentId, partId),
      loadEditorModels(
        projectId,
        documentId,
        partId,
        documentLevel ? api.listInferenceModels() : null,
      ),
    ]);

  const setPairingError = (message: string) =>
    apply(setters.setPairingError, message);
  const reads = [
    () =>
      applyPartialRead({
        result: layout,
        onValue: (value) =>
          apply(setters.setLayout, value ?? { blocks: [], lines: [] }),
        setError: (message) => apply(setters.setLayoutError, message),
        unavailable: "Layout editing is not available for this page.",
        fallback: "Failed to load layout.",
      }),
    () =>
      applyPartialRead({
        result: lines,
        onValue: (value) => apply(setters.setLines, value),
        setError: (message) => apply(setters.setLineError, message),
        unavailable: "Segment geometry is not available for this page.",
        fallback: "Failed to load Segment geometry.",
      }),
    () =>
      applyPartialRead({
        result: transcriptions,
        onValue: (value) => applyTranscriptionLayers(value, apply, setters),
        setError: setPairingError,
        unavailable: "Transcriptions are not available for this page.",
        fallback: "Failed to load transcriptions.",
      }),
    () =>
      applyPartialRead({
        result: pairing,
        onValue: (value) => {
          apply(setters.setTextLines, value.text_lines);
          apply(setters.setPairingProgress, value.pairing_progress);
        },
        setError: setPairingError,
        unavailable: "Pairing is not available for this page.",
        fallback: "Failed to load Pairing progress.",
      }),
  ];
  // In order, stopping at a 401: the reads after it are not applied.
  for (const read of reads) {
    if (!read()) return;
  }

  if (models.status === "fulfilled") {
    applyEditorModels(models.value, apply, setters, segmentChoiceIsExplicit);
  } else {
    applyNoModels(apply, setters, segmentChoiceIsExplicit);
  }
}
