import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type InferenceModelResponse,
  type LineResponse,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import { ApiError } from "../../../api/errors";
import { queryClient, taggedMeta } from "../../../api/queryClient";
import { resourceTags } from "../../../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  redirectToLogin,
} from "../../../auth/session";
import { useBackgroundJobs } from "../../../context/BackgroundJobsContext";
import { resolveSegmentModelId } from "../segmentModelChoice";
import { resolveTranscribeModelId } from "../transcribeModelChoice";

function accessMessage(error: ApiError): string {
  if (error.status === 401) {
    redirectToLogin();
    return "";
  }
  if (error.status === 403 || error.status === 404) {
    return "This page is not available to your account.";
  }
  return error.message;
}

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
  return error instanceof Error ? error.message : fallback;
}

function sortedParts(
  document: DocumentWithPartsResponse,
): DocumentPartResponse[] {
  return [...document.parts].sort((a, b) => a.order - b.order);
}

function resolvePart(
  document: DocumentWithPartsResponse,
  partId: string,
): DocumentPartResponse | null {
  return sortedParts(document).find((item) => item.id === partId) ?? null;
}

function canReuseDocument(
  document: DocumentWithPartsResponse | null | undefined,
  projectId: string,
  documentId: string,
): document is DocumentWithPartsResponse {
  return document?.project_id === projectId && document.id === documentId;
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
): Promise<{
  transcribeModels: InferenceModelResponse[] | null;
  resolvedTranscribeModel: InferenceModelResponse | null;
  segmentModels: InferenceModelResponse[] | null;
  resolvedSegmentModel: InferenceModelResponse | null;
}> {
  let catalogModels: InferenceModelResponse[] | null = null;
  if (catalog) {
    try {
      catalogModels = await catalog;
    } catch {
      catalogModels = [];
    }
  }
  const transcribeModels = catalogModels
    ? catalogModels.filter((model) => model.task === "transcribe")
    : null;
  const segmentModels = catalogModels
    ? catalogModels.filter((model) => model.task === "segment")
    : null;

  const [transcribeResult, segmentResult] = await Promise.allSettled([
    api.resolvePartModelBinding(projectId, documentId, partId, "transcribe"),
    api.resolvePartModelBinding(projectId, documentId, partId, "segment"),
  ]);
  return {
    transcribeModels,
    resolvedTranscribeModel:
      transcribeResult.status === "fulfilled"
        ? transcribeResult.value.model
        : null,
    segmentModels,
    resolvedSegmentModel:
      segmentResult.status === "fulfilled" ? segmentResult.value.model : null,
  };
}

type PartContentSetters = {
  setLayout: Dispatch<SetStateAction<PartLayoutResponse>>;
  setLayoutError: Dispatch<SetStateAction<string | null>>;
  setLines: Dispatch<SetStateAction<LineResponse[]>>;
  setLineError: Dispatch<SetStateAction<string | null>>;
  setTranscriptionLayers: Dispatch<
    SetStateAction<TranscriptionLayerResponse[]>
  >;
  setGroundTruthTranscriptionId: Dispatch<SetStateAction<string | null>>;
  setSelectedTranscriptionLayerId: Dispatch<SetStateAction<string | null>>;
  setPairingError: Dispatch<SetStateAction<string | null>>;
  setTextLines: Dispatch<
    SetStateAction<
      { order: number; text: string; paired_line_id: string | null }[]
    >
  >;
  setPairingProgress: Dispatch<
    SetStateAction<{
      paired_lines: number;
      total_lines: number;
      percent: number;
    }>
  >;
  setTranscribeModels: Dispatch<SetStateAction<InferenceModelResponse[]>>;
  setSelectedTranscribeModelId: Dispatch<SetStateAction<string | null>>;
  setSegmentModels: Dispatch<SetStateAction<InferenceModelResponse[]>>;
  setSelectedSegmentModelId: Dispatch<SetStateAction<string | null>>;
};

/**
 * The layout/lines/transcriptions/pairing/models read for one part.
 *
 * Shared by the route-keyed mount effect below and the job-completion refresh
 * effect: the first runs it once resolving a fresh part, the second re-runs it
 * when a segmentation or OCR job finishes for the part already on screen.
 * `apply` is the caller's own cancelled/stale guard - this function does not
 * know or care which one it was given.
 *
 * `documentLevel` is what separates a page turn from a load. The transcription
 * layers and the model catalog belong to the document and do not change when
 * the page does, so a page turn fetches only what is the page's own: its
 * layout, its Segments, its pairing and its model binding. A cold load and a
 * finished job (which may have written a new layer) fetch everything.
 *
 * `segmentChoiceIsExplicitRef` is what keeps an explicit picker choice for
 * the rest of the document. The picker's change handler sets it, a
 * document-level load clears it, and the selection below reads it at apply
 * time so a choice made while the read is in flight still wins.
 */
async function fetchPartContent(
  projectId: string,
  documentId: string,
  partId: string,
  apply: <T>(setter: (value: T) => void, value: T) => void,
  setters: PartContentSetters,
  {
    documentLevel,
    segmentChoiceIsExplicitRef,
    transcribeChoiceIsExplicitRef,
  }: {
    documentLevel: boolean;
    segmentChoiceIsExplicitRef?: { current: boolean };
    transcribeChoiceIsExplicitRef?: { current: boolean };
  },
): Promise<void> {
  const [
    layoutResult,
    linesResult,
    transcriptionsResult,
    pairingResult,
    modelsResult,
  ] = await Promise.allSettled([
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

  if (layoutResult.status === "fulfilled") {
    apply(setters.setLayout, layoutResult.value ?? { blocks: [], lines: [] });
  } else {
    const err = layoutResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setLayoutError,
      partialLoadMessage(
        err,
        "Layout editing is not available for this page.",
        "Failed to load layout.",
      ),
    );
  }

  if (linesResult.status === "fulfilled") {
    apply(setters.setLines, linesResult.value);
  } else {
    const err = linesResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setLineError,
      partialLoadMessage(
        err,
        "Segment geometry is not available for this page.",
        "Failed to load Segment geometry.",
      ),
    );
  }

  if (transcriptionsResult.status === "fulfilled") {
    const layers = transcriptionsResult.value;
    if (layers !== null) {
      const groundTruth = layers.find((layer) => layer.kind === "ground_truth");
      apply(setters.setTranscriptionLayers, layers);
      apply(setters.setGroundTruthTranscriptionId, groundTruth?.id ?? null);
      apply(
        setters.setSelectedTranscriptionLayerId,
        groundTruth?.id ?? layers[0]?.id ?? null,
      );
    }
  } else {
    const err = transcriptionsResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setPairingError,
      partialLoadMessage(
        err,
        "Pairing is not available for this page.",
        "Failed to load Pairing progress.",
      ),
    );
  }

  if (pairingResult.status === "fulfilled") {
    apply(setters.setTextLines, pairingResult.value.text_lines);
    apply(setters.setPairingProgress, pairingResult.value.pairing_progress);
  } else {
    const err = pairingResult.reason;
    if (isUnauthorized(err)) {
      redirectToLogin();
      return;
    }
    apply(
      setters.setPairingError,
      partialLoadMessage(
        err,
        "Pairing is not available for this page.",
        "Failed to load Pairing progress.",
      ),
    );
  }

  if (modelsResult.status === "fulfilled") {
    const {
      transcribeModels,
      resolvedTranscribeModel,
      segmentModels,
      resolvedSegmentModel,
    } = modelsResult.value;
    // On a page turn the catalogs are null and the catalogs already on screen
    // are kept; a bound model still joins its catalog if the catalog does not
    // list it.
    apply(setters.setTranscribeModels, (current: InferenceModelResponse[]) => {
      const catalog = transcribeModels ?? current;
      return resolvedTranscribeModel &&
        !catalog.some((model) => model.id === resolvedTranscribeModel.id)
        ? [resolvedTranscribeModel, ...catalog]
        : catalog;
    });
    // Same order as the segment picker: an explicit choice wins for the rest
    // of the document, then the binding, then the first catalog row. On a
    // page turn an explicit choice keeps the current value, else the binding
    // if any else the current value is kept.
    apply(setters.setSelectedTranscribeModelId, (current: string | null) => {
      const persisted = transcribeChoiceIsExplicitRef?.current ? current : null;
      if (!transcribeModels) {
        if (persisted) return persisted;
        return resolvedTranscribeModel ? resolvedTranscribeModel.id : current;
      }
      const catalog =
        resolvedTranscribeModel &&
        !transcribeModels.some(
          (model) => model.id === resolvedTranscribeModel.id,
        )
          ? [resolvedTranscribeModel, ...transcribeModels]
          : transcribeModels;
      return resolveTranscribeModelId(
        catalog,
        persisted,
        resolvedTranscribeModel?.id ?? null,
      );
    });
    apply(setters.setSegmentModels, (current: InferenceModelResponse[]) => {
      const catalog = segmentModels ?? current;
      return resolvedSegmentModel &&
        !catalog.some((model) => model.id === resolvedSegmentModel.id)
        ? [resolvedSegmentModel, ...catalog]
        : catalog;
    });
    // There is no empty choice here: one catalog row is always selected, so
    // a new catalog row must not silently change what runs. An explicit
    // choice wins over bindings for the rest of the document: on a
    // document-level load the flag was cleared before this read, so the
    // binding if any else the canonical row else the first row applies; on
    // a page turn an explicit choice keeps the current value, else the
    // binding if any else the current value is kept.
    apply(setters.setSelectedSegmentModelId, (current: string | null) => {
      const persisted = segmentChoiceIsExplicitRef?.current ? current : null;
      if (!segmentModels) {
        if (persisted) return persisted;
        return resolvedSegmentModel ? resolvedSegmentModel.id : current;
      }
      const catalog =
        resolvedSegmentModel &&
        !segmentModels.some((model) => model.id === resolvedSegmentModel.id)
          ? [resolvedSegmentModel, ...segmentModels]
          : segmentModels;
      return resolveSegmentModelId(
        catalog,
        persisted,
        resolvedSegmentModel?.id ?? null,
      );
    });
  } else {
    apply(setters.setTranscribeModels, []);
    apply(setters.setSelectedTranscribeModelId, null);
    apply(setters.setSegmentModels, []);
    // The catalog failed to load: keep whatever was selected (null on a
    // fresh load), so the request omits the model id exactly as before.
    apply(
      setters.setSelectedSegmentModelId,
      (current: string | null) => current,
    );
  }
}

export function usePageEditorData(
  projectId: string | undefined,
  documentId: string | undefined,
  partId: string | undefined,
  onRouteChange?: () => void,
  initialDocument?: DocumentWithPartsResponse | null,
) {
  const [document, setDocument] = useState<DocumentWithPartsResponse | null>(
    null,
  );
  const [part, setPart] = useState<DocumentPartResponse | null>(null);
  const [layout, setLayout] = useState<PartLayoutResponse>({
    blocks: [],
    lines: [],
  });
  const [lines, setLines] = useState<LineResponse[]>([]);
  const [loading, setLoading] = useState(true);
  /**
   * True while this part's own content is in flight, including the page turns
   * that keep the editor on screen. `loading` blanks the editor and can only
   * mean "there is nothing to show yet"; this one means "what you are looking
   * at is the next page, and its Segments have not arrived".
   */
  const [partLoading, setPartLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [lineError, setLineError] = useState<string | null>(null);
  const [transcriptionLayers, setTranscriptionLayers] = useState<
    TranscriptionLayerResponse[]
  >([]);
  const [selectedTranscriptionLayerId, setSelectedTranscriptionLayerId] =
    useState<string | null>(null);
  const [groundTruthTranscriptionId, setGroundTruthTranscriptionId] = useState<
    string | null
  >(null);
  const [textLines, setTextLines] = useState<
    { order: number; text: string; paired_line_id: string | null }[]
  >([]);
  const [pairingProgress, setPairingProgress] = useState({
    paired_lines: 0,
    total_lines: 0,
    percent: 0,
  });
  const [pairingError, setPairingError] = useState<string | null>(null);
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const [selectedTranscribeModelId, setSelectedTranscribeModelId] = useState<
    string | null
  >(null);
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  /**
   * One of the `segmentModels` ids once the catalog loads: the picker has
   * no empty entry, so the segment request always carries it. Null only
   * while the catalog is empty or failed to load, when the select is
   * disabled and `model_id` is not sent. A page turn keeps an explicit
   * choice (see fetchPartContent).
   */
  const [selectedSegmentModelId, setSelectedSegmentModelId] = useState<
    string | null
  >(null);

  /**
   * Whether the segment picker choice came from the user. Set by the
   * picker's change handler, cleared on a document-level load. While set,
   * page turns keep the choice instead of applying the next part's binding.
   */
  const segmentChoiceIsExplicitRef = useRef(false);
  const handleSelectedSegmentModelIdChange = useCallback(
    (value: SetStateAction<string | null>) => {
      segmentChoiceIsExplicitRef.current = true;
      setSelectedSegmentModelId(value);
    },
    [],
  );

  /**
   * Whether the transcribe picker choice came from the user. Same contract as
   * the segment one above: set by the picker's change handler, cleared on a
   * document-level load, read when a read lands so an in-flight choice wins.
   */
  const transcribeChoiceIsExplicitRef = useRef(false);
  const handleSelectedTranscribeModelIdChange = useCallback(
    (value: SetStateAction<string | null>) => {
      transcribeChoiceIsExplicitRef.current = true;
      setSelectedTranscribeModelId(value);
    },
    [],
  );

  /**
   * Which read of this part is the newest, counted across every effect that
   * writes the part's state rather than per effect.
   *
   * Two of them do: the route-mount load below and the job-completion refresh
   * after it, both filling the same setters from their own request. Guarding
   * them separately leaves the case where a job finishes while the first load
   * is still in flight - the refresh lands the new Segments, then the older
   * response overwrites them, and the page sits on pre-job state until a hard
   * reload. One counter, so the loser of that race stays quiet.
   */
  const contentGenerationRef = useRef(0);

  /**
   * The document already on screen, readable from the route effect without
   * making that effect depend on the state it sets.
   *
   * Paging inside one document is what needs it. The next part is already in
   * `document.parts`, so the editor can swap to it without blanking itself:
   * the toolbar, the page rail and whatever has focus stay mounted while the
   * new page's layout, Segments and pairing load underneath. Blanking is
   * reserved for the case that has nothing to show, a cold load of a document
   * this hook has not read yet.
   */
  const documentRef = useRef<DocumentWithPartsResponse | null>(null);
  documentRef.current = document;

  /**
   * Which page the setters handed out of this hook may still write to.
   *
   * A geometry save, a segmentation, a pairing or an OCR request starts on
   * one page and finishes whenever the server answers. If the researcher has
   * paged on by then, the answer belongs to a page that is no longer on
   * screen, and writing it through the shared setters would put the previous
   * page's Segments, layers or pairing under the current page. So the setters
   * given to the mutation hooks are minted per part, and refuse a write once
   * the part they were minted for is not the active one. The load effects
   * below keep the raw setters; they have their own generation counter.
   */
  const activePartIdRef = useRef(partId);
  activePartIdRef.current = partId;
  const partSetters = useMemo(() => {
    const mintedFor = partId;
    const forPart = <T>(set: Dispatch<SetStateAction<T>>) =>
      ((value: SetStateAction<T>) => {
        if (activePartIdRef.current !== mintedFor) return;
        set(value);
      }) as Dispatch<SetStateAction<T>>;
    return {
      setPart: forPart(setPart),
      setLayout: forPart(setLayout),
      setLines: forPart(setLines),
      setLineError: forPart(setLineError),
      setTranscriptionLayers: forPart(setTranscriptionLayers),
      setTextLines: forPart(setTextLines),
      setPairingProgress: forPart(setPairingProgress),
      setPairingError: forPart(setPairingError),
    };
  }, [partId]);

  useEffect(() => {
    if (!projectId || !documentId || !partId) {
      setLoading(false);
      setError("Page route is incomplete.");
      return;
    }
    if (!hasAccessToken()) {
      redirectToLogin();
      return;
    }

    let cancelled = false;
    const generation = ++contentGenerationRef.current;
    const apply = <T>(setter: (value: T) => void, value: T) => {
      if (!cancelled && generation === contentGenerationRef.current) {
        setter(value);
      }
    };

    const carriedDocument = canReuseDocument(
      documentRef.current,
      projectId,
      documentId,
    )
      ? documentRef.current
      : null;
    const carriedPart = carriedDocument
      ? resolvePart(carriedDocument, partId)
      : null;
    const isDocumentLoad = !carriedPart;
    if (isDocumentLoad) {
      segmentChoiceIsExplicitRef.current = false;
      transcribeChoiceIsExplicitRef.current = false;
    }

    setLoading(!carriedPart);
    setPartLoading(true);
    setError(null);
    setLayoutError(null);
    setLineError(null);
    if (carriedPart) {
      setPart(carriedPart);
    } else {
      setDocument(null);
      setPart(null);
    }
    setLayout({ blocks: [], lines: [] });
    setLines([]);
    if (!carriedPart) {
      // Document-level state survives a page turn; see fetchPartContent.
      setTranscriptionLayers([]);
      setSelectedTranscriptionLayerId(null);
      setGroundTruthTranscriptionId(null);
    }
    setTextLines([]);
    setPairingProgress({ paired_lines: 0, total_lines: 0, percent: 0 });
    setPairingError(null);
    onRouteChange?.();

    void (async () => {
      try {
        // Paging through a document re-runs this effect for every part, and the
        // document itself does not change between them; the cache is what stops
        // each page turn from refetching it.
        const doc = canReuseDocument(initialDocument, projectId, documentId)
          ? initialDocument
          : await queryClient.fetchQuery({
              queryKey: ["document", projectId, documentId],
              queryFn: () => api.getDocument(projectId, documentId),
              meta: taggedMeta([resourceTags.document(projectId, documentId)]),
            });
        if (cancelled) return;

        const selectedPart = resolvePart(doc, partId);
        if (!selectedPart) {
          apply(setError, "This document part was not found.");
          return;
        }

        apply(setDocument, doc);
        apply(setPart, selectedPart);
        if (cancelled) return;

        await fetchPartContent(
          projectId,
          documentId,
          partId,
          apply,
          {
            setLayout,
            setLayoutError,
            setLines,
            setLineError,
            setTranscriptionLayers,
            setGroundTruthTranscriptionId,
            setSelectedTranscriptionLayerId,
            setPairingError,
            setTextLines,
            setPairingProgress,
            setTranscribeModels,
            setSelectedTranscribeModelId,
            setSegmentModels,
            setSelectedSegmentModelId,
          },
          {
            documentLevel: !carriedPart,
            segmentChoiceIsExplicitRef,
            transcribeChoiceIsExplicitRef,
          },
        );
      } catch (err) {
        if (isUnauthorized(err)) {
          redirectToLogin();
          return;
        }
        apply(
          setError,
          err instanceof ApiError ? accessMessage(err) : "Failed to load page.",
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
          setPartLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // onRouteChange resets page-local UI state when route params change; omit from deps intentionally.
    // initialDocument is only read on first mount for the current route key.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- route-keyed reset only
  }, [projectId, documentId, partId]);

  const { subscribeToJobCompletion } = useBackgroundJobs();

  /**
   * The gap this closes: the mount effect above re-syncs the page from a
   * promise held by the one component instance whose button was clicked. If
   * that continuation never runs against a live instance - the tab was
   * backgrounded and its timers throttled, this component remounted, or the
   * researcher navigated away and back mid-job - nothing else re-syncs, and
   * only a hard reload recovers. A job going "done" is announced through
   * BackgroundJobsContext regardless of who started it or whether they are
   * still around to see it; this effect is what makes that announcement
   * useful to whichever instance is mounted and showing the affected part
   * right now.
   */
  useEffect(() => {
    if (!projectId || !documentId || !partId) return;

    let cancelled = false;

    const unsubscribe = subscribeToJobCompletion((event) => {
      if (cancelled) return;
      // Only "done" says there is anything new to read; the context is not
      // expected to announce failed or cancelled runs, but nothing here
      // should rely on that rather than say so itself.
      if (event.status !== "done") return;
      // Not this part's job - the instance actually showing that part (if any
      // is mounted) gets its own event.
      if (event.documentPartId !== partId) return;

      const generation = ++contentGenerationRef.current;
      const apply = <T>(setter: (value: T) => void, value: T) => {
        if (cancelled || generation !== contentGenerationRef.current) return;
        setter(value);
      };

      void fetchPartContent(
        projectId,
        documentId,
        partId,
        apply,
        {
          setLayout,
          setLayoutError,
          setLines,
          setLineError,
          setTranscriptionLayers,
          setGroundTruthTranscriptionId,
          setSelectedTranscriptionLayerId,
          setPairingError,
          setTextLines,
          setPairingProgress,
          setTranscribeModels,
          setSelectedTranscribeModelId,
          setSegmentModels,
          setSelectedSegmentModelId,
        },
        {
          documentLevel: true,
          segmentChoiceIsExplicitRef,
          transcribeChoiceIsExplicitRef,
        },
      );
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [projectId, documentId, partId, subscribeToJobCompletion]);

  /** Every page of this document, in page order. The page rail reads it. */
  const parts = useMemo(
    () => (document ? sortedParts(document) : []),
    [document],
  );

  const partIndex =
    document && part
      ? parts.findIndex((item) => item.id === part.id) + 1
      : null;

  return {
    document,
    setDocument,
    part,
    setPart: partSetters.setPart,
    layout,
    setLayout: partSetters.setLayout,
    lines,
    setLines: partSetters.setLines,
    loading,
    partLoading,
    error,
    layoutError,
    lineError,
    setLineError: partSetters.setLineError,
    transcriptionLayers,
    setTranscriptionLayers: partSetters.setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    textLines,
    setTextLines: partSetters.setTextLines,
    pairingProgress,
    setPairingProgress: partSetters.setPairingProgress,
    pairingError,
    setPairingError: partSetters.setPairingError,
    transcribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId: handleSelectedTranscribeModelIdChange,
    segmentModels,
    selectedSegmentModelId,
    setSelectedSegmentModelId: handleSelectedSegmentModelIdChange,
    parts,
    partIndex,
  };
}
