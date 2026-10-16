import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type LineResponse,
  type PageTranscriptionTextLineResponse,
  type PartLayoutResponse,
  type TranscriptionLayerResponse,
} from "../../../api/client";
import type { components } from "../../../api/schema";
import { ApiError } from "../../../api/errors";
import { queryClient, taggedMeta } from "../../../api/queryClient";
import { resourceTags } from "../../../api/resources";
import {
  hasAccessToken,
  isUnauthorized,
  redirectToLogin,
} from "../../../auth/session";
import { useBackgroundJobs } from "../../../context/BackgroundJobsContext";
import type { SetState } from "./editorHookTypes";
import {
  fetchPartContent,
  type ApplyIfCurrent,
  type PartContentSetters,
} from "./fetchPartContent";
import { useEditorModels } from "./useEditorModels";

type PairingProgressResponse = components["schemas"]["PairingProgressResponse"];

const EMPTY_LAYOUT: PartLayoutResponse = { blocks: [], lines: [] };
const NO_PAIRING: PairingProgressResponse = {
  paired_lines: 0,
  total_lines: 0,
  percent: 0,
};

/** A 401 never reaches here: the caller redirects to login first. */
function accessMessage(error: ApiError): string {
  if (error.status === 403 || error.status === 404) {
    return "This page is not available to your account.";
  }
  return error.message;
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
 * The part a page turn lands on, when the document on screen already lists
 * it; null means a cold load of a document this hook has not read yet.
 */
function carriedPartFor(
  document: DocumentWithPartsResponse | null,
  projectId: string,
  documentId: string,
  partId: string,
): DocumentPartResponse | null {
  if (!canReuseDocument(document, projectId, documentId)) return null;
  return resolvePart(document, partId);
}

/**
 * The document for a route. Paging through a document re-runs the route
 * effect for every part, and the document itself does not change between
 * them; the cache is what stops each page turn from refetching it.
 */
async function documentFor(
  projectId: string,
  documentId: string,
  initialDocument: DocumentWithPartsResponse | null | undefined,
): Promise<DocumentWithPartsResponse> {
  if (canReuseDocument(initialDocument, projectId, documentId)) {
    return initialDocument;
  }
  return queryClient.fetchQuery({
    queryKey: ["document", projectId, documentId],
    queryFn: () => api.getDocument(projectId, documentId),
    meta: taggedMeta([resourceTags.document(projectId, documentId)]),
  });
}

/** The error a failed route load shows, or null after a redirect to login. */
function routeLoadMessage(error: unknown): string | null {
  if (isUnauthorized(error)) {
    redirectToLogin();
    return null;
  }
  return error instanceof ApiError
    ? accessMessage(error)
    : "Failed to load page.";
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
  const [layout, setLayout] = useState<PartLayoutResponse>(EMPTY_LAYOUT);
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
    PageTranscriptionTextLineResponse[]
  >([]);
  const [pairingProgress, setPairingProgress] = useState(NO_PAIRING);
  const [pairingError, setPairingError] = useState<string | null>(null);
  const models = useEditorModels();
  const {
    setTranscribeModels,
    setSelectedTranscribeModelId,
    setSegmentModels,
    setSelectedSegmentModelId,
    segmentChoiceIsExplicit,
    forgetSegmentChoice,
  } = models;

  /** Every setter `fetchPartContent` writes through, built once. */
  const contentSetters = useMemo<PartContentSetters>(
    () => ({
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
    }),
    [
      setTranscribeModels,
      setSelectedTranscribeModelId,
      setSegmentModels,
      setSelectedSegmentModelId,
    ],
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

  // The route effect runs per route, not per render: these are read from it
  // as they are when it runs. `initialDocument` only matters on the first
  // load for a route key.
  const onRouteChangeRef = useRef(onRouteChange);
  onRouteChangeRef.current = onRouteChange;
  const initialDocumentRef = useRef(initialDocument);
  initialDocumentRef.current = initialDocument;

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
    const forPart =
      <T>(set: SetState<T>): SetState<T> =>
      (value) => {
        if (activePartIdRef.current !== mintedFor) return;
        set(value);
      };
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

  /**
   * Clears what belonged to the previous route. A page turn keeps the
   * document and its document-level state; see fetchPartContent.
   */
  const resetForRoute = useCallback(
    (carriedPart: DocumentPartResponse | null) => {
      setLoading(!carriedPart);
      setPartLoading(true);
      setError(null);
      setLayoutError(null);
      setLineError(null);
      setPart(carriedPart);
      setLayout(EMPTY_LAYOUT);
      setLines([]);
      setTextLines([]);
      setPairingProgress(NO_PAIRING);
      setPairingError(null);
      if (carriedPart) return;
      setDocument(null);
      setTranscriptionLayers([]);
      setSelectedTranscriptionLayerId(null);
      setGroundTruthTranscriptionId(null);
    },
    [],
  );

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
    const apply: ApplyIfCurrent = (setter, value) => {
      if (!cancelled && generation === contentGenerationRef.current) {
        setter(value);
      }
    };

    const carriedPart = carriedPartFor(
      documentRef.current,
      projectId,
      documentId,
      partId,
    );
    if (!carriedPart) forgetSegmentChoice();
    resetForRoute(carriedPart);
    onRouteChangeRef.current?.();

    async function load(projectId: string, documentId: string, partId: string) {
      const doc = await documentFor(
        projectId,
        documentId,
        initialDocumentRef.current,
      );
      if (cancelled) return;
      const selectedPart = resolvePart(doc, partId);
      if (!selectedPart) {
        apply(setError, "This document part was not found.");
        return;
      }
      apply(setDocument, doc);
      apply(setPart, selectedPart);
      await fetchPartContent(
        projectId,
        documentId,
        partId,
        apply,
        contentSetters,
        {
          documentLevel: !carriedPart,
          segmentChoiceIsExplicit,
        },
      );
    }

    void load(projectId, documentId, partId)
      .catch((err: unknown) => {
        const message = routeLoadMessage(err);
        if (message !== null) apply(setError, message);
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
        setPartLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    projectId,
    documentId,
    partId,
    contentSetters,
    resetForRoute,
    forgetSegmentChoice,
    segmentChoiceIsExplicit,
  ]);

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
      const apply: ApplyIfCurrent = (setter, value) => {
        if (cancelled || generation !== contentGenerationRef.current) return;
        setter(value);
      };

      void fetchPartContent(
        projectId,
        documentId,
        partId,
        apply,
        contentSetters,
        {
          documentLevel: true,
          segmentChoiceIsExplicit,
        },
      );
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [
    projectId,
    documentId,
    partId,
    subscribeToJobCompletion,
    contentSetters,
    segmentChoiceIsExplicit,
  ]);

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
    transcribeModels: models.transcribeModels,
    selectedTranscribeModelId: models.selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    segmentModels: models.segmentModels,
    selectedSegmentModelId: models.selectedSegmentModelId,
    setSelectedSegmentModelId: models.chooseSegmentModelId,
    parts,
    partIndex,
  };
}
