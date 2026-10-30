import { useMemo, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import {
  api,
  publicPartMediaUrl,
  type DocumentPartResponse,
  type DocumentWithPartsResponse,
  type PublicLayoutResponse,
  type PublicLineResponse,
} from "../api/client";
import { ApiError } from "../api/errors";
import { resourceTags } from "../api/resources";
import { getAccessToken } from "../auth/storage";
import { useServerQuery } from "../hooks/useServerQuery";
import type { Region } from "../types";
import { linesForPart, publicLinesToRegions } from "../utils/publicLayout";
/**
 * A 404 here is not a failure to report but a state of the document - it is not
 * published, or never existed - and gets its own copy, so the two are kept apart
 * rather than collapsed into one message.
 */
type PublicDocumentData = {
  document: DocumentWithPartsResponse;
  layout: PublicLayoutResponse;
};

type PublicDocumentFailure =
  { kind: "not-found" } | { kind: "message"; text: string };

export interface PublicDocumentModel {
  projectId: string | undefined;
  documentId: string | undefined;
  token: string | null;
  isLoggedIn: boolean;
  document: DocumentWithPartsResponse | null;
  layout: PublicLayoutResponse | null;
  parts: DocumentPartResponse[];
  partTabs: { id: string; label: string }[];
  activePart: DocumentPartResponse | null;
  activePartId: string | null;
  activePartIndex: number;
  partLines: PublicLineResponse[];
  regions: Region[];
  selectedLineIndex: number | null;
  selectedRegionId: number | null;
  canvasView: "image" | "pdf";
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  loading: boolean;
  notFound: boolean;
  errorMessage: string | null;
  selectPart: (partId: string) => void;
  setSelectedLineIndex: (index: number | null) => void;
  setCanvasView: (view: "image" | "pdf") => void;
  selectRegion: (regionId: number | null) => void;
}

function readPublicDocument(
  projectId: string,
  documentId: string,
  token: string | null,
): Promise<PublicDocumentData> {
  return Promise.all([
    api.getPublicDocument(projectId, documentId, token),
    api.getPublicLayout(projectId, documentId, token),
  ]).then(([document, layout]) => ({ document, layout }));
}

function publicDocumentFailure(err: unknown): PublicDocumentFailure {
  return err instanceof ApiError && err.status === 404
    ? { kind: "not-found" }
    : {
        kind: "message",
        text: err instanceof ApiError ? err.message : "Failed to load document",
      };
}

function selectLoadState(
  token: string | null,
  error: PublicDocumentFailure | null,
): { notFound: boolean; errorMessage: string | null } {
  // A missing token disables the query above, which would otherwise leave
  // `loading` stuck true forever (see useServerQuery) - it is the same "not
  // published, or never existed" state as a real 404, not a reason to spin.
  return {
    notFound: !token || error?.kind === "not-found",
    errorMessage: error?.kind === "message" ? error.text : null,
  };
}

function selectActivePart(
  parts: DocumentPartResponse[],
  activePartId: string | null,
): DocumentPartResponse | null {
  return parts.find((part) => part.id === activePartId) ?? parts[0] ?? null;
}

function selectedRegionId(selectedLineIndex: number | null): number | null {
  return selectedLineIndex !== null && selectedLineIndex >= 0
    ? selectedLineIndex + 1
    : null;
}

function partTabLabels(
  parts: DocumentPartResponse[],
): { id: string; label: string }[] {
  return parts.map((part, index) => ({
    id: part.id,
    label: `Page ${index + 1}`,
  }));
}

function activePartPosition(
  parts: DocumentPartResponse[],
  activePart: DocumentPartResponse | null,
): number {
  return activePart
    ? parts.findIndex((part) => part.id === activePart.id) + 1
    : 1;
}

function publicDocumentKey(
  projectId: string | undefined,
  documentId: string | undefined,
  token: string | null,
): readonly [string, string, string, string] | null {
  return projectId && documentId && token
    ? ["public-document", projectId, documentId, token]
    : null;
}

function sortedDocumentParts(
  document: DocumentWithPartsResponse | null,
): DocumentPartResponse[] {
  return [...(document?.parts ?? [])].sort((a, b) => a.order - b.order);
}

function usePublicDocumentQuery(): {
  projectId: string | undefined;
  documentId: string | undefined;
  token: string | null;
  isLoggedIn: boolean;
  document: DocumentWithPartsResponse | null;
  layout: PublicLayoutResponse | null;
  loading: boolean;
  notFound: boolean;
  errorMessage: string | null;
  parts: DocumentPartResponse[];
} {
  const { projectId, documentId } =
    useParams<{ projectId: string; documentId: string }>() ?? {};
  // Read once: the token belongs to this page's own link, not to anything a
  // reader can change from inside the app, so there is no case where it
  // should be re-read after mount.
  const searchParams = useSearchParams();
  const token = searchParams?.get("t") ?? null;
  const isLoggedIn = !!getAccessToken();

  const { data, loading, error } = useServerQuery<
    PublicDocumentData,
    PublicDocumentFailure
  >({
    key: publicDocumentKey(projectId, documentId, token),
    tags: [resourceTags.publicDocument(projectId ?? "", documentId ?? "")],
    read: () => readPublicDocument(projectId!, documentId!, token),
    onError: publicDocumentFailure,
  });

  const document = data?.document ?? null;
  return {
    projectId,
    documentId,
    token,
    isLoggedIn,
    document,
    layout: data?.layout ?? null,
    loading,
    ...selectLoadState(token, error),
    parts: sortedDocumentParts(document),
  };
}

export function usePublicDocument(): PublicDocumentModel {
  const query = usePublicDocumentQuery();
  const { layout, parts, token } = query;
  const [activePartId, setActivePartId] = useState<string | null>(null);
  const [selectedLineIndex, setSelectedLineIndex] = useState<number | null>(
    null,
  );
  const [canvasView, setCanvasView] = useState<"image" | "pdf">("image");

  const activePart = selectActivePart(parts, activePartId);

  // Settled during render rather than in an effect, so the page never spends a
  // commit disagreeing with the tab strip about which page is open.
  if (activePart && activePart.id !== activePartId) {
    setActivePartId(activePart.id);
  }

  /**
   * `PublicPartTabs` echoes the id it was given back through `onChange` on
   * essentially every render, so only a genuine change of page may clear the
   * reader's line selection - hence the guard rather than an effect keyed on
   * the active id.
   */
  function selectPart(partId: string) {
    if (partId === activePartId) return;
    setActivePartId(partId);
    setSelectedLineIndex(null);
    setCanvasView("image");
  }

  const partLines = useMemo(
    () => (activePart ? linesForPart(layout?.lines, activePart.id) : []),
    [layout, activePart],
  );

  const regions = useMemo(() => publicLinesToRegions(partLines), [partLines]);

  return {
    ...query,
    parts,
    partTabs: partTabLabels(parts),
    activePart,
    activePartId,
    activePartIndex: activePartPosition(parts, activePart),
    partLines,
    regions,
    selectedLineIndex,
    selectedRegionId: selectedRegionId(selectedLineIndex),
    canvasView,
    imageUrl: activePart ? publicPartMediaUrl(activePart.id, token) : null,
    imageWidth: activePart?.width ?? 0,
    imageHeight: activePart?.height ?? 0,
    selectPart,
    setSelectedLineIndex,
    setCanvasView,
    selectRegion: (regionId) =>
      setSelectedLineIndex(regionId === null ? null : regionId - 1),
  };
}
