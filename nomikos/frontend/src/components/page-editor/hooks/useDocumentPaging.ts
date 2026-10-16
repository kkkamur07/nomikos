import { useCallback, useEffect, useRef, useState } from "react";
import type { DocumentPartResponse } from "../../../api/client";

/** The editor route for one page of a document. */
function partEditorHref(
  projectId: string,
  documentId: string,
  partId: string,
): string {
  return `/projects/${projectId}/documents/${documentId}/parts/${partId}`;
}

/**
 * The pages either side of the one on screen.
 *
 * A plain function rather than part of the hook below, because the ordered
 * parts only exist once the document has loaded, and the document is loaded
 * for whichever page the hook says is active. The dependency runs one way.
 */
function pageNeighbours(
  parts: DocumentPartResponse[],
  activePartId: string | undefined,
): { previousPartId: string | null; nextPartId: string | null } {
  const index = parts.findIndex((part) => part.id === activePartId);
  if (index < 0) return { previousPartId: null, nextPartId: null };
  return {
    previousPartId: index > 0 ? parts[index - 1].id : null,
    nextPartId: index < parts.length - 1 ? parts[index + 1].id : null,
  };
}

/**
 * Page Up / Page Down: a turn to either side, and whether there is a page
 * there. At either end of the document the turn is undefined, so the key
 * stays the browser's.
 */
export function usePageTurns(
  parts: DocumentPartResponse[],
  activePartId: string | undefined,
  goToPart: (partId: string) => void,
) {
  const { previousPartId, nextPartId } = pageNeighbours(parts, activePartId);
  const goToPreviousPage = useCallback(() => {
    if (previousPartId) goToPart(previousPartId);
  }, [previousPartId, goToPart]);
  const goToNextPage = useCallback(() => {
    if (nextPartId) goToPart(nextPartId);
  }, [nextPartId, goToPart]);
  return {
    hasPreviousPage: previousPartId !== null,
    hasNextPage: nextPartId !== null,
    goToPreviousPage,
    goToNextPage,
    turnBack: previousPartId ? goToPreviousPage : undefined,
    turnForward: nextPartId ? goToNextPage : undefined,
  };
}

type UseDocumentPagingArgs = {
  projectId: string | undefined;
  documentId: string | undefined;
  /** The page the address bar currently names. */
  routePartId: string | undefined;
};

/**
 * Which page of the document the editor is showing, and how to move off it.
 *
 * The route stays the record of where the researcher is - a reload or a shared
 * link has to land on the same page - but it is not what the editor reads. The
 * editor reads this state and mirrors it into the URL, so a page turn is a
 * state change inside a mounted editor rather than a route transition that
 * tears the toolbar down and takes the focused control with it. Anything that
 * moves the route on its own, browser Back and Forward above all, is adopted
 * back.
 */
export function useDocumentPaging({
  projectId,
  documentId,
  routePartId,
}: UseDocumentPagingArgs) {
  const [activePartId, setActivePartId] = useState<string | undefined>(
    routePartId,
  );

  const activePartIdRef = useRef(activePartId);
  activePartIdRef.current = activePartId;

  // Only a route change this hook did not make is worth adopting; its own
  // pushes arrive here as the value that is already active.
  const routePartIdRef = useRef(routePartId);
  useEffect(() => {
    if (routePartId === routePartIdRef.current) return;
    routePartIdRef.current = routePartId;
    setActivePartId(routePartId);
  }, [routePartId]);

  const goToPart = useCallback(
    (partId: string) => {
      if (!projectId || !documentId) return;
      if (partId === activePartIdRef.current) return;
      setActivePartId(partId);
      activePartIdRef.current = partId;
      routePartIdRef.current = partId;
      // pushState, not router.push: partId is a dynamic route segment, so a
      // router navigation re-renders the whole editor tree and refetches every
      // query under it - the "entire page reloads" feel when paging. The hook's
      // own state already drives partId, so only the part-scoped queries need to
      // move. Next integrates the native history API: the URL updates, Back and
      // Forward still walk the pages (popstate is a real navigation, and the
      // route-sync effect above picks the new partId up from useParams).
      // push, not replace: paging through a manuscript is history.
      window.history.pushState(
        null,
        "",
        partEditorHref(projectId, documentId, partId),
      );
    },
    [projectId, documentId],
  );

  return { activePartId, goToPart };
}
