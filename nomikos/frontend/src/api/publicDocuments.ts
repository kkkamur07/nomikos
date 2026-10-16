/**
 * Public (share-link) endpoints: document, keyset-paginated layout,
 * transcriptions and exports. Every route takes the share token.
 */
import { apiRequest, fetchBinaryApi } from "./http";
import { cursorQuery, withShareToken } from "./paths";
import type {
  DocumentWithPartsResponse,
  PublicLayoutResponse,
  PublicTranscriptionLayerResponse,
} from "./types";

/**
 * Safety valve for the public layout drain: 500 lines per page, so 400 pages is
 * 200,000 lines. Reaching it does not fail the viewer; the layout comes back
 * with the lines drained so far and the cursor that would continue it.
 */
const PUBLIC_LAYOUT_MAX_PAGES = 400;

async function fetchPublicLayoutPage(
  projectId: string,
  documentId: string,
  token: string | null,
  cursor: string | null,
  signal?: AbortSignal,
): Promise<PublicLayoutResponse> {
  const path = `/public/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}/layout`;
  const query = cursorQuery({ cursor });
  return apiRequest<PublicLayoutResponse>(
    withShareToken(query ? `${path}?${query}` : path, token),
    { skipAuth: true, signal },
  );
}

function publicPartPath(
  projectId: string,
  documentId: string,
  partId: string,
): string {
  return `/public/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}/parts/${encodeURIComponent(partId)}`;
}

export const publicDocumentsApi = {
  getPublicDocument: (
    projectId: string,
    documentId: string,
    token: string | null,
  ) =>
    apiRequest<DocumentWithPartsResponse>(
      withShareToken(
        `/public/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}`,
        token,
      ),
      { skipAuth: true },
    ),

  /**
   * The public layout endpoint is keyset paginated (500 lines per response, the
   * rest behind ``next_cursor``), so a single request only covers the first
   * pages of a long document. The reader needs every line to draw any page, so
   * this drains the cursor and returns one merged response; blocks and
   * ``blocks_truncated`` come from the first page, which carries them all.
   */
  getPublicLayout: async (
    projectId: string,
    documentId: string,
    token: string | null,
    options: { signal?: AbortSignal } = {},
  ): Promise<PublicLayoutResponse> => {
    // Drain every page: blocks come from the first page only, lines from all of them.
    const first = await fetchPublicLayoutPage(
      projectId,
      documentId,
      token,
      null,
      options.signal,
    );
    const lines = [...(first.lines ?? [])];
    const seenCursors = new Set<string>();
    let cursor = first.next_cursor ?? null;
    let pagesFetched = 1;
    while (cursor && pagesFetched < PUBLIC_LAYOUT_MAX_PAGES) {
      if (options.signal?.aborted) {
        throw new DOMException("The request was cancelled.", "AbortError");
      }
      if (seenCursors.has(cursor)) {
        throw new Error("Public layout pagination repeated a cursor.");
      }
      seenCursors.add(cursor);
      const page = await fetchPublicLayoutPage(
        projectId,
        documentId,
        token,
        cursor,
        options.signal,
      );
      lines.push(...(page.lines ?? []));
      cursor = page.next_cursor ?? null;
      pagesFetched += 1;
    }
    return { ...first, lines, next_cursor: cursor };
  },

  listPublicTranscriptions: (
    projectId: string,
    documentId: string,
    token: string | null,
  ) =>
    apiRequest<PublicTranscriptionLayerResponse[]>(
      withShareToken(
        `/public/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}/transcriptions`,
        token,
      ),
      { skipAuth: true },
    ),

  getPublicTranscriptionPdf: (
    projectId: string,
    documentId: string,
    partId: string,
    token: string | null,
  ) =>
    fetchBinaryApi(
      withShareToken(
        `${publicPartPath(projectId, documentId, partId)}/transcription-pdf`,
        token,
      ),
      { skipAuth: true },
    ),

  getPublicPageXml: (
    projectId: string,
    documentId: string,
    partId: string,
    token: string | null,
  ) =>
    fetchBinaryApi(
      withShareToken(
        `${publicPartPath(projectId, documentId, partId)}/page-xml`,
        token,
      ),
      { skipAuth: true },
    ),

  getPublicPageXmlBundle: (
    projectId: string,
    documentId: string,
    partId: string,
    token: string | null,
  ) =>
    fetchBinaryApi(
      withShareToken(
        `${publicPartPath(projectId, documentId, partId)}/page-xml-bundle`,
        token,
      ),
      { skipAuth: true },
    ),
};
