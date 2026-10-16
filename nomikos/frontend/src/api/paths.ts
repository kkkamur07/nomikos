/**
 * API path builders - one spelling per resource, ids always encoded.
 *
 * Server-issued ids are UUIDs today, so the encoding is defensive only; the
 * point is that no call site gets to interpolate a raw id into a path. The
 * one exception with a comment is `unshareProject`, where the id is a user id
 * and a username may contain a `/`.
 */
import type { CursorPageOptions } from "./cursorPagination";
import { API_BASE_URL } from "./http";

/** Every export route takes the same one flag, spelled the same one way. */
export function reviewedOnlyQuery(reviewedOnly: boolean): string {
  return `?reviewed_only=${reviewedOnly ? "true" : "false"}`;
}

export function cursorQuery(
  options: CursorPageOptions,
  extra?: (params: URLSearchParams) => void,
): string {
  const params = new URLSearchParams();
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.limit) params.set("limit", String(options.limit));
  extra?.(params);
  return params.toString();
}

/** `/projects/:projectId/documents/:documentId`, ids encoded. */
export function documentPath(projectId: string, documentId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}`;
}

/** A page of a document: `documentPath` plus `/parts/:partId`. */
export function partPath(
  projectId: string,
  documentId: string,
  partId: string,
): string {
  return `${documentPath(projectId, documentId)}/parts/${encodeURIComponent(partId)}`;
}

/**
 * Every `/public/*` route 404s without `?t=<public_share_token>`, so each
 * caller must merge it into whatever query string it already has rather than
 * assume it is the only param.
 */
export function withShareToken(path: string, token: string | null): string {
  if (!token) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}t=${encodeURIComponent(token)}`;
}

export function publicPartMediaUrl(
  partId: string,
  token: string | null,
): string {
  return withShareToken(
    `${API_BASE_URL}/public/media/parts/${encodeURIComponent(partId)}`,
    token,
  );
}
