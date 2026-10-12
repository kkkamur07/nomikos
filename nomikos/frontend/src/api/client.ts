/**
 * Platform API client - short-lived JWT in memory, OpenAPI-aligned types.
 * Regenerate types: `npm run codegen:api` (after `python scripts/platform/export_openapi.py`).
 */
import { redirectToLogin } from "../auth/session";
import {
  clearCsrfToken,
  getAccessToken,
  getCsrfToken,
  setAccessToken,
  setCsrfToken,
} from "../auth/storage";
import { JOB_WAIT_POLL_INTERVAL_MS } from "../utils/jobPolling";
import { waitForSubscribedJob } from "../utils/jobSubscription";
import {
  collectCursorPages,
  type CursorPage,
  type CursorPageOptions,
} from "./cursorPagination";
import { ApiError, parseApiError } from "./errors";
import { dedupedGet, getAuthRequestVersion } from "./getCache";
import type { components } from "./schema";

export type TokenResponse = components["schemas"]["TokenResponse"];
export type UserResponse = components["schemas"]["UserResponse"];
export type AdminOverview = components["schemas"]["AdminOverview"];
export type FeedbackCreateRequest =
  components["schemas"]["FeedbackCreateRequest"];
export type FeedbackResponse = components["schemas"]["FeedbackResponse"];
export type FeedbackRow = components["schemas"]["FeedbackRow"];
export type LoginRequest = components["schemas"]["LoginRequest"];
export type RegisterRequest = components["schemas"]["RegisterRequest"];
export type ProjectResponse = components["schemas"]["ProjectResponse"];
export type ProjectCreateRequest =
  components["schemas"]["ProjectCreateRequest"];
export type ProjectUpdateRequest =
  components["schemas"]["ProjectUpdateRequest"];
export type ShareUserRequest = components["schemas"]["ShareUserRequest"];
export type ProjectCollaboratorResponse =
  components["schemas"]["ProjectCollaboratorResponse"];
export type DocumentResponse = components["schemas"]["DocumentResponse"];
export type DocumentWithPartsResponse =
  components["schemas"]["DocumentWithPartsResponse"];
export type DocumentCreateRequest =
  components["schemas"]["DocumentCreateRequest"];
export type DocumentUpdateRequest =
  components["schemas"]["DocumentUpdateRequest"];
export type DocumentPartResponse =
  components["schemas"]["DocumentPartResponse"];
export type DocumentPartUpdateRequest =
  components["schemas"]["DocumentPartUpdateRequest"];
export type PartsPublishedUpdateRequest =
  components["schemas"]["PartsPublishedUpdateRequest"];
export type DocumentWorkflow = components["schemas"]["DocumentWorkflow"];
export type ReorderPartsRequest = components["schemas"]["ReorderPartsRequest"];

export type PartUploadBeginRequest =
  components["schemas"]["PartUploadBeginRequest"];
export type PartUploadBeginResponse =
  components["schemas"]["PartUploadBeginResponse"];
export type PartUploadFinalizeRequest =
  components["schemas"]["PartUploadFinalizeRequest"];
export type PublicLayoutResponse =
  components["schemas"]["PublicLayoutResponse"];
export type PublicLineResponse = NonNullable<
  PublicLayoutResponse["lines"]
>[number];
export type PublicTranscriptionLayerResponse =
  components["schemas"]["PublicTranscriptionLayerResponse"];
export type TranscriptionLayerResponse =
  components["schemas"]["TranscriptionLayerResponse"];
export type LineTranscriptionResponse =
  components["schemas"]["LineTranscriptionResponse"];
export type CharacterConfidence = components["schemas"]["CharacterConfidence"];
export type JobResponse = components["schemas"]["JobResponse"];
export type JobStatus = components["schemas"]["JobStatus"];
/** The **inference host** one job runs on. Fixed at submission (ADR 0002). */
export type ExecutionTarget = components["schemas"]["ExecutionTarget"];
export type ExecutionPreferenceResponse =
  components["schemas"]["ExecutionPreferenceResponse"];
export type EnqueueJobResponse = components["schemas"]["EnqueueJobResponse"];
/** Every tuning field is server-defaulted, so any subset is a valid body. */
export type SegmentPartRequest = Partial<
  components["schemas"]["SegmentPartRequest"]
>;
export type TranscribePartRequest =
  components["schemas"]["TranscribePartRequest"];
export type InferenceModelResponse =
  components["schemas"]["InferenceModelResponse"];
export type InferenceTask = components["schemas"]["InferenceTask"];
export type ResolvedModelBindingResponse =
  components["schemas"]["ResolvedModelBindingResponse"];
export type ModelBindingResponse =
  components["schemas"]["ModelBindingResponse"];
export type EnqueueTestJobRequest =
  components["schemas"]["EnqueueTestJobRequest"];
export type EnqueueTestJobResponse =
  components["schemas"]["EnqueueTestJobResponse"];
/**
 * What a finished transcribe carries back. The server types it as a bare
 * `dict` on the job (`JobResponse.result`), so the shape is the client's
 * claim, made once here and read by `applyTranscribeResult`.
 */
export type TranscribeJobResult = {
  transcription_id: string;
  lines: Array<{ line_id: string; text: string; confidence: number }>;
};
/**
 * The one place the generated wire types are narrowed by hand.
 *
 * The server types a geometry column as a bare `dict` and a point list as
 * `list[list[float]]`, so the schema can only say "an object" and
 * "number[][]". The canvas needs "an `[x, y]` pair" and "a point list, or an
 * object wrapping one", and every helper in `canvasGeometry.ts` is written
 * against those. The claim is checked where it's consumed, not where it
 * arrives: `normalizeGeometryPoints` unwraps the object forms and drops
 * anything that isn't a pair of numbers.
 *
 * Nothing else about these payloads is asserted here. Every field below comes
 * from `schema.d.ts`, so a backend change reaches the frontend through
 * `codegen:api` and fails the typecheck instead of being shadowed by a
 * second copy of the type.
 */
export type LinePoint = [number, number];
/** The same pair, under the name the layout overlay uses. */
export type LayoutPoint = LinePoint;
export type GeometryValue =
  | LayoutPoint[]
  | {
      points?: LayoutPoint[];
      type?: string;
      coordinates?: LayoutPoint[];
    };

/** A geometry column as the schema renders it: an object with no shape. */
type WireGeometry = { [key: string]: unknown };

/** Applies the narrowing above to a generated shape, field by field. */
type NarrowGeometry<T> = {
  [K in keyof T]: K extends "points" | "kraken_ceiling"
    ? Exclude<T[K], number[][]> | LinePoint[]
    : K extends "baseline" | "mask" | "box"
      ? Exclude<T[K], WireGeometry> | GeometryValue
      : T[K];
};

export type BlockResponse = NarrowGeometry<
  components["schemas"]["BlockResponse"]
>;
export type BlockCreateRequest = NarrowGeometry<
  components["schemas"]["BlockCreateRequest"]
>;
export type BlockPatchRequest = NarrowGeometry<
  components["schemas"]["BlockPatchRequest"]
>;
export type LineGeometryKind = components["schemas"]["LineGeometryKind"];
export type LineSource = components["schemas"]["LineSource"];
export type LineResponse = Omit<
  NarrowGeometry<components["schemas"]["LineResponse"]>,
  "line_transcriptions"
> & {
  /**
   * Generated as optional only because the server field has a
   * `default_factory` and so never reaches the schema's `required` list.
   * FastAPI serializes it on every response.
   */
  line_transcriptions: LineTranscriptionResponse[];
};
export type LineCreateRequest = NarrowGeometry<
  components["schemas"]["LineCreateRequest"]
>;
export type LinePatchRequest = NarrowGeometry<
  components["schemas"]["LinePatchRequest"]
>;
export type LineUpsertRequest = NarrowGeometry<
  components["schemas"]["LineUpsertRequest"]
>;
/** `LinesReplaceRequest`, with the narrowing carried into its lines. */
export type LinesReplaceRequest = {
  lines?: LineUpsertRequest[];
};
/** The geometry-only slice of `LinePatchRequest` a manual edit sends. */
export type UpdateLineGeometryRequest = Pick<
  LinePatchRequest,
  "baseline" | "mask"
>;
export type ResetPartLayoutRequest =
  components["schemas"]["LayoutResetRequest"];

/**
 * The editor's working copy of a layout, not a response shape.
 *
 * `GET .../layout` answers with whole blocks and lines, but the editor only
 * reads and rewrites their geometry, and `syncLayoutLinesFromSegments` folds
 * a segment edit back in as a line carrying geometry and nothing else. Field
 * names and types are the generated ones; the optionality is the editor's own.
 */
export type LayoutBlockResponse = Pick<BlockResponse, "id"> &
  Partial<Pick<BlockResponse, "box" | "manual_geometry" | "order" | "label">>;
export type SegmentHealthResponse =
  components["schemas"]["SegmentHealthResponse"];
export type SegmentSuspectResponse =
  components["schemas"]["SegmentSuspectResponse"];
export type SegmentSplitResponse =
  components["schemas"]["SegmentSplitResponse"];
export type SegmentMergeResponse =
  components["schemas"]["SegmentMergeResponse"];
export type SegmentOverlapResponse =
  components["schemas"]["SegmentOverlapResponse"];

export type LayoutLineResponse = Pick<LineResponse, "id"> &
  Partial<
    Pick<LineResponse, "block_id" | "baseline" | "mask" | "manual_geometry">
  >;
export type PartLayoutResponse = {
  blocks: LayoutBlockResponse[];
  lines: LayoutLineResponse[];
};

export type PageTranscriptionTextLineResponse =
  components["schemas"]["PageTranscriptionTextLineResponse"];
export type PairingProgressResponse =
  components["schemas"]["PairingProgressResponse"];
export type PagePairingResponse = components["schemas"]["PagePairingResponse"];
export type PageTranscriptionImportRequest =
  components["schemas"]["PageTranscriptionImportRequest"];
export type PairTextLineRequest = components["schemas"]["PairTextLineRequest"];
export type LineTranscriptionPatchRequest =
  components["schemas"]["LineTranscriptionPatchRequest"];
export type CopyToGroundTruthRequest =
  components["schemas"]["CopyToGroundTruthRequest"];
export type CopyToGroundTruthResponse =
  components["schemas"]["CopyToGroundTruthResponse"];

/*
 * ─── Document-level workflow and export types, written by hand ──────────
 *
 * The routes these describe are new, so `schema.d.ts` does not carry them yet.
 * Every type below should collapse into `components["schemas"][...]` once
 * `npm run generate:api` has re-exported the OpenAPI document and regenerated
 * the artifacts; nothing here is meant to outlive that run. Until then this is
 * the only hand-written copy of a wire shape in the client, and a backend
 * change to any of it will reach the frontend as a runtime surprise rather
 * than a failing typecheck.
 */

/** What the Workflow and Download menus put in their count badges. */
export type DocumentWorkflowCounts = {
  total: number;
  reviewed: number;
  unsegmented: number;
  unpaired: number;
};

/**
 * Which pages a batch segment touches.
 *
 * `unsegmented` is the safe one: it skips any page that already has lines, so
 * nothing anybody has transcribed is thrown away. `all` re-runs every page; lines
 * a human touched (approved text, a pairing, hand-drawn geometry) survive, every
 * other line is redrawn and the model's unapproved text on it is discarded, which
 * is why the item that sends it is confirmed at the call site.
 */
export type DocumentSegmentScope = "unsegmented" | "all";

/** Which pages a batch transcribe touches. */
export type DocumentTranscribeScope = "unpaired" | "all";

export type DocumentSegmentJobRequest = {
  scope: DocumentSegmentScope;
  /** Null means "whatever the document resolves to", which is all the UI offers. */
  model_id: string | null;
};

export type DocumentTranscribeJobRequest = {
  scope: DocumentTranscribeScope;
  model_id: string | null;
};

/**
 * A 202 from either batch route. `skipped` is the count the scope excluded,
 * so "queued 0, skipped 18" is a complete answer rather than a silent no-op.
 * Each entry of `jobs` is the same shape a single enqueue returns, host
 * included.
 */
export type DocumentBatchJobResponse =
  components["schemas"]["DocumentBatchJobsResponse"];

export type PageResponse<T> = CursorPage<T>;

export type ListPageOptions = CursorPageOptions;

/** Every export route takes the same one flag, spelled the same one way. */
function reviewedOnlyQuery(reviewedOnly: boolean): string {
  return `?reviewed_only=${reviewedOnly ? "true" : "false"}`;
}

function cursorQuery(
  options: CursorPageOptions,
  extra?: (params: URLSearchParams) => void,
): string {
  const params = new URLSearchParams();
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.limit) params.set("limit", String(options.limit));
  extra?.(params);
  return params.toString();
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";
export const API_ORIGIN = new URL(API_BASE_URL).origin;
const CSRF_COOKIE_NAME =
  process.env.NEXT_PUBLIC_CSRF_COOKIE_NAME || "greekocr-csrf";

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: unknown;
  skipAuth?: boolean;
  /** Bypass in-flight GET deduplication (default: dedupe GETs). */
  skipDedup?: boolean;
};

let refreshPromise: Promise<TokenResponse> | null = null;

/** The CSRF token as `document.cookie` exposes it, or null if it is not there. */
function csrfCookieToken(): string | null {
  if (typeof document === "undefined") return null;
  const encodedName = `${encodeURIComponent(CSRF_COOKIE_NAME)}=`;
  const cookie = document.cookie
    .split("; ")
    .find((item) => item.startsWith(encodedName));
  if (!cookie) return null;
  try {
    return decodeURIComponent(cookie.slice(encodedName.length));
  } catch {
    return null;
  }
}

/**
 * The CSRF token to echo back, from whichever channel has it.
 *
 * The server sends the same value twice: in the `TokenResponse` body and in
 * the `greekocr-csrf` cookie. The body copy is preferred because a cookie
 * policy can't take it away, the cookie is set on `api.nomikos.app` for
 * `.nomikos.app` purely so script on `app.nomikos.app` can read it, and that
 * sibling-subdomain read is what a stricter browser blocks. The cookie stays
 * as the fallback for a session established before this code shipped, or a
 * tab that never called an auth route.
 */
function csrfToken(): string | null {
  return getCsrfToken() ?? csrfCookieToken();
}

function addCsrfHeader(headers: Headers, method: string): void {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(method)) return;
  const token = csrfToken();
  if (token) headers.set("X-CSRF-Token", token);
}

/** Record the CSRF token an auth response carried, if it carried one. */
function rememberCsrfToken(token: TokenResponse): TokenResponse {
  // Optional on the wire: this client can be talking to an API deployed before
  // the field existed, in which case the cookie is still the only channel.
  if (token.csrf_token) setCsrfToken(token.csrf_token);
  return token;
}

/**
 * POST one of the two routes that check CSRF, retrying once from the cookie.
 *
 * The in-memory token belongs to a single tab, and `/auth/refresh` rotates
 * the session's token for every tab at once, so a second tab's copy goes
 * stale the moment the first one refreshes and gets a 403. The cookie is
 * shared by every tab and always current, so one retry restores the same
 * behavior as reading the cookie on every request. Skipped when the cookie
 * is unreadable or already matches memory, so Safari costs no extra request.
 */
async function postCsrfProtected<T>(path: string): Promise<T> {
  try {
    return await apiRequest<T>(path, { method: "POST", skipAuth: true });
  } catch (error) {
    const cookieToken = csrfCookieToken();
    if (
      !(error instanceof ApiError) ||
      error.status !== 403 ||
      cookieToken === null ||
      cookieToken === getCsrfToken()
    ) {
      throw error;
    }
    clearCsrfToken();
    return apiRequest<T>(path, { method: "POST", skipAuth: true });
  }
}

/**
 * Refresh the cookie-backed session once for all concurrent callers.
 * The refresh route deliberately bypasses auth recovery to avoid recursion.
 */
export function refreshAccessToken(): Promise<TokenResponse> {
  refreshPromise ??= postCsrfProtected<TokenResponse>("/auth/refresh")
    .then(rememberCsrfToken)
    .then((token) => {
      setAccessToken(token.access_token);
      return token;
    })
    .finally(() => {
      refreshPromise = null;
    });
  return refreshPromise;
}

/**
 * Fetch a protected resource, refreshing the access token once after a 401.
 * It is shared by API requests and the event-stream client so each path has
 * the same recovery and sign-out behavior.
 */
export async function fetchWithAuthRecovery(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  return fetchWithAuthRecoveryAttempt(input, init, false);
}

async function fetchWithAuthRecoveryAttempt(
  input: RequestInfo | URL,
  init: RequestInit,
  retried: boolean,
): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(input, {
    ...init,
    credentials: init.credentials ?? "include",
    headers,
  });
  if (response.status !== 401) return response;

  if (retried) {
    redirectToLogin();
    throw new ApiError("Unauthorized", 401);
  }

  try {
    await refreshAccessToken();
  } catch {
    redirectToLogin();
    throw new ApiError("Unauthorized", 401);
  }

  return fetchWithAuthRecoveryAttempt(input, init, true);
}

async function requestApiResponse(
  path: string,
  options: RequestOptions,
): Promise<Response> {
  const { body, skipAuth, headers: initHeaders, ...rest } = options;
  const headers = new Headers(initHeaders);
  const method = (options.method ?? "GET").toUpperCase();

  if (body !== undefined && !(body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  addCsrfHeader(headers, method);

  const init: RequestInit = {
    ...rest,
    credentials: "include",
    headers,
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  };

  if (skipAuth) {
    return fetch(`${API_BASE_URL}${path}`, init);
  }
  return fetchWithAuthRecovery(`${API_BASE_URL}${path}`, init);
}

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const method = (options.method ?? "GET").toUpperCase();
  // Abort signals belong to one component lifecycle. Sharing an abortable
  // fetch lets one Strict Mode cleanup cancel a newer mount's request.
  if (method === "GET" && !options.skipDedup && !options.signal) {
    return dedupedGet(`auth:${getAuthRequestVersion()} GET ${path}`, () =>
      apiRequest<T>(path, { ...options, skipDedup: true }),
    );
  }

  const response = await requestApiResponse(path, options);

  if (!response.ok) {
    throw await parseApiError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function fetchBinaryApi(
  path: string,
  options: RequestOptions = {},
): Promise<Blob> {
  const response = await requestApiResponse(path, options);

  if (!response.ok) {
    throw await parseApiError(response);
  }

  return response.blob();
}

/**
 * Every `/public/*` route 404s without `?t=<public_share_token>`, so each
 * caller must merge it into whatever query string it already has rather than
 * assume it is the only param.
 */
/**
 * Safety valve for the public layout drain: 500 lines per page, so 400 pages is
 * 200,000 lines. Reaching it does not fail the viewer; the layout comes back
 * with the lines drained so far and the cursor that would continue it.
 */
const PUBLIC_LAYOUT_MAX_PAGES = 400;

function withShareToken(path: string, token: string | null): string {
  if (!token) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}t=${encodeURIComponent(token)}`;
}

export function publicPartMediaUrl(
  partId: string,
  token: string | null,
): string {
  return withShareToken(`${API_BASE_URL}/public/media/parts/${partId}`, token);
}

export const api = {
  login: (body: LoginRequest) =>
    apiRequest<TokenResponse>("/auth/login", {
      method: "POST",
      body,
      skipAuth: true,
    }).then(rememberCsrfToken),

  register: (body: RegisterRequest) =>
    apiRequest<TokenResponse>("/auth/register", {
      method: "POST",
      body,
      skipAuth: true,
    }).then(rememberCsrfToken),

  refresh: refreshAccessToken,

  // Revoking the session server-side is CSRF-checked too, so it gets the same
  // cookie retry: a tab whose token another tab rotated must still be able to
  // sign out for real, not just forget its own copy of the credentials.
  logout: () => postCsrfProtected<void>("/auth/logout"),

  me: () => apiRequest<UserResponse>("/me"),

  /**
   * `/me` signed with a given token and without auth recovery: a failure here
   * must never refresh or clear the session, because the caller only uses it
   * to decide whether to show a link.
   */
  meWithoutRecovery: (accessToken: string) =>
    apiRequest<UserResponse>("/me", {
      skipAuth: true,
      skipDedup: true,
      headers: { Authorization: `Bearer ${accessToken}` },
    }),

  /** Everything the admin dashboard shows; 403 unless `users.is_admin`. */
  adminOverview: (days: number) =>
    apiRequest<AdminOverview>(`/admin/overview?days=${days}`),

  /** Newest feedback notes with their authors; 403 unless `users.is_admin`. */
  listFeedback: (limit = 50) =>
    apiRequest<FeedbackRow[]>(`/admin/feedback?limit=${limit}`),

  /** Send a short note from anywhere; 429 past 20 notes in an hour. */
  sendFeedback: (body: FeedbackCreateRequest) =>
    apiRequest<FeedbackResponse>("/feedback", { method: "POST", body }),

  /**
   * The account-level **host preference** - "use my computer when it is
   * available" - plus what it resolves to right now.
   *
   * There is deliberately no per-job variant of either call: a researcher
   * cannot know which host is faster for a given page, so the choice is made
   * once for the account and announced on each job (ADR 0002).
   */
  getExecutionPreference: (options: { signal?: AbortSignal } = {}) =>
    apiRequest<ExecutionPreferenceResponse>("/account/execution-target", {
      signal: options.signal,
    }),

  setExecutionPreference: (preferLocalInference: boolean) =>
    apiRequest<ExecutionPreferenceResponse>("/account/execution-target", {
      method: "PUT",
      body: { prefer_local_inference: preferLocalInference },
    }),

  listProjectsPage: (options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<ProjectResponse>>(
      query ? `/projects?${query}` : "/projects",
      {
        signal: options.signal,
      },
    );
  },

  listProjects: (options?: { maxPages?: number; signal?: AbortSignal }) =>
    collectCursorPages(
      (pageOptions) => api.listProjectsPage(pageOptions),
      options,
    ),

  createProject: (body: ProjectCreateRequest) =>
    apiRequest<ProjectResponse>("/projects", { method: "POST", body }),

  getProject: (projectId: string) =>
    apiRequest<ProjectResponse>(`/projects/${projectId}`),

  updateProject: (projectId: string, body: ProjectUpdateRequest) =>
    apiRequest<ProjectResponse>(`/projects/${projectId}`, {
      method: "PATCH",
      body,
    }),

  deleteProject: (projectId: string) =>
    apiRequest<void>(`/projects/${projectId}`, { method: "DELETE" }),

  /** Owner-only: the list carries collaborators' emails. */
  listProjectCollaborators: (projectId: string) =>
    apiRequest<ProjectCollaboratorResponse[]>(`/projects/${projectId}/share`),

  shareProject: (projectId: string, body: ShareUserRequest) =>
    apiRequest<void>(`/projects/${projectId}/share`, {
      method: "POST",
      body,
    }),

  /** By user id: a username may contain a `/`, which no encoding survives in a path segment. */
  unshareProject: (projectId: string, userId: string) =>
    apiRequest<void>(
      `/projects/${projectId}/share/${encodeURIComponent(userId)}`,
      { method: "DELETE" },
    ),

  listDocumentsPage: (
    projectId: string,
    includeArchived = false,
    options: ListPageOptions = {},
  ) => {
    const query = cursorQuery(options, (params) =>
      params.set("include_archived", String(includeArchived)),
    );
    return apiRequest<PageResponse<DocumentResponse>>(
      `/projects/${projectId}/documents?${query}`,
      { signal: options.signal },
    );
  },

  listDocuments: (
    projectId: string,
    includeArchived = false,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) =>
        api.listDocumentsPage(projectId, includeArchived, pageOptions),
      options,
    ),

  createDocument: (projectId: string, body: DocumentCreateRequest) =>
    apiRequest<DocumentResponse>(`/projects/${projectId}/documents`, {
      method: "POST",
      body,
    }),

  getDocument: (projectId: string, documentId: string) =>
    apiRequest<DocumentWithPartsResponse>(
      `/projects/${projectId}/documents/${documentId}`,
    ),

  updateDocument: (
    projectId: string,
    documentId: string,
    body: DocumentUpdateRequest,
  ) =>
    apiRequest<DocumentResponse>(
      `/projects/${projectId}/documents/${documentId}`,
      {
        method: "PATCH",
        body,
      },
    ),

  deleteDocument: (projectId: string, documentId: string) =>
    apiRequest<void>(`/projects/${projectId}/documents/${documentId}`, {
      method: "DELETE",
    }),

  /**
   * Mint a fresh share token. Owner-only, and every link built from the old
   * token stops resolving the moment this returns.
   */
  rotateDocumentShareToken: (projectId: string, documentId: string) =>
    apiRequest<DocumentResponse>(
      `/projects/${projectId}/documents/${documentId}/share-token/rotate`,
      { method: "POST" },
    ),

  /**
   * The four numbers the document action menus label themselves with.
   *
   * Deliberately a separate read from the document: the parts list carries
   * `reviewed` but says nothing about whether a page has lines or a pairing,
   * so the counts a Workflow menu needs cannot be derived from what this page
   * already holds.
   */
  getDocumentWorkflowCounts: (projectId: string, documentId: string) =>
    apiRequest<DocumentWorkflowCounts>(
      `/projects/${projectId}/documents/${documentId}/workflow-counts`,
    ),

  /** Zip of the PAGE XML for every page next to the image it describes. */
  exportDocumentPageXml: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/export/page-xml${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentTranscriptionPdf: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/export/transcription-pdf${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentText: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/export/text${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  exportDocumentDocx: (
    projectId: string,
    documentId: string,
    reviewedOnly = false,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/export/docx${reviewedOnlyQuery(reviewedOnly)}`,
    ),

  enqueueDocumentSegment: (
    projectId: string,
    documentId: string,
    body: DocumentSegmentJobRequest,
  ) =>
    apiRequest<DocumentBatchJobResponse>(
      `/projects/${projectId}/documents/${documentId}/jobs/segment`,
      { method: "POST", body },
    ),

  enqueueDocumentTranscribe: (
    projectId: string,
    documentId: string,
    body: DocumentTranscribeJobRequest,
  ) =>
    apiRequest<DocumentBatchJobResponse>(
      `/projects/${projectId}/documents/${documentId}/jobs/transcribe`,
      { method: "POST", body },
    ),

  listTranscriptions: (projectId: string, documentId: string) =>
    apiRequest<TranscriptionLayerResponse[]>(
      `/projects/${projectId}/documents/${documentId}/transcriptions`,
    ),

  uploadPart: (projectId: string, documentId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts`,
      { method: "POST", body: form },
    );
  },

  beginPartUpload: (
    projectId: string,
    documentId: string,
    body: PartUploadBeginRequest,
  ) =>
    apiRequest<PartUploadBeginResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/upload`,
      { method: "POST", body },
    ),

  finalizePartUpload: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PartUploadFinalizeRequest,
  ) =>
    apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/finalize`,
      { method: "POST", body },
    ),

  reorderParts: (
    projectId: string,
    documentId: string,
    body: ReorderPartsRequest,
  ) =>
    apiRequest<DocumentPartResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/reorder`,
      { method: "PATCH", body },
    ),

  deletePart: (projectId: string, documentId: string, partId: string) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}`,
      { method: "DELETE" },
    ),

  updatePartReviewStatus: (
    projectId: string,
    documentId: string,
    partId: string,
    body: DocumentPartUpdateRequest,
  ) =>
    apiRequest<DocumentPartResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}`,
      { method: "PATCH", body },
    ),

  /**
   * Which pages of a published document the public reader can actually reach.
   *
   * Takes a batch because the endpoint does. The only caller today sends one
   * page per click, deliberately: the response is authoritative for the whole
   * document, so a request carrying rows this tab loaded minutes ago could
   * overwrite a flag someone else has since changed.
   */
  updatePartsPublished: (
    projectId: string,
    documentId: string,
    body: PartsPublishedUpdateRequest,
  ) =>
    apiRequest<DocumentPartResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/published`,
      { method: "PATCH", body },
    ),

  getPartLayout: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout`,
    ),

  listPartLines: (projectId: string, documentId: string, partId: string) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
    ),

  /**
   * A region (ADR 0011). With `assign_contained` (the server default) every
   * unassigned line whose centre lies inside the box joins it, and the page is
   * renumbered region-first: reload the lines after.
   */
  createPartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    body: BlockCreateRequest,
  ) =>
    apiRequest<BlockResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/blocks`,
      { method: "POST", body },
    ),

  patchPartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    blockId: string,
    body: BlockPatchRequest,
  ) =>
    apiRequest<BlockResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/blocks/${blockId}`,
      { method: "PATCH", body },
    ),

  /** The region's lines stay on the page, in no region. */
  deletePartBlock: (
    projectId: string,
    documentId: string,
    partId: string,
    blockId: string,
  ) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/blocks/${blockId}`,
      { method: "DELETE" },
    ),

  createPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LineCreateRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
      { method: "POST", body },
    ),

  replacePartLines: (
    projectId: string,
    documentId: string,
    partId: string,
    body: LinesReplaceRequest,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines`,
      { method: "PUT", body },
    ),

  deletePartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<void>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "DELETE" },
    ),

  patchPartLine: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: LinePatchRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  updateLineGeometry: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
    body: UpdateLineGeometryRequest,
  ) =>
    apiRequest<LineResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  resetPartLayout: (
    projectId: string,
    documentId: string,
    partId: string,
    body: ResetPartLayoutRequest,
  ) =>
    apiRequest<PartLayoutResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/layout/reset`,
      { method: "POST", body },
    ),

  getPagePairing: (projectId: string, documentId: string, partId: string) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/pairing`,
    ),

  importPageTranscription: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PageTranscriptionImportRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/page-transcription`,
      { method: "PUT", body },
    ),

  pairTextLine: (
    projectId: string,
    documentId: string,
    partId: string,
    body: PairTextLineRequest,
  ) =>
    apiRequest<PagePairingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/pairings`,
      { method: "POST", body },
    ),

  generateTranscriptionPdf: (
    projectId: string,
    documentId: string,
    partId: string,
  ) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/transcription-pdf`,
      { method: "POST" },
    ),

  exportPartDocx: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/docx`,
    ),

  getPageXml: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/page-xml`,
    ),

  /** Zip of the PAGE XML next to the full-resolution page image it describes. */
  getPageXmlBundle: (projectId: string, documentId: string, partId: string) =>
    fetchBinaryApi(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/page-xml-bundle`,
    ),

  segmentPart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: SegmentPartRequest,
  ) =>
    apiRequest<EnqueueJobResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment`,
      { method: "POST", body: body ?? {} },
    ),

  getSegmentHealth: (projectId: string, documentId: string, partId: string) =>
    apiRequest<SegmentHealthResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment-health`,
    ),

  // Every apply takes ids and nothing else. The server re-derives the outline
  // from the rows as they stand, so a panel left open while somebody else
  // edited the page cannot write geometry measured against the old one.
  splitSpanningSegment: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment-health/splits`,
      { method: "POST", body: { line_id: lineId } },
    ),

  mergeSegmentFragment: (
    projectId: string,
    documentId: string,
    partId: string,
    primaryId: string,
    fragmentId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment-health/merges`,
      {
        method: "POST",
        body: { primary_id: primaryId, fragment_id: fragmentId },
      },
    ),

  trimSegmentOverlap: (
    projectId: string,
    documentId: string,
    partId: string,
    upperId: string,
    lowerId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment-health/trims`,
      { method: "POST", body: { upper_id: upperId, lower_id: lowerId } },
    ),

  deleteSegmentSuspect: (
    projectId: string,
    documentId: string,
    partId: string,
    lineId: string,
  ) =>
    apiRequest<LineResponse[]>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/segment-health/deletions`,
      { method: "POST", body: { line_id: lineId } },
    ),

  enqueueTranscribePart: (
    projectId: string,
    documentId: string,
    partId: string,
    body?: TranscribePartRequest,
  ) =>
    apiRequest<EnqueueJobResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/transcribe`,
      { method: "POST", body: body ?? {} },
    ),

  listInferenceModels: () =>
    apiRequest<InferenceModelResponse[]>("/inference/models"),

  listProjectModelBindings: (projectId: string) =>
    apiRequest<ModelBindingResponse[]>(`/projects/${projectId}/model-bindings`),

  listDocumentModelBindings: (projectId: string, documentId: string) =>
    apiRequest<ModelBindingResponse[]>(
      `/projects/${projectId}/documents/${documentId}/model-bindings`,
    ),

  resolvePartModelBinding: (
    projectId: string,
    documentId: string,
    partId: string,
    task: InferenceTask,
  ) =>
    apiRequest<ResolvedModelBindingResponse>(
      `/projects/${projectId}/documents/${documentId}/parts/${partId}/model-bindings/resolve?task=${task}`,
    ),

  updateGroundTruthLineText: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    lineId: string,
    body: LineTranscriptionPatchRequest,
  ) =>
    apiRequest<LineTranscriptionResponse>(
      `/projects/${projectId}/documents/${documentId}/transcriptions/${transcriptionId}/lines/${lineId}`,
      { method: "PATCH", body },
    ),

  copyToGroundTruth: (
    projectId: string,
    documentId: string,
    transcriptionId: string,
    body: CopyToGroundTruthRequest,
  ) =>
    apiRequest<CopyToGroundTruthResponse>(
      `/projects/${projectId}/documents/${documentId}/transcriptions/${transcriptionId}/copy-to-ground-truth`,
      { method: "POST", body },
    ),

  getPublicDocument: (
    projectId: string,
    documentId: string,
    token: string | null,
  ) =>
    apiRequest<DocumentWithPartsResponse>(
      withShareToken(
        `/public/projects/${projectId}/documents/${documentId}`,
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
    const path = `/public/projects/${projectId}/documents/${documentId}/layout`;
    const fetchPage = (cursor: string | null) => {
      const query = cursorQuery({ cursor });
      return apiRequest<PublicLayoutResponse>(
        withShareToken(query ? `${path}?${query}` : path, token),
        { skipAuth: true, signal: options.signal },
      );
    };
    // Drain every page: blocks come from the first page only, lines from all of them.
    const first = await fetchPage(null);
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
      const page = await fetchPage(cursor);
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
        `/public/projects/${projectId}/documents/${documentId}/transcriptions`,
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
        `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/transcription-pdf`,
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
        `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/page-xml`,
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
        `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/page-xml-bundle`,
        token,
      ),
      { skipAuth: true },
    ),

  getPublicDocx: (
    projectId: string,
    documentId: string,
    partId: string,
    token: string | null,
  ) =>
    fetchBinaryApi(
      withShareToken(
        `/public/projects/${projectId}/documents/${documentId}/parts/${partId}/docx`,
        token,
      ),
      { skipAuth: true },
    ),

  enqueueTestJob: (body: EnqueueTestJobRequest = { handler: "noop" }) =>
    apiRequest<EnqueueTestJobResponse>("/jobs/test", { method: "POST", body }),

  getJob: (jobId: string) => apiRequest<JobResponse>(`/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    apiRequest<JobResponse>(`/jobs/${jobId}/cancel`, { method: "POST" }),

  /**
   * Remove finished jobs (done / failed / cancelled) from a project's history.
   * Active jobs are left alone by the server.
   */
  clearProjectJobHistory: (projectId: string) =>
    apiRequest<{ deleted: number }>(
      `/jobs/history?project_id=${encodeURIComponent(projectId)}`,
      { method: "DELETE" },
    ),

  listProjectJobsPage: (projectId: string, options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<JobResponse>>(
      `/projects/${projectId}/jobs${query ? `?${query}` : ""}`,
      { signal: options.signal },
    );
  },

  listProjectJobs: (
    projectId: string,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) => api.listProjectJobsPage(projectId, pageOptions),
      options,
    ),
};

export async function waitForJob(
  jobId: string,
  options?: { timeoutMs?: number; onUpdate?: (job: JobResponse) => void },
): Promise<JobResponse> {
  return waitForSubscribedJob(jobId, {
    ...options,
    eventsUrl: `${API_BASE_URL}/jobs/${jobId}/events`,
    token: getAccessToken(),
    getJob: api.getJob,
    intervalMs: JOB_WAIT_POLL_INTERVAL_MS,
  });
}
