/**
 * HTTP transport for the platform API - base URL, CSRF, auth recovery.
 *
 * `fetchWithAuthRecovery` is shared by API requests and the event-stream
 * client so each path has the same recovery and sign-out behaviour. It lives
 * here, not in `client.ts`, so `utils/jobSubscription` can use it without an
 * import cycle back into the resource modules.
 */
import { redirectToLogin } from "../auth/session";
import {
  clearCsrfToken,
  getAccessToken,
  getCsrfToken,
  setAccessToken,
  setCsrfToken,
} from "../auth/storage";
import { ApiError, parseApiError } from "./errors";
import { dedupedGet, getAuthRequestVersion } from "./getCache";
import type { TokenResponse } from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";
const CSRF_COOKIE_NAME =
  process.env.NEXT_PUBLIC_CSRF_COOKIE_NAME || "greekocr-csrf";

/**
 * What `fetch` itself accepts as a body, minus the streaming forms the
 * client never sends: a JSON-serialisable object, a multipart form, or a
 * pre-encoded string.
 */
type RequestBody = object | FormData | string;

type RequestOptions = Omit<RequestInit, "body"> & {
  body?: RequestBody;
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
export function rememberCsrfToken(token: TokenResponse): TokenResponse {
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
export async function postCsrfProtected<T>(path: string): Promise<T> {
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
