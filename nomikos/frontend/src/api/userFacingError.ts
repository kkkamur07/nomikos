import { ApiError } from "./errors";

const STATUS_FALLBACKS: Record<number, string> = {
  400: "That request was invalid. Check your input and try again.",
  401: "Your session expired. Sign in again to continue.",
  403: "You do not have permission to do that.",
  404: "We could not find that resource.",
  409: "That conflicts with the current state. Refresh and try again.",
  422: "Some of the submitted values are invalid.",
  429: "Too many requests. Wait a moment and try again.",
  500: "Something went wrong on the server. Try again shortly.",
  503: "The service is temporarily unavailable. Try again shortly.",
};

function looksLikeStackOrNoise(message: string): boolean {
  return (
    /traceback|stack trace|exception in|at \S+\.\w+\(|File ".*", line \d+/i.test(
      message,
    ) || message.trim().length > 280
  );
}

function cleanMessageText(text: string | undefined): string | null {
  const raw = text?.trim();
  if (raw && !looksLikeStackOrNoise(raw)) {
    return raw;
  }
  return null;
}

const NETWORK_FAILURE =
  "Could not reach the server. Check your connection and try again.";

/** How much of the ref to show: enough to grep the server log, short enough to read out. */
const SHORT_REF_LENGTH = 8;

function refSuffix(error: ApiError): string {
  return error.ref ? ` (ref ${error.ref.slice(0, SHORT_REF_LENGTH)})` : "";
}

/**
 * The server's one-line condensed reason on a 500 (`error.summary`), already
 * stripped of SQL, parameters and paths server side. Re-checked here anyway.
 */
function serverSummary(error: ApiError): string | null {
  const summary = error.detail?.summary;
  return typeof summary === "string" ? cleanMessageText(summary) : null;
}

function apiErrorMessage(error: ApiError, fallback: string): string {
  const base =
    cleanMessageText(error.message) ??
    STATUS_FALLBACKS[error.status] ??
    fallback;
  if (error.status < 500) {
    return base;
  }
  // A 5xx is only useful to report if the person can say which one: the ref
  // finds the server log line, the summary says what broke.
  const summary = serverSummary(error);
  return summary
    ? `Server error: ${summary}${refSuffix(error)}`
    : `${base}${refSuffix(error)}`;
}

/** Map thrown values to short researcher-facing copy (no stacks). */
export function userFacingMessage(
  error: unknown,
  fallback = "Something went wrong. Try again.",
): string {
  if (error instanceof ApiError) {
    return apiErrorMessage(error, fallback);
  }
  // `fetch` rejects with a bare TypeError when the request never got an HTTP
  // answer the page may read: offline, DNS, or a response without CORS headers.
  if (error instanceof TypeError) {
    return NETWORK_FAILURE;
  }
  if (error instanceof Error) {
    return cleanMessageText(error.message) ?? fallback;
  }
  return (
    cleanMessageText(typeof error === "string" ? error : undefined) ?? fallback
  );
}

export function errorRef(error: unknown): string | null {
  if (error instanceof ApiError && error.ref) {
    return error.ref;
  }
  return null;
}
