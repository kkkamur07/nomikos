type ValidationErrorItem = {
  loc?: Array<string | number>;
  msg?: string;
};

export class ApiError extends Error {
  readonly status: number;
  readonly ref: string | null;
  /**
   * The parsed `error` object, for the handful of routes whose refusal carries
   * data the caller has to act on rather than only a sentence to show. The
   * merge route's 409 is the first: the client cannot ask "keep whose text?"
   * without the ids it lists. Unknown to every other caller, and `null` when
   * the body was not JSON.
   */
  readonly detail: Record<string, unknown> | null;

  constructor(
    message: string,
    status: number,
    ref: string | null = null,
    detail: Record<string, unknown> | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.ref = ref;
    this.detail = detail;
  }
}

function errorDetail(body: unknown): Record<string, unknown> | null {
  if (!body || typeof body !== "object") return null;
  const error = (body as { error?: unknown }).error;
  if (!error || typeof error !== "object") return null;
  return error as Record<string, unknown>;
}

function formatDetail(body: unknown): string {
  if (!body || typeof body !== "object") {
    return "Request failed";
  }
  const error = (body as { error?: { details?: unknown; message?: unknown } })
    .error;
  if (error && Array.isArray(error.details)) {
    const detail = formatValidationDetails(error.details);
    if (detail) {
      return detail;
    }
  }
  if (error && typeof error.message === "string") {
    return error.message;
  }
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return formatValidationDetails(detail) || "Request failed";
  }
  return "Request failed";
}

function formatValidationDetails(details: unknown[]): string {
  return details
    .map((item) => {
      const err = item as ValidationErrorItem;
      const message = err.msg ?? String(item);
      const location = Array.isArray(err.loc) ? err.loc.join(".") : "";
      return location ? `${location}: ${message}` : message;
    })
    .join("; ");
}

function extractRef(body: unknown, response: Response): string | null {
  if (body && typeof body === "object") {
    const error = (body as { error?: { ref?: unknown } }).error;
    if (error && typeof error.ref === "string" && error.ref.trim()) {
      return error.ref.trim();
    }
  }
  const header = response.headers.get("X-Error-ID");
  return header && header.trim() ? header.trim() : null;
}

/**
 * A request that stopped because its `AbortSignal` fired.
 *
 * Lives here rather than under `inference/`: local-run signals died with the
 * loopback transport (ADR 0002), so what still aborts is a tracked request
 * the researcher cancelled.
 */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export async function parseApiError(response: Response): Promise<ApiError> {
  let message = response.statusText || "Request failed";
  let body: unknown = null;
  try {
    body = await response.json();
    message = formatDetail(body);
  } catch {
    // non-JSON body
  }
  return new ApiError(
    message,
    response.status,
    extractRef(body, response),
    errorDetail(body),
  );
}
