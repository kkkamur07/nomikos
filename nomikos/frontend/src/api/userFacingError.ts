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

/** Map thrown values to short researcher-facing copy (no stacks). */
export function userFacingMessage(
  error: unknown,
  fallback = "Something went wrong. Try again.",
): string {
  if (error instanceof ApiError) {
    return (
      cleanMessageText(error.message) ??
      STATUS_FALLBACKS[error.status] ??
      fallback
    );
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
