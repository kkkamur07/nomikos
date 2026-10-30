import { API_BASE_URL } from "./client";
import { errorRef, userFacingMessage } from "./userFacingError";

type ClientFailurePayload = {
  message: string;
  ref?: string | null;
  path?: string;
  status?: number | null;
  source?: string;
};

function payloadFromError(
  error: unknown,
  source: string,
): ClientFailurePayload {
  const message = userFacingMessage(error);
  const ref = errorRef(error);
  const status =
    error && typeof error === "object" && "status" in error
      ? Number((error as { status?: unknown }).status) || null
      : null;
  return {
    message,
    ref,
    status,
    source,
    path: typeof window !== "undefined" ? window.location.pathname : undefined,
  };
}

/** Fire-and-forget UI failure beacon for logging-first observability. */
export function reportClientFailure(
  error: unknown,
  source = "ui",
): ClientFailurePayload {
  const payload = payloadFromError(error, source);
  const body = JSON.stringify(payload);
  const url = `${API_BASE_URL}/client-failures`;

  try {
    if (
      typeof navigator !== "undefined" &&
      typeof navigator.sendBeacon === "function"
    ) {
      const blob = new Blob([body], { type: "application/json" });
      if (navigator.sendBeacon(url, blob)) {
        return payload;
      }
    }
  } catch {
    // fall through to fetch
  }

  try {
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      credentials: "include",
      keepalive: true,
    }).catch((err) => {
      if (typeof console !== "undefined") {
        console.warn("client_failure_beacon_failed", payload, err);
      }
    });
  } catch {
    if (typeof console !== "undefined") {
      console.warn("client_failure_beacon_failed", payload);
    }
  }
  return payload;
}
