import { invalidateAuthGetCache } from "../api/getCache";
import { clearImageCache } from "../api/imageCache";
import { queryClient } from "../api/queryClient";

let accessToken: string | null = null;
/** `sub` of the last token `setAccessToken` saw; `null` before the first. */
let lastTokenSubject: string | null = null;

/**
 * The browser session's CSRF token, as the server handed it back in the body.
 *
 * A second copy of what the readable `greekocr-csrf` cookie carries, kept for
 * the case where script cannot read that cookie. The cookie lives on
 * `.nomikos.app` so that `app.nomikos.app` can read a value the API set on
 * `api.nomikos.app`; a browser that declines that sibling-subdomain read
 * leaves the client with no way to build `X-CSRF-Token`, and every
 * `POST /auth/refresh` answers 403. Held in memory rather than in
 * `localStorage`, for the same reason the access token is: a value that
 * survives the tab outlives the session it belongs to.
 */

let csrfToken: string | null = null;
/** The JWT `sub` without verifying: identity comparison only, never trust. */
function tokenSubject(token: string): string | null {
  try {
    const encoded = token.split(".")[1];
    if (!encoded) return null;
    const padded = encoded.replace(/-/g, "+").replace(/_/g, "/");
    const json: unknown = JSON.parse(
      atob(padded + "=".repeat((4 - (padded.length % 4)) % 4)),
    );
    if (json && typeof json === "object" && "sub" in json) {
      return typeof json.sub === "string" ? json.sub : null;
    }
    return null;
  } catch {
    return null;
  }
}

export function getAccessToken(): string | null {
  return accessToken;
}

export function getCsrfToken(): string | null {
  return csrfToken;
}

export function setCsrfToken(token: string): void {
  csrfToken = token;
}

export function clearCsrfToken(): void {
  csrfToken = null;
}

export function setAccessToken(token: string): void {
  // Retained reads belong to whoever was signed in when they were made, so a
  // new session must not be able to see them. Reset rather than clear:
  // clear() destroys in-flight queries and strands their mounted observers in
  // `pending` forever (the public document page wedged on its spinner
  // whenever the AuthProvider's startup refresh settled while its reads were
  // still in flight). resetQueries() drops the data just the same, but
  // refetches every query something on screen is watching, now under the new
  // session. A transparent refresh for the same user keeps its data: only a
  // changed subject - or a token with no readable subject - resets.
  const subject = tokenSubject(token);
  const changed = subject === null || subject !== lastTokenSubject;
  lastTokenSubject = subject;
  accessToken = token;
  invalidateAuthGetCache();
  if (changed) {
    void queryClient.resetQueries();
  }
}

export function clearAccessToken(): void {
  accessToken = null;
  lastTokenSubject = null;
  // Both credentials belong to the one session that just ended. Leaving the
  // CSRF token behind would let the next sign-in start by presenting the
  // previous session's token.
  csrfToken = null;
  // Same reset-not-clear reasoning as `setAccessToken`: an anonymous visitor's
  // failed session restore lands here while the page's first reads are still
  // in flight.
  invalidateAuthGetCache();
  void queryClient.resetQueries();
  clearImageCache();
}
