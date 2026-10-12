import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiRequest } from "../api/client";
import { AuthProvider, useAuthSession } from "./AuthProvider";
import { clearAccessToken, getAccessToken, setAccessToken } from "./storage";

function SessionStatus() {
  const { status } = useAuthSession();
  return <output>{status}</output>;
}

/** Renders the status and exposes the two actions every sign-in and sign-out uses. */
function SessionControls({
  onLogoutError,
}: {
  onLogoutError?: (error: unknown) => void;
}) {
  const { status, establish, logout } = useAuthSession();
  return (
    <>
      <output>{status}</output>
      <button onClick={() => establish("granted-token")}>establish</button>
      {/* `logout` clears in a `finally` but does not swallow: a failing `POST
          /auth/logout` still rejects out to the caller, so a caller that ignores the
          promise leaks an unhandled rejection. Catching here is what a real caller
          must do, and the failure test asserts the clearing happened anyway. */}
      <button onClick={() => void logout().catch(onLogoutError)}>logout</button>
    </>
  );
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("AuthProvider refresh recovery", () => {
  beforeEach(() => {
    clearAccessToken();
  });

  it("shares bootstrap refresh with a protected request", async () => {
    let resolveRefresh: ((response: Response) => void) | undefined;
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/auth/refresh")) {
        return new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        });
      }
      const token = new Headers(init?.headers).get("Authorization");
      return Promise.resolve(
        token === "Bearer restored-token"
          ? jsonResponse({ id: "project-1" })
          : new Response(null, { status: 401 }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AuthProvider>
        <SessionStatus />
      </AuthProvider>,
    );
    expect(screen.getByText("restoring")).toBeInTheDocument();
    const request = apiRequest<{ id: string }>("/projects/project-1");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    resolveRefresh?.(jsonResponse({ access_token: "restored-token" }));

    await expect(request).resolves.toEqual({ id: "project-1" });
    await waitFor(() =>
      expect(screen.getByText("authenticated")).toBeInTheDocument(),
    );
    expect(
      fetchMock.mock.calls.filter(([url]) => url.endsWith("/auth/refresh")),
    ).toHaveLength(1);
    expect(getAccessToken()).toBe("restored-token");
  });

  it("stays restoring until refresh fails, then becomes anonymous", async () => {
    const fetchMock = vi.fn((url: string) => {
      if (url.endsWith("/auth/refresh")) {
        return Promise.resolve(new Response(null, { status: 401 }));
      }
      return Promise.resolve(new Response(null, { status: 500 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AuthProvider>
        <SessionStatus />
      </AuthProvider>,
    );
    expect(screen.getByText("restoring")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("anonymous")).toBeInTheDocument(),
    );
  });
});

// `establish` and `logout` are the whole of sign-in and sign-out. Until these existed,
// both could be replaced with empty bodies without failing a single test in the repo,
// because every consumer stubbed `useAuthSession` instead of rendering the provider.
describe("AuthProvider session lifecycle", () => {
  beforeEach(() => {
    clearAccessToken();
  });

  async function renderControls(onLogoutError?: (error: unknown) => void) {
    render(
      <AuthProvider>
        <SessionControls onLogoutError={onLogoutError} />
      </AuthProvider>,
    );
    // Let the bootstrap refresh settle so the assertions below are about the action
    // under test rather than about a race with it.
    await waitFor(() =>
      expect(screen.getByText("anonymous")).toBeInTheDocument(),
    );
  }

  it("establish stores the token and authenticates the session", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        // The admin lookup that follows establish, answered as a live token would be.
        if (url.endsWith("/me")) return jsonResponse({ is_admin: false });
        return new Response(null, { status: 401 });
      }),
    );
    await renderControls();

    screen.getByRole("button", { name: "establish" }).click();

    await waitFor(() =>
      expect(screen.getByText("authenticated")).toBeInTheDocument(),
    );
    // The token is what every later request is signed with; a status flip alone is a
    // logged-in-looking UI over an unauthenticated client.
    expect(getAccessToken()).toBe("granted-token");
  });

  it("logout calls the API, clears the token, and ends anonymous", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/auth/refresh"))
        return new Response(null, { status: 401 });
      if (url.endsWith("/auth/logout"))
        return new Response(null, { status: 204 });
      throw new Error(`unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    await renderControls();

    setAccessToken("live-token");
    screen.getByRole("button", { name: "logout" }).click();

    await waitFor(() => expect(getAccessToken()).toBeNull());
    expect(screen.getByText("anonymous")).toBeInTheDocument();
    // The server must be told, or the refresh cookie outlives the sign-out and the
    // next page load silently restores the session the user just ended.
    expect(
      fetchMock.mock.calls.filter(([url]) =>
        (typeof url === "string" ? url : url.toString()).endsWith(
          "/auth/logout",
        ),
      ),
    ).toHaveLength(1);
  });

  it("logout still clears the token when the API call fails", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/auth/refresh"))
        return new Response(null, { status: 401 });
      throw new Error("network down");
    });
    vi.stubGlobal("fetch", fetchMock);
    const logoutError = vi.fn();
    await renderControls(logoutError);

    setAccessToken("live-token");
    screen.getByRole("button", { name: "logout" }).click();

    // A sign-out that leaves a live JWT in memory because the network blipped is the
    // worst of both worlds: the user believes they are out, the tab is still armed.
    await waitFor(() => expect(getAccessToken()).toBeNull());
    expect(screen.getByText("anonymous")).toBeInTheDocument();
    // ...and the failure is still reported rather than silently swallowed.
    expect(logoutError).toHaveBeenCalled();
  });
});

function AdminFlag() {
  const { status, isAdmin, establish } = useAuthSession();
  return (
    <>
      <output>{status}</output>
      <output aria-label="is admin">{String(isAdmin)}</output>
      <button onClick={() => establish("granted-token")}>establish</button>
    </>
  );
}

// A failed `/me` is not an answer: reading it as "not an admin" sent a real admin
// to the not-found page for the rest of the session.
describe("AuthProvider admin lookup", () => {
  beforeEach(() => {
    clearAccessToken();
  });

  function meCalls(fetchMock: ReturnType<typeof vi.fn>) {
    return fetchMock.mock.calls.filter(([url]) =>
      (typeof url === "string" ? url : url.toString()).endsWith("/me"),
    );
  }

  async function establishWith(fetchMock: ReturnType<typeof vi.fn>) {
    vi.stubGlobal("fetch", fetchMock);
    render(
      <AuthProvider>
        <AdminFlag />
      </AuthProvider>,
    );
    await waitFor(() =>
      expect(screen.getByText("anonymous")).toBeInTheDocument(),
    );
    screen.getByRole("button", { name: "establish" }).click();
  }

  it("retries a failed /me once and then knows the account is an admin", async () => {
    let meAttempts = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/me")) {
        meAttempts += 1;
        return meAttempts === 1
          ? new Response(null, { status: 503 })
          : jsonResponse({ id: "u-1", is_admin: true });
      }
      return new Response(null, { status: 401 });
    });
    await establishWith(fetchMock);

    await waitFor(() =>
      expect(screen.getByLabelText("is admin")).toHaveTextContent("true"),
    );
    expect(meCalls(fetchMock)).toHaveLength(2);
  });

  it("stays unknown, not false, when /me keeps failing", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/me")) return new Response(null, { status: 503 });
      return new Response(null, { status: 401 });
    });
    await establishWith(fetchMock);

    await waitFor(() => expect(meCalls(fetchMock)).toHaveLength(2));
    // Let the second failure settle before reading the flag.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByLabelText("is admin")).toHaveTextContent("null");
  });
});
