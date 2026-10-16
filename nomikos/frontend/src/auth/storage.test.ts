import { beforeEach, describe, expect, it, vi } from "vitest";

import { queryClient } from "../api/queryClient";
import { invalidateAuthGetCache } from "../api/getCache";
import { clearAccessToken, getAccessToken, setAccessToken } from "./storage";

vi.mock("../api/queryClient", () => ({
  queryClient: { resetQueries: vi.fn() },
}));
vi.mock("../api/getCache", () => ({
  invalidateAuthGetCache: vi.fn(),
}));
vi.mock("../api/imageCache", () => ({
  clearImageCache: vi.fn(),
}));

function tokenWithSub(sub: string): string {
  const payload = btoa(JSON.stringify({ sub }))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  return `header.${payload}.sig`;
}

const resetQueries = vi.mocked(queryClient.resetQueries);
const invalidateGets = vi.mocked(invalidateAuthGetCache);

describe("setAccessToken query reset", () => {
  beforeEach(() => {
    clearAccessToken();
    resetQueries.mockClear();
    invalidateGets.mockClear();
  });

  it("resets on the first token", () => {
    setAccessToken(tokenWithSub("user-1"));
    expect(getAccessToken()).toBe(tokenWithSub("user-1"));
    expect(resetQueries).toHaveBeenCalledTimes(1);
  });

  it("keeps queries on a same-user refresh", () => {
    setAccessToken(tokenWithSub("user-1"));
    resetQueries.mockClear();
    invalidateGets.mockClear();

    setAccessToken(tokenWithSub("user-1"));
    expect(resetQueries).not.toHaveBeenCalled();
    expect(invalidateGets).toHaveBeenCalledTimes(1);
  });

  it("resets when the user changes", () => {
    setAccessToken(tokenWithSub("user-1"));
    resetQueries.mockClear();

    setAccessToken(tokenWithSub("user-2"));
    expect(resetQueries).toHaveBeenCalledTimes(1);
  });

  it("resets for an opaque token", () => {
    setAccessToken(tokenWithSub("user-1"));
    resetQueries.mockClear();

    setAccessToken("granted-token");
    expect(resetQueries).toHaveBeenCalledTimes(1);
  });

  it("forgets the subject on sign-out", () => {
    setAccessToken(tokenWithSub("user-1"));
    clearAccessToken();
    resetQueries.mockClear();

    setAccessToken(tokenWithSub("user-1"));
    expect(resetQueries).toHaveBeenCalledTimes(1);
  });
});
