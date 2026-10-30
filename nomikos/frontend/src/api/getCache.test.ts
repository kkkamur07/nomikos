import { describe, expect, it, vi } from "vitest";
import {
  dedupedGet,
  getAuthRequestVersion,
  invalidateAuthGetCache,
} from "./getCache";

describe("dedupedGet", () => {
  it("shares one in-flight request for the same key", async () => {
    let release!: (value: string) => void;
    const gate = new Promise<string>((resolve) => {
      release = resolve;
    });
    const fetcher = vi.fn(() => gate);

    const pending = Promise.all([
      dedupedGet("GET /example", fetcher),
      dedupedGet("GET /example", fetcher),
    ]);
    expect(fetcher).toHaveBeenCalledTimes(1);
    release("ok");

    const [first, second] = await pending;
    expect(first).toBe("ok");
    expect(second).toBe("ok");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("allows a new request after the previous one settles", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce("first")
      .mockResolvedValueOnce("second");

    expect(await dedupedGet("GET /example", fetcher)).toBe("first");
    expect(await dedupedGet("GET /example", fetcher)).toBe("second");
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("does not share in-flight GETs across auth-session changes", async () => {
    let releaseFirst: (value: string) => void;
    const first = dedupedGet(
      "GET /projects",
      () =>
        new Promise<string>((resolve) => {
          releaseFirst = resolve;
        }),
    );

    const before = getAuthRequestVersion();
    invalidateAuthGetCache();
    expect(getAuthRequestVersion()).toBe(before + 1);

    const secondFetcher = vi.fn().mockResolvedValue("second-session");
    const second = dedupedGet("GET /projects", secondFetcher);
    releaseFirst!("first-session");

    await expect(first).resolves.toBe("first-session");
    await expect(second).resolves.toBe("second-session");
    expect(secondFetcher).toHaveBeenCalledTimes(1);
  });
});
