import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { securityHeaders } from "../securityHeaders";

type Header = { key: string; value: string };

/**
 * The security headers live in two places: `securityHeaders.ts` (Docker
 * `node server.js`, via next.config.ts) and `vercel.json` (edge). Neither
 * reads the other, so this test is what keeps them identical.
 */
describe("security headers", () => {
  it("sends the same headers from next.config.ts and vercel.json", () => {
    const fromConfig = Object.fromEntries(
      securityHeaders.map((header) => [header.key, header.value]),
    );

    const vercel = JSON.parse(
      readFileSync(join(process.cwd(), "vercel.json"), "utf8"),
    ) as {
      headers: { headers: Header[] }[];
    };
    const fromVercel = Object.fromEntries(
      vercel.headers[0].headers.map((header) => [header.key, header.value]),
    );

    expect(fromConfig).toEqual(fromVercel);
  });
});
