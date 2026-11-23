import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const publicFonts = join(here, "..", "..", "public", "fonts");
const backendFonts = join(
  here,
  "..",
  "..",
  "..",
  "backend",
  "core",
  "assets",
  "fonts",
);

const backendShipped = readdirSync(backendFonts).filter(
  (name) => name.endsWith(".ttf") || name === "OFL.txt",
);

describe("bundled script fonts", () => {
  it("has backend fonts to compare", () => {
    expect(backendShipped).toContain("OFL.txt");
    expect(
      backendShipped.filter((n) => n.endsWith(".ttf")).length,
    ).toBeGreaterThan(0);
  });

  it("copies every backend font and license into public/fonts", () => {
    const published = new Set(readdirSync(publicFonts));
    for (const name of backendShipped) {
      expect(published.has(name), `${name} missing from public/fonts`).toBe(
        true,
      );
    }
  });

  it("keeps every public font byte-identical to the backend file", () => {
    for (const name of readdirSync(publicFonts)) {
      const mine = readFileSync(join(publicFonts, name));
      const theirs = readFileSync(join(backendFonts, name));
      expect(mine.equals(theirs), `${name} differs from backend copy`).toBe(
        true,
      );
    }
  });

  it("points every @font-face url at an existing public font", () => {
    const css = readFileSync(join(here, "script-fonts.css"), "utf8");
    const urls = [...css.matchAll(/url\("\/fonts\/([^"]+)"\)/g)].map(
      (m) => m[1],
    );
    expect(urls.length).toBeGreaterThan(0);
    const published = new Set(readdirSync(publicFonts));
    for (const file of urls) {
      expect(published.has(file), `${file} not in public/fonts`).toBe(true);
    }
  });
});
