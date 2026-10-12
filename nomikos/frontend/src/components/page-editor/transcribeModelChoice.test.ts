import { describe, expect, it } from "vitest";

import { resolveTranscribeModelId } from "./transcribeModelChoice";

const SYRIAC = { id: "htr-syriac", name: "syriac" };
const GREEK = { id: "htr-greek", name: "greek" };

describe("resolveTranscribeModelId", () => {
  it("keeps the explicit choice when that id still exists", () => {
    expect(resolveTranscribeModelId([SYRIAC, GREEK], "htr-greek")).toBe(
      "htr-greek",
    );
  });

  it("prefers the binding over the first catalog row", () => {
    expect(resolveTranscribeModelId([SYRIAC, GREEK], null, "htr-greek")).toBe(
      "htr-greek",
    );
  });

  it("prefers the explicit choice over the binding", () => {
    expect(
      resolveTranscribeModelId([SYRIAC, GREEK], "htr-syriac", "htr-greek"),
    ).toBe("htr-syriac");
  });

  it("ignores ids that left the catalog and falls back to the first row", () => {
    expect(
      resolveTranscribeModelId([SYRIAC, GREEK], "htr-retired", "htr-gone"),
    ).toBe("htr-syriac");
  });

  it("falls back to the first model when nothing else applies", () => {
    expect(resolveTranscribeModelId([SYRIAC, GREEK])).toBe("htr-syriac");
  });

  it("returns null for an empty list", () => {
    expect(resolveTranscribeModelId([])).toBeNull();
  });
});
