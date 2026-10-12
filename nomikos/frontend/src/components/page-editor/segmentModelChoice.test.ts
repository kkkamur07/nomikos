import { describe, expect, it } from "vitest";

import {
  resolveSegmentModelId,
  segmentRegistryIdOf,
} from "./segmentModelChoice";

const KRAKEN = {
  id: "seg-kraken",
  name: "kraken",
  artifact_ref: "registry://blla-segment?tag=stable",
};

const PPOCR = {
  id: "seg-ppocr",
  name: "ppocr",
  artifact_ref: "registry://ppocr-segment?tag=stable",
};

describe("resolveSegmentModelId", () => {
  it("preselects by artifact_ref even though the display name is kraken", () => {
    expect(resolveSegmentModelId([PPOCR, KRAKEN])).toBe("seg-kraken");
  });

  it("keeps the persisted choice when that id still exists", () => {
    expect(resolveSegmentModelId([KRAKEN, PPOCR], "seg-ppocr")).toBe(
      "seg-ppocr",
    );
  });

  it("ignores a persisted choice whose id is gone and falls back to the canonical row", () => {
    expect(resolveSegmentModelId([PPOCR, KRAKEN], "seg-retired")).toBe(
      "seg-kraken",
    );
  });

  it("falls back to the first model when no row carries the canonical registry id", () => {
    expect(resolveSegmentModelId([PPOCR])).toBe("seg-ppocr");
  });

  it("returns null for an empty list", () => {
    expect(resolveSegmentModelId([])).toBeNull();
  });

  it("reads the registry id from artifact_ref, never the display name", () => {
    expect(segmentRegistryIdOf(KRAKEN)).toBe("blla-segment");
    expect(segmentRegistryIdOf(PPOCR)).toBe("ppocr-segment");
    expect(
      segmentRegistryIdOf({ id: "x", artifact_ref: "not-a-ref" }),
    ).toBeNull();
  });
});
