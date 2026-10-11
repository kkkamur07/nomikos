import { describe, expect, it } from "vitest";

import {
  directionOfScript,
  dominantScript,
  scriptOfCodePoint,
  textDirection,
} from "./scriptDirection";

describe("scriptDirection", () => {
  it("names the script of a code point", () => {
    expect(scriptOfCodePoint("ܫ".codePointAt(0) ?? 0)).toBe("syriac");
    expect(scriptOfCodePoint("א".codePointAt(0) ?? 0)).toBe("hebrew");
    expect(scriptOfCodePoint("ع".codePointAt(0) ?? 0)).toBe("arabic");
    expect(scriptOfCodePoint("ἐ".codePointAt(0) ?? 0)).toBe("greek");
    expect(scriptOfCodePoint("ⲁ".codePointAt(0) ?? 0)).toBe("coptic");
    expect(scriptOfCodePoint("Ա".codePointAt(0) ?? 0)).toBe("armenian");
    expect(scriptOfCodePoint("7".codePointAt(0) ?? 0)).toBeNull();
  });

  it("weighs letters to find the dominant script", () => {
    expect(dominantScript("ܫܠܡܐ ܥܠܝܟܘܢ 12 a")).toBe("syriac");
    expect(dominantScript(["καὶ ἐγένετο", "ab"])).toBe("greek");
    expect(dominantScript("12, 13.")).toBeNull();
    expect(dominantScript([])).toBeNull();
  });

  it("reads Syriac, Hebrew and Arabic right to left", () => {
    expect(directionOfScript("syriac")).toBe("rtl");
    expect(directionOfScript("hebrew")).toBe("rtl");
    expect(directionOfScript("arabic")).toBe("rtl");
    expect(directionOfScript("greek")).toBe("ltr");
    expect(textDirection("ܫܠܡܐ ܥܠܝܟܘܢ")).toBe("rtl");
    expect(textDirection("καὶ")).toBe("ltr");
  });

  it("falls back for text without letters", () => {
    expect(textDirection("", "rtl")).toBe("rtl");
    expect(textDirection("123")).toBe("ltr");
  });
});
