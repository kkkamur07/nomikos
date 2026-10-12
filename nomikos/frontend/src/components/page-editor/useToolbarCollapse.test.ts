import { describe, expect, it } from "vitest";

import {
  nextCollapseLevel,
  type ToolbarCollapseLevel,
} from "./useToolbarCollapse";

type Needed = Partial<Record<ToolbarCollapseLevel, number>>;

describe("nextCollapseLevel", () => {
  it("folds one level at a time while the bar overflows, then stops", () => {
    const needed: Needed = {};
    expect(
      nextCollapseLevel(0, { available: 1000, required: 1200 }, needed),
    ).toBe(1);
    expect(needed[0]).toBe(1200);
    expect(
      nextCollapseLevel(1, { available: 1000, required: 1080 }, needed),
    ).toBe(2);
    expect(needed[1]).toBe(1080);
    // The last level is the floor, even if the bar still overflows.
    expect(
      nextCollapseLevel(2, { available: 700, required: 760 }, needed),
    ).toBe(2);
  });

  it("stays put when the content fits and the wider level would not", () => {
    const needed: Needed = { 0: 1200, 1: 1080 };
    expect(
      nextCollapseLevel(2, { available: 1050, required: 1050 }, needed),
    ).toBe(2);
    expect(
      nextCollapseLevel(1, { available: 1100, required: 1100 }, needed),
    ).toBe(1);
  });

  it("unfolds once the width the wider level needed is back", () => {
    const needed: Needed = { 0: 1200, 1: 1080 };
    expect(
      nextCollapseLevel(2, { available: 1080, required: 1080 }, needed),
    ).toBe(1);
    expect(
      nextCollapseLevel(1, { available: 1300, required: 1300 }, needed),
    ).toBe(0);
  });

  it("does not flicker: an unfold that overflows records the real need", () => {
    const needed: Needed = { 0: 1200 };
    // Unfold to 0 at 1250 px, but the title grew and 0 now needs 1320 px.
    expect(
      nextCollapseLevel(0, { available: 1250, required: 1320 }, needed),
    ).toBe(1);
    expect(needed[0]).toBe(1320);
    expect(
      nextCollapseLevel(1, { available: 1250, required: 1250 }, needed),
    ).toBe(1);
  });

  it("ignores a bar that has not been laid out yet", () => {
    expect(nextCollapseLevel(1, { available: 0, required: 0 }, {})).toBe(1);
  });
});
