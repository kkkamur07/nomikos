import { describe, expect, it } from "vitest";

import { regionName, regionSeparators, regionsInOrder } from "./regions";

const first = { id: "a", order: 0, label: "4v" };
const second = { id: "b", order: 1, label: null };

describe("regions", () => {
  it("names a region by its label, else by its position", () => {
    expect(regionName(first, [second, first])).toBe("4v");
    expect(regionName(second, [second, first])).toBe("Region 2");
    expect(regionsInOrder([second, first]).map((r) => r.id)).toEqual([
      "a",
      "b",
    ]);
  });

  it("puts one separator before each region's first segment and one before the unassigned", () => {
    const separators = regionSeparators(
      [
        { id: "1", block_id: "a" },
        { id: "2", block_id: "a" },
        { id: "3", block_id: "b" },
        { id: "4", block_id: null },
        { id: "5", block_id: "gone" },
      ],
      [first, second],
    );
    expect([...separators]).toEqual([
      ["1", "4v"],
      ["3", "Region 2"],
      ["4", "Unassigned"],
    ]);
  });

  it("adds nothing on a page without regions", () => {
    expect(regionSeparators([{ id: "1", block_id: null }], []).size).toBe(0);
  });
});
