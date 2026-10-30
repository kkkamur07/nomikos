/**
 * Picking several segments without losing the one already picked.
 *
 * The rules that matter are the ones a click can get wrong: the first
 * additive click has to keep what was selected, an additive click on a member
 * has to remove it, and a group that falls back to one member has to stop
 * being a group. Each of those failing looks like the canvas "forgetting" a
 * segment, which is the report every one of them would produce.
 */
import { describe, expect, it } from "vitest";

import {
  EMPTY_SELECTION,
  inPageOrder,
  isAdditiveClick,
  prunedGroup,
  selectionAfterClick,
  type SegmentSelection,
} from "./segmentSelection";

const plain = { shiftKey: false, metaKey: false, ctrlKey: false };

describe("isAdditiveClick", () => {
  it("treats shift, cmd and ctrl alike", () => {
    expect(isAdditiveClick({ ...plain, shiftKey: true })).toBe(true);
    expect(isAdditiveClick({ ...plain, metaKey: true })).toBe(true);
    expect(isAdditiveClick({ ...plain, ctrlKey: true })).toBe(true);
  });

  it("leaves an unmodified click alone", () => {
    expect(isAdditiveClick(plain)).toBe(false);
  });
});

describe("selectionAfterClick", () => {
  it("selects one segment and drops any group on a plain click", () => {
    const grouped: SegmentSelection = { anchorId: "b", groupIds: ["a", "b"] };
    expect(selectionAfterClick(grouped, "c", false)).toEqual({
      anchorId: "c",
      groupIds: [],
    });
  });

  it("pulls the already-selected segment into the first additive click", () => {
    const one: SegmentSelection = { anchorId: "a", groupIds: [] };
    expect(selectionAfterClick(one, "b", true)).toEqual({
      anchorId: "b",
      groupIds: ["a", "b"],
    });
  });

  it("adds to a group that already exists", () => {
    const two: SegmentSelection = { anchorId: "b", groupIds: ["a", "b"] };
    expect(selectionAfterClick(two, "c", true)).toEqual({
      anchorId: "c",
      groupIds: ["a", "b", "c"],
    });
  });

  it("removes a member that is clicked again", () => {
    const three: SegmentSelection = {
      anchorId: "c",
      groupIds: ["a", "b", "c"],
    };
    expect(selectionAfterClick(three, "b", true)).toEqual({
      anchorId: "c",
      groupIds: ["a", "c"],
    });
  });

  it("stops being a group once only one member is left", () => {
    const two: SegmentSelection = { anchorId: "b", groupIds: ["a", "b"] };
    expect(selectionAfterClick(two, "b", true)).toEqual({
      anchorId: "a",
      groupIds: [],
    });
  });

  it("an additive click with nothing selected is an ordinary selection", () => {
    expect(selectionAfterClick(EMPTY_SELECTION, "a", true)).toEqual({
      anchorId: "a",
      groupIds: [],
    });
  });
});

describe("prunedGroup", () => {
  it("drops ids the page no longer has", () => {
    expect(prunedGroup(["a", "b", "c"], ["a", "c"])).toEqual(["a", "c"]);
  });

  it("collapses a group left with one survivor", () => {
    expect(prunedGroup(["a", "b"], ["a"])).toEqual([]);
  });

  it("returns the same array when nothing went away", () => {
    const group = ["a", "b"];
    expect(prunedGroup(group, ["a", "b", "c"])).toBe(group);
  });
});

describe("inPageOrder", () => {
  it("reads the group back in the order the page has it", () => {
    expect(inPageOrder(["c", "a"], ["a", "b", "c", "d"])).toEqual(["a", "c"]);
  });
});
