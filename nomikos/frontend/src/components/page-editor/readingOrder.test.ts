import { describe, expect, it } from "vitest";

import type { LinePoint } from "../../api/client";

import {
  nextInReadingOrder,
  readingOrder,
  segmentOrigin,
  type OrderableSegment,
} from "./readingOrder";

function line(
  id: string,
  baseline?: LinePoint[],
  extra?: Partial<OrderableSegment>,
): OrderableSegment {
  return { id, ...(baseline ? { baseline } : {}), ...extra };
}

function ids(segments: OrderableSegment[]): string[] {
  return segments.map((segment) => segment.id);
}

describe("readingOrder", () => {
  it("reads a single shuffled column top to bottom", () => {
    const segments = [
      line("c", [
        [100, 300],
        [500, 300],
      ]),
      line("a", [
        [100, 100],
        [500, 100],
      ]),
      line("b", [
        [100, 200],
        [500, 200],
      ]),
    ];

    expect(ids(readingOrder(segments))).toEqual(["a", "b", "c"]);
  });

  it("reads a two-folio spread one full column before the other", () => {
    const left = [100, 200, 300].map((y) =>
      line(`left-${y}`, [
        [100, y],
        [500, y],
      ]),
    );
    const right = [100, 200, 300].map((y) =>
      line(`right-${y}`, [
        [1100, y],
        [1500, y],
      ]),
    );
    const shuffled = [right[2], left[1], right[0], left[2], left[0], right[1]];

    expect(ids(readingOrder(shuffled))).toEqual([
      "left-100",
      "left-200",
      "left-300",
      "right-100",
      "right-200",
      "right-300",
    ]);
  });

  it("reads the right column first for rtl scripts", () => {
    const left = [100, 200].map((y) =>
      line(`left-${y}`, [
        [100, y],
        [500, y],
      ]),
    );
    const right = [100, 200].map((y) =>
      line(`right-${y}`, [
        [1100, y],
        [1500, y],
      ]),
    );

    expect(
      ids(readingOrder([...left, ...right], { direction: "rtl" })),
    ).toEqual(["right-100", "right-200", "left-100", "left-200"]);
  });

  it("places a diagonal marginal note by its origin, not its centre", () => {
    const note = line("note", [
      [400, 310],
      [100, 100],
    ]);
    const a = line("a", [
      [100, 150],
      [500, 150],
    ]);
    const b = line("b", [
      [100, 250],
      [500, 250],
    ]);

    expect(segmentOrigin(note, "ltr")).toEqual([100, 100]);
    expect(ids(readingOrder([a, b, note]))).toEqual(["note", "a", "b"]);
  });

  it("orders segments that have only points and no baseline", () => {
    const segments = [
      line("lower", undefined, {
        points: [
          [100, 300],
          [500, 300],
          [500, 320],
          [100, 320],
        ],
      }),
      line("upper", undefined, {
        points: [
          [100, 100],
          [500, 100],
          [500, 120],
          [100, 120],
        ],
      }),
    ];

    expect(ids(readingOrder(segments))).toEqual(["upper", "lower"]);
  });

  it("sends segments without geometry last, in id order", () => {
    const segments = [
      { id: "ghost-b" },
      line("b", [
        [100, 200],
        [500, 200],
      ]),
      { id: "ghost-a" },
      line("a", [
        [100, 100],
        [500, 100],
      ]),
    ];

    expect(ids(readingOrder(segments))).toEqual([
      "a",
      "b",
      "ghost-a",
      "ghost-b",
    ]);
  });

  it("mirrors the origin for rtl scripts", () => {
    const segment = line("a", [
      [100, 100],
      [500, 120],
    ]);

    expect(segmentOrigin(segment, "ltr")).toEqual([100, 100]);
    expect(segmentOrigin(segment, "rtl")).toEqual([500, 120]);
  });

  it("contains every input exactly once", () => {
    const segments = [
      line("c", [
        [1100, 100],
        [1500, 100],
      ]),
      { id: "ghost" },
      line("a", [
        [100, 100],
        [500, 100],
      ]),
      line("b", [
        [100, 200],
        [500, 200],
      ]),
    ];

    const ordered = readingOrder(segments);
    expect(ordered).toHaveLength(segments.length);
    expect([...ids(ordered)].sort()).toEqual(["a", "b", "c", "ghost"]);
  });

  it("leaves the input array untouched", () => {
    const segments = [
      line("b", [
        [100, 200],
        [500, 200],
      ]),
      line("a", [
        [100, 100],
        [500, 100],
      ]),
    ];
    readingOrder(segments);

    expect(ids(segments)).toEqual(["b", "a"]);
  });
});

describe("nextInReadingOrder", () => {
  const segments = [
    line("b", [
      [100, 200],
      [500, 200],
    ]),
    line("a", [
      [100, 100],
      [500, 100],
    ]),
    line("c", [
      [100, 300],
      [500, 300],
    ]),
  ];

  it("steps forward and backward through the order", () => {
    expect(nextInReadingOrder(segments, "a", 1)?.id).toBe("b");
    expect(nextInReadingOrder(segments, "b", 1)?.id).toBe("c");
    expect(nextInReadingOrder(segments, "c", -1)?.id).toBe("b");
    expect(nextInReadingOrder(segments, "b", -1)?.id).toBe("a");
  });

  it("returns null at the ends instead of wrapping", () => {
    expect(nextInReadingOrder(segments, "a", -1)).toBeNull();
    expect(nextInReadingOrder(segments, "c", 1)).toBeNull();
  });

  it("starts at the first or last segment for a null current id", () => {
    expect(nextInReadingOrder(segments, null, 1)?.id).toBe("a");
    expect(nextInReadingOrder(segments, null, -1)?.id).toBe("c");
  });

  it("returns null for an unknown id or an empty page", () => {
    expect(nextInReadingOrder(segments, "missing", 1)).toBeNull();
    expect(nextInReadingOrder([], null, 1)).toBeNull();
  });
});
