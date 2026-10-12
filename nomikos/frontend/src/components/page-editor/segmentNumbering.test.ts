import { describe, expect, it } from "vitest";

import {
  segmentNumberFor,
  segmentNumbersById,
  segmentsInNumberOrder,
  withServerOrders,
} from "./segmentNumbering";

function page(...orders: number[]) {
  return orders.map((order) => ({ id: `line-${order}`, order }));
}

describe("segmentNumbering", () => {
  it("numbers the Segments of a Page from 1 in stored order", () => {
    expect(segmentNumbersById(page(0, 1, 2))).toEqual(
      new Map([
        ["line-0", 1],
        ["line-1", 2],
        ["line-2", 3],
      ]),
    );
  });

  it("numbers densely when a delete has left a gap in the orders", () => {
    const lines = page(0, 1, 3);

    expect(segmentNumberFor(lines, "line-3")).toBe(3);
    expect(segmentNumbersById(lines)).toEqual(
      new Map([
        ["line-0", 1],
        ["line-1", 2],
        ["line-3", 3],
      ]),
    );
  });

  it("numbers by stored order, not by the order the API listed", () => {
    const lines = [
      { id: "line-c", order: 3 },
      { id: "line-a", order: 0 },
      { id: "line-b", order: 1 },
    ];

    expect(segmentsInNumberOrder(lines).map((line) => line.id)).toEqual([
      "line-a",
      "line-b",
      "line-c",
    ]);
    expect(segmentNumberFor(lines, "line-b")).toBe(2);
  });

  it("orders colliding orders the same way whatever the input order", () => {
    const collided = [
      { id: "line-b", order: 3 },
      { id: "line-a", order: 3 },
    ];

    expect(segmentsInNumberOrder(collided).map((line) => line.id)).toEqual([
      "line-a",
      "line-b",
    ]);
    expect(
      segmentsInNumberOrder([...collided].reverse()).map((line) => line.id),
    ).toEqual(["line-a", "line-b"]);
  });

  it("leaves the input array untouched", () => {
    const lines = page(2, 0, 1);
    segmentsInNumberOrder(lines);

    expect(lines.map((line) => line.id)).toEqual([
      "line-2",
      "line-0",
      "line-1",
    ]);
  });

  it("has no number for a Segment that is not on the Page", () => {
    expect(segmentNumberFor(page(0, 1), "line-9")).toBeNull();
    expect(segmentNumberFor(page(0, 1), null)).toBeNull();
    expect(segmentNumberFor([], "line-0")).toBeNull();
  });

  it("numbers the only Segment of a single-Segment Page 1", () => {
    expect(segmentNumberFor(page(0), "line-0")).toBe(1);
    expect(segmentNumberFor(page(7), "line-7")).toBe(1);
  });

  it("takes the shifted orders the server returns after an insert", () => {
    const local = [...page(0, 1, 2), { id: "new", order: 1 }];
    const server = [
      { id: "line-0", order: 0 },
      { id: "new", order: 1 },
      { id: "line-1", order: 2 },
      { id: "line-2", order: 3 },
    ];

    const synced = withServerOrders(local, server);

    expect(segmentsInNumberOrder(synced).map((s) => s.id)).toEqual([
      "line-0",
      "new",
      "line-1",
      "line-2",
    ]);
    expect(segmentNumberFor(synced, "line-2")).toBe(4);
  });

  it("leaves a Segment the server does not list, and keeps unchanged ones", () => {
    const local = page(0, 1);
    const synced = withServerOrders(local, [{ id: "line-1", order: 1 }]);

    expect(synced[0]).toBe(local[0]);
    expect(synced[1]).toBe(local[1]);
  });
});
