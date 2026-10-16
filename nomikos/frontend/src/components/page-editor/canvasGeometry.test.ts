import { describe, expect, it } from "vitest";

import {
  canvasStrokeWidth,
  cleanPolygonPoints,
  findPolygonEdgeInsert,
  insertPolygonVertexAtClick,
  normalizeGeometryPoints,
  offsetGeometry,
  points,
  removePolygonVertex,
  withGeometryPoints,
} from "./canvasGeometry";

describe("canvasGeometry", () => {
  it("formats plain point arrays for SVG", () => {
    expect(
      points([
        [1, 2],
        [3, 4],
      ]),
    ).toBe("1,2 3,4");
  });

  it("normalizes backend box and baseline objects with points", () => {
    expect(
      points({
        points: [
          [0, 0],
          [10, 0],
          [10, 10],
        ],
      }),
    ).toBe("0,0 10,0 10,10");

    expect(
      normalizeGeometryPoints({
        points: [
          [2, 2],
          [22, 2],
        ],
      }),
    ).toEqual([
      [2, 2],
      [22, 2],
    ]);
  });

  it("normalizes GeoJSON line coordinates", () => {
    expect(
      points({
        type: "LineString",
        coordinates: [
          [1, 1],
          [2, 1],
        ],
      }),
    ).toBe("1,1 2,1");
  });

  it("offsets geometry while preserving object shape", () => {
    expect(
      offsetGeometry(
        {
          points: [
            [2, 2],
            [22, 2],
          ],
        },
        5,
      ),
    ).toEqual({
      points: [
        [2, 7],
        [22, 7],
      ],
    });
  });

  it("writes updated points back into geometry objects", () => {
    expect(
      withGeometryPoints({ type: "LineString", coordinates: [[1, 1]] }, [
        [1, 6],
        [2, 6],
      ]),
    ).toEqual({
      type: "LineString",
      coordinates: [
        [1, 6],
        [2, 6],
      ],
    });
  });

  it("scales stroke width inversely with zoom for screen consistency", () => {
    const zoomedOut = canvasStrokeWidth(2, 0.5, 1.5, 2000);
    const zoomedIn = canvasStrokeWidth(2, 2, 1.5, 2000);
    expect(zoomedIn).toBeLessThan(zoomedOut);
    expect(canvasStrokeWidth(2, 1, 2, 2000)).toBeGreaterThan(
      canvasStrokeWidth(2, 1, 1, 2000),
    );
  });

  it("finds the nearest polygon edge for vertex insertion", () => {
    const square: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    expect(findPolygonEdgeInsert(square, [50, 5], 12)).toEqual({
      insertIndex: 1,
      point: [50, 0],
    });
    expect(findPolygonEdgeInsert(square, [50, 50], 12)).toBeNull();
    expect(findPolygonEdgeInsert(square, [0, 0], 12)).toBeNull();
  });

  it("inserts a vertex on the clicked edge", () => {
    const triangle: [number, number][] = [
      [0, 0],
      [100, 0],
      [50, 80],
    ];
    expect(insertPolygonVertexAtClick(triangle, [50, 2], 12)).toEqual([
      [0, 0],
      [50, 0],
      [100, 0],
      [50, 80],
    ]);
  });

  it("removes consecutive vertices closer than the minimum spacing", () => {
    expect(
      cleanPolygonPoints([
        [0, 0],
        [1, 0],
        [50, 0],
        [100, 0],
        [100, 50],
        [0, 50],
      ]),
    ).toEqual([
      [0, 0],
      [50, 0],
      [100, 0],
      [100, 50],
      [0, 50],
    ]);
  });

  it("drops a last vertex that closes back onto the first", () => {
    expect(
      cleanPolygonPoints([
        [0, 0],
        [100, 0],
        [100, 50],
        [1, 1],
      ]),
    ).toEqual([
      [0, 0],
      [100, 0],
      [100, 50],
    ]);
  });

  it("drops a closing vertex that only shows up after the repeats go", () => {
    // [4, 0] is clear of the first vertex, so it survives the first check;
    // it is a repeat of [2, 0], which then closes onto [0, 0].
    expect(
      cleanPolygonPoints([
        [0, 0],
        [100, 0],
        [100, 50],
        [2, 0],
        [4, 0],
      ]),
    ).toEqual([
      [0, 0],
      [100, 0],
      [100, 50],
    ]);
  });

  it("keeps the polygon as given when cleaning would leave too few vertices", () => {
    const sliver: [number, number][] = [
      [0, 0],
      [1, 0],
      [50, 0],
      [0, 1],
    ];
    expect(cleanPolygonPoints(sliver)).toBe(sliver);
    const pair: [number, number][] = [
      [0, 0],
      [1, 1],
    ];
    expect(cleanPolygonPoints(pair)).toBe(pair);
  });

  it("returns copies, never the caller's vertices", () => {
    const square: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    const cleaned = cleanPolygonPoints(square);
    expect(cleaned).toEqual(square);
    expect(cleaned[0]).not.toBe(square[0]);
  });

  it("removes a polygon vertex while keeping at least three points", () => {
    const square: [number, number][] = [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ];
    const triangle: [number, number][] = [
      [0, 0],
      [100, 0],
      [50, 80],
    ];
    expect(removePolygonVertex(square, 1)).toEqual([
      [0, 0],
      [100, 100],
      [0, 100],
    ]);
    expect(removePolygonVertex(triangle, 0)).toBeNull();
  });
});
