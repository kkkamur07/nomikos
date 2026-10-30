import { fireEvent, render, screen } from "@testing-library/react";
import { useState, type ComponentProps } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { LineResponse } from "../../api/client";
import { DEFAULT_PAGE_EDITOR_SETTINGS } from "./pageEditorSettings";

/** Per-line geometry work, counted to see whether a render reached the polygons. */
const geometryCalls = vi.hoisted(() => ({ normalize: 0, points: 0 }));

vi.mock("./canvasGeometry", async (importOriginal) => {
  const real = await importOriginal<typeof import("./canvasGeometry")>();
  return {
    ...real,
    normalizeGeometryPoints: (
      ...args: Parameters<typeof real.normalizeGeometryPoints>
    ) => {
      geometryCalls.normalize += 1;
      return real.normalizeGeometryPoints(...args);
    },
    points: (...args: Parameters<typeof real.points>) => {
      geometryCalls.points += 1;
      return real.points(...args);
    },
  };
});

vi.mock("../AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

const { PageEditorCanvas } = await import("./PageEditorCanvas");

type CanvasProps = ComponentProps<typeof PageEditorCanvas>;

function segment(id: string, order: number): LineResponse {
  return {
    id,
    part_id: "part-1",
    block_id: null,
    order,
    kind: "polygon",
    points: [
      [10, 10 + order],
      [90, 10 + order],
      [90, 40 + order],
      [10, 40 + order],
    ],
    baseline: [
      [10, 30 + order],
      [90, 30 + order],
    ],
    mask: null,
    source: "machine",
    source_metadata: null,
    kraken_ceiling: null,
    manual_geometry: false,
    created_at: "2026-06-16T10:00:00Z",
    line_transcriptions: [],
  };
}

const noop = () => undefined;

function canvasProps(overrides: Partial<CanvasProps> = {}): CanvasProps {
  return {
    imageUrl: "blob:page",
    imageAlt: "page",
    imageWidth: 100,
    imageHeight: 300,
    layout: { blocks: [], lines: [] },
    lines: [segment("S", 0)],
    selectedSegmentId: null,
    selectedSegmentIds: [],
    pairedSegmentIds: new Set<string>(),
    mergeBusy: false,
    mergeError: null,
    onMergeSelected: () => undefined,
    onClearMergeSelection: () => undefined,
    drawingRectangle: false,
    drawingPolygon: false,
    draftStart: null,
    draftPolygon: [],
    settings: DEFAULT_PAGE_EDITOR_SETTINGS,
    segmentVertexEditEnabled: true,
    onSelectTool: noop,
    onPickDrawMode: noop,
    canDelete: false,
    onDeleteSelected: noop,
    selectedVertexIndex: null,
    onSelectedVertexChange: noop,
    commitSignal: 0,
    onDraftStart: noop,
    onRectangleDrawn: noop,
    onPolygonPoint: noop,
    onPolygonComplete: noop,
    onSelectLine: noop,
    onSelectSegment: noop,
    onSegmentPointsChange: noop,
    ...overrides,
  };
}

/**
 * The page editor around the canvas: it owns the selected vertex, and hands
 * in a fresh `onSegmentPointsChange` on every render, as `useLayoutMutations`
 * does.
 */
function EditorHarness({
  commitSignal,
  selections,
  onCommit,
}: {
  commitSignal: number;
  selections: (number | null)[];
  onCommit: (points: LineResponse["points"]) => void;
}) {
  const [vertex, setVertex] = useState<number | null>(null);
  return (
    <PageEditorCanvas
      {...canvasProps({
        selectedSegmentId: "S",
        selectedVertexIndex: vertex,
        commitSignal,
        onSelectedVertexChange: (index) => {
          selections.push(index);
          setVertex(index);
        },
        onSegmentPointsChange: (_segmentId, points) => onCommit(points),
      })}
    />
  );
}

function svgOf(element: Element): SVGSVGElement {
  const svg = element.closest("svg");
  if (!svg) throw new Error("not inside the canvas");
  return svg;
}

beforeEach(() => {
  geometryCalls.normalize = 0;
  geometryCalls.points = 0;
});

describe("PageEditorCanvas vertex editing after Esc", () => {
  it.each([0, 1])(
    "completes a press on a vertex with commitSignal %i",
    (commitSignal) => {
      const selections: (number | null)[] = [];
      render(
        <EditorHarness
          commitSignal={commitSignal}
          selections={selections}
          onCommit={noop}
        />,
      );

      fireEvent.pointerDown(
        screen.getByRole("button", { name: /^Segment vertex 2/ }),
      );
      fireEvent.pointerUp(window);

      expect(selections).toEqual([1, 1]);
    },
  );

  it("drags a vertex once Esc has been pressed", () => {
    const commits: LineResponse["points"][] = [];
    const selections: (number | null)[] = [];
    render(
      <EditorHarness
        commitSignal={1}
        selections={selections}
        onCommit={(points) => commits.push(points)}
      />,
    );
    const handle = screen.getByRole("button", { name: /^Segment vertex 2/ });
    const svg = svgOf(handle);
    vi.spyOn(svg, "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 0, 100, 300),
    );

    fireEvent.pointerDown(handle);
    fireEvent.pointerMove(svg, { clientX: 70, clientY: 60 });
    fireEvent.pointerMove(svg, { clientX: 70, clientY: 60 });
    fireEvent.pointerUp(window);

    expect(commits).toEqual([
      [
        [10, 10],
        [70, 60],
        [90, 40],
        [10, 40],
      ],
    ]);
  });
});

describe("PageEditorCanvas vertex handles from the keyboard", () => {
  it("selects a vertex with Enter instead of removing it", () => {
    const onSelectedVertexChange = vi.fn();
    const onSegmentPointsChange = vi.fn();
    render(
      <PageEditorCanvas
        {...canvasProps({
          selectedSegmentId: "S",
          onSelectedVertexChange,
          onSegmentPointsChange,
        })}
      />,
    );

    fireEvent.keyDown(
      screen.getByRole("button", { name: /^Segment vertex 3/ }),
      { key: "Enter" },
    );

    expect(onSelectedVertexChange).toHaveBeenCalledWith(2);
    expect(onSegmentPointsChange).not.toHaveBeenCalled();
  });
});

describe("PageEditorCanvas re-render cost", () => {
  const LINES = Array.from({ length: 200 }, (_, index) =>
    segment(`L${index}`, index),
  );

  it.each([false, true])(
    "does no per-segment geometry work for a parent re-render with the same data (baselines %s)",
    (showBaselines) => {
      const settings = { ...DEFAULT_PAGE_EDITOR_SETTINGS, showBaselines };
      const props = canvasProps({ lines: LINES, settings });
      const view = render(<PageEditorCanvas {...props} />);
      const before = { ...geometryCalls };

      // Fresh closures, as a parent render hands them in.
      view.rerender(
        <PageEditorCanvas
          {...props}
          onSelectSegment={(id) => noop(id)}
          onSelectLine={(id) => noop(id)}
          onSegmentPointsChange={() => undefined}
          onSelectedVertexChange={() => undefined}
          onDraftStart={() => undefined}
        />,
      );

      expect({
        normalize: geometryCalls.normalize - before.normalize,
        points: geometryCalls.points - before.points,
      }).toEqual({ normalize: 0, points: 0 });
    },
  );

  it("redraws only the dragged segment while a vertex moves", () => {
    render(
      <PageEditorCanvas
        {...canvasProps({ lines: LINES, selectedSegmentId: "L0" })}
      />,
    );
    const handle = screen.getByRole("button", { name: /^Segment vertex 1/ });
    const svg = svgOf(handle);
    vi.spyOn(svg, "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 0, 100, 300),
    );
    fireEvent.pointerDown(handle);
    fireEvent.pointerMove(svg, { clientX: 40, clientY: 50 });
    const before = geometryCalls.points;

    fireEvent.pointerMove(svg, { clientX: 45, clientY: 55 });

    expect(geometryCalls.points - before).toBe(1);
  });
});
