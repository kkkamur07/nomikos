import { act, fireEvent, render, screen } from "@testing-library/react";
import { createRef, type ComponentProps, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";

import type { LineResponse, PartLayoutResponse } from "../../api/client";
import { PageEditorCanvas } from "./PageEditorCanvas";
import { DEFAULT_PAGE_EDITOR_SETTINGS } from "./pageEditorSettings";

type TransformState = {
  scale: number;
  positionX: number;
  positionY: number;
};

const capture = vi.hoisted(() => ({
  instance: null as unknown,
  onTransformed: null as ((ref: unknown, state: TransformState) => void) | null,
}));

vi.mock("react-zoom-pan-pinch", async () => {
  const { useEffect } = await import("react");
  function TransformWrapper(props: {
    children: (helpers: { resetTransform: () => void }) => ReactNode;
    onTransformed?: (ref: unknown, state: TransformState) => void;
    ref?: ((instance: unknown) => void) | { current: unknown };
  }) {
    const { onTransformed, ref, children } = props;
    useEffect(() => {
      capture.onTransformed = onTransformed ?? null;
      if (typeof ref === "function") {
        ref(capture.instance);
      } else if (ref) {
        ref.current = capture.instance;
      }
    });
    return <>{children({ resetTransform: () => undefined })}</>;
  }
  function TransformComponent(props: { children?: ReactNode }) {
    return <div>{props.children}</div>;
  }
  return { TransformWrapper, TransformComponent };
});

vi.mock("../AuthenticatedImage", () => ({
  AuthenticatedImage: ({ alt }: { alt: string }) => <img alt={alt} />,
}));

function makeLine(
  id: string,
  order: number,
  pts: [number, number][],
): LineResponse {
  return { id, order, points: pts } as unknown as LineResponse;
}

function fakeViewport() {
  const setTransform = vi.fn();
  const viewport = {
    state: { scale: 1, positionX: 0, positionY: 0, previousScale: 1 },
    instance: {
      transformState: {
        scale: 1,
        positionX: 0,
        positionY: 0,
        previousScale: 1,
      },
      wrapperComponent: null,
    },
    setTransform,
  } as unknown as ReactZoomPanPinchRef;
  return { viewport, setTransform };
}

type CanvasProps = ComponentProps<typeof PageEditorCanvas>;

function baseProps(overrides: Partial<CanvasProps> = {}): CanvasProps {
  return {
    imageUrl: "/media/parts/part-1",
    imageAlt: "Page 0",
    imageWidth: 640,
    imageHeight: 900,
    layout: { blocks: [] } as unknown as PartLayoutResponse,
    lines: [
      makeLine("line-1", 0, [
        [10, 10],
        [50, 10],
        [50, 30],
        [10, 30],
      ]),
      makeLine("line-2", 1, [
        [60, 60],
        [120, 60],
        [120, 90],
        [60, 90],
      ]),
    ],
    selectedSegmentId: null,
    pairedSegmentIds: new Set<string>(),
    drawingRectangle: false,
    drawingPolygon: false,
    draftStart: null,
    draftPolygon: [],
    settings: DEFAULT_PAGE_EDITOR_SETTINGS,
    segmentVertexEditEnabled: false,
    onSelectTool: () => undefined,
    onPickDrawMode: () => undefined,
    canDelete: false,
    onDeleteSelected: () => undefined,
    selectedVertexIndex: null,
    onSelectedVertexChange: () => undefined,
    commitSignal: 0,
    onDraftStart: () => undefined,
    onRectangleDrawn: () => undefined,
    onPolygonPoint: () => undefined,
    onPolygonComplete: () => undefined,
    onSelectLine: () => undefined,
    onSelectSegment: () => undefined,
    onSegmentPointsChange: () => undefined,
    ...overrides,
  };
}

describe("PageEditorCanvas sync", () => {
  it("reports hover enter and leave per segment", () => {
    capture.instance = fakeViewport().viewport;
    const onHoverSegment = vi.fn();
    render(<PageEditorCanvas {...baseProps({ onHoverSegment })} />);

    const first = screen.getByRole("button", { name: /^Segment 1/ });
    fireEvent.mouseOver(first);
    expect(onHoverSegment).toHaveBeenCalledWith("line-1");
    fireEvent.mouseOut(first);
    expect(onHoverSegment).toHaveBeenCalledWith(null);
  });

  it("marks only the hovered segment", () => {
    capture.instance = fakeViewport().viewport;
    const view = render(
      <PageEditorCanvas {...baseProps({ hoveredSegmentId: "line-1" })} />,
    );

    expect(
      screen
        .getByRole("button", { name: /^Segment 1/ })
        .classList.contains("is-hovered"),
    ).toBe(true);
    expect(
      screen
        .getByRole("button", { name: /^Segment 2/ })
        .classList.contains("is-hovered"),
    ).toBe(false);

    view.rerender(
      <PageEditorCanvas {...baseProps({ hoveredSegmentId: "line-2" })} />,
    );
    expect(
      screen
        .getByRole("button", { name: /^Segment 1/ })
        .classList.contains("is-hovered"),
    ).toBe(false);
    expect(
      screen
        .getByRole("button", { name: /^Segment 2/ })
        .classList.contains("is-hovered"),
    ).toBe(true);
  });

  it("exposes the viewport and forwards transform state", () => {
    const { viewport } = fakeViewport();
    capture.instance = viewport;
    const viewportRef = createRef<ReactZoomPanPinchRef>();
    const onTransformed = vi.fn();
    render(<PageEditorCanvas {...baseProps({ viewportRef, onTransformed })} />);

    expect(viewportRef.current).toBe(viewport);
    act(() => {
      capture.onTransformed?.(viewport, {
        scale: 2,
        positionX: 5,
        positionY: 6,
      });
    });
    expect(onTransformed).toHaveBeenCalledWith({
      scale: 2,
      positionX: 5,
      positionY: 6,
    });
  });

  it("centres the focused segment at the current scale, once", () => {
    const { viewport, setTransform } = fakeViewport();
    capture.instance = viewport;
    const view = render(<PageEditorCanvas {...baseProps()} />);
    const host = view.container.querySelector(
      ".pe-canvas-host",
    ) as HTMLElement | null;
    if (!host) throw new Error("canvas host missing");
    Object.defineProperty(host, "clientWidth", {
      value: 640,
      configurable: true,
    });
    Object.defineProperty(host, "clientHeight", {
      value: 900,
      configurable: true,
    });
    expect(setTransform).not.toHaveBeenCalled();

    view.rerender(
      <PageEditorCanvas
        {...baseProps({ focusRequest: { segmentId: "line-1", nonce: 1 } })}
      />,
    );
    expect(setTransform).toHaveBeenCalledTimes(1);
    expect(setTransform).toHaveBeenCalledWith(290, 430, 1, 200);

    view.rerender(
      <PageEditorCanvas
        {...baseProps({ focusRequest: { segmentId: "missing", nonce: 2 } })}
      />,
    );
    expect(setTransform).toHaveBeenCalledTimes(1);
  });

  it("reads the live scale when the ref has no state key", () => {
    const setTransform = vi.fn();
    const viewport = {
      instance: {
        transformState: {
          scale: 2,
          positionX: 0,
          positionY: 0,
          previousScale: 2,
        },
        wrapperComponent: null,
      },
      setTransform,
    } as unknown as ReactZoomPanPinchRef;
    expect("state" in viewport).toBe(false);
    capture.instance = viewport;
    const view = render(<PageEditorCanvas {...baseProps()} />);
    const host = view.container.querySelector(
      ".pe-canvas-host",
    ) as HTMLElement | null;
    if (!host) throw new Error("canvas host missing");
    Object.defineProperty(host, "clientWidth", {
      value: 640,
      configurable: true,
    });
    Object.defineProperty(host, "clientHeight", {
      value: 900,
      configurable: true,
    });

    view.rerender(
      <PageEditorCanvas
        {...baseProps({ focusRequest: { segmentId: "line-1", nonce: 1 } })}
      />,
    );
    expect(setTransform).toHaveBeenCalledTimes(1);
    expect(setTransform).toHaveBeenCalledWith(260, 410, 2, 200);
  });
});
