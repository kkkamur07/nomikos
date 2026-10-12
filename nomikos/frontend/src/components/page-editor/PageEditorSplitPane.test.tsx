import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { PageEditorSplitPane } from "./PageEditorSplitPane";

function renderPane(ratio = 0.55, onRatioChange = vi.fn()) {
  return {
    onRatioChange,
    ...render(
      <PageEditorSplitPane
        ratio={ratio}
        onRatioChange={onRatioChange}
        left={<div>manuscript content</div>}
        right={<div>transcript content</div>}
      />,
    ),
  };
}

function mockContainerRect(container: HTMLElement) {
  vi.spyOn(container, "getBoundingClientRect").mockReturnValue({
    left: 0,
    width: 1000,
    top: 0,
    right: 1000,
    bottom: 600,
    height: 600,
    x: 0,
    y: 0,
    toJSON: () => {},
  });
}

function containerOf(renderResult: { container: HTMLElement }): HTMLElement {
  const split = renderResult.container.querySelector(".pe-split");
  if (!(split instanceof HTMLElement)) throw new Error("missing .pe-split");
  return split;
}

describe("PageEditorSplitPane", () => {
  it("renders both children with the left basis from ratio", () => {
    renderPane(0.55);
    expect(screen.getByText("manuscript content")).toBeInTheDocument();
    expect(screen.getByText("transcript content")).toBeInTheDocument();
    const left = document.querySelector(".pe-split-left");
    if (!(left instanceof HTMLElement)) throw new Error("missing left pane");
    expect(left.style.flexBasis).toBe(`${0.55 * 100}%`);
  });

  it("pointer drag calls onRatioChange with the clamped value", () => {
    const { onRatioChange, container } = renderPane();
    const split = containerOf({ container });
    mockContainerRect(split);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");
    fireEvent.pointerDown(handle, { clientX: 600 });
    fireEvent.pointerMove(handle, { clientX: 700 });
    expect(onRatioChange).toHaveBeenCalledWith(0.7);
    fireEvent.pointerUp(handle);
    expect(split.className).not.toContain("is-dragging");
  });

  it("adds is-dragging to the container while dragging", () => {
    const { container } = renderPane();
    const split = containerOf({ container });
    mockContainerRect(split);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");
    fireEvent.pointerDown(handle, { clientX: 600 });
    expect(split.className).toContain("is-dragging");
    fireEvent.pointerUp(handle);
  });

  it("clamps drag values below 0.2 and above 0.8", () => {
    const { onRatioChange, container } = renderPane();
    const split = containerOf({ container });
    mockContainerRect(split);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");
    fireEvent.pointerDown(handle, { clientX: 50 });
    expect(onRatioChange).toHaveBeenCalledWith(0.2);
    fireEvent.pointerMove(handle, { clientX: 950 });
    expect(onRatioChange).toHaveBeenCalledWith(0.8);
    fireEvent.pointerUp(handle);
  });

  it("supports ArrowRight, ArrowLeft, Shift steps, Home and End", () => {
    function Harness() {
      const [ratio, setRatio] = useState(0.5);
      return (
        <PageEditorSplitPane
          ratio={ratio}
          onRatioChange={setRatio}
          left={<div>left</div>}
          right={<div>right</div>}
        />
      );
    }
    const { container } = render(<Harness />);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");

    fireEvent.keyDown(handle, { key: "ArrowRight" });
    expect(handle.getAttribute("aria-valuenow")).toBe("52");

    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(handle.getAttribute("aria-valuenow")).toBe("50");

    fireEvent.keyDown(handle, { key: "ArrowRight", shiftKey: true });
    expect(handle.getAttribute("aria-valuenow")).toBe("60");

    fireEvent.keyDown(handle, { key: "Home" });
    expect(handle.getAttribute("aria-valuenow")).toBe("20");

    fireEvent.keyDown(handle, { key: "End" });
    expect(handle.getAttribute("aria-valuenow")).toBe("80");
  });

  it("double-click resets to 0.5", () => {
    const onRatioChange = vi.fn();
    const { container } = renderPane(0.7, onRatioChange);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");
    fireEvent.doubleClick(handle);
    expect(onRatioChange).toHaveBeenCalledWith(0.5);
  });

  it("aria-valuenow reflects the ratio", () => {
    const { container } = renderPane(0.33);
    const handle = container.querySelector(".pe-split-handle");
    if (!(handle instanceof HTMLElement)) throw new Error("missing handle");
    expect(handle.getAttribute("aria-valuenow")).toBe("33");
    expect(handle.getAttribute("role")).toBe("separator");
    expect(handle.getAttribute("aria-orientation")).toBe("vertical");
    expect(handle.getAttribute("aria-valuemin")).toBe("20");
    expect(handle.getAttribute("aria-valuemax")).toBe("80");
  });

  it("uses default labels for the panes", () => {
    renderPane();
    expect(screen.getByLabelText("Manuscript")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcript")).toBeInTheDocument();
  });
});
