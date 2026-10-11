import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

describe("PageEditorPlaceholderPage segment mutations", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await flushPageEditorEffects();
  });

  it("draws a rectangle Segment and saves it as Line geometry for the document part", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText("ANNOTE PAGE WORKSPACE")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /rectangle segment/i }));

    const canvas = screen.getByRole("group", { name: /page geometry canvas/i });
    fireEvent.pointerDown(canvas, { clientX: 20, clientY: 30 });
    fireEvent.pointerMove(canvas, { clientX: 120, clientY: 80 });
    fireEvent.pointerUp(canvas, { clientX: 120, clientY: 80 });

    await waitFor(() => {
      expect(mockedApi.createPartLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        {
          order: 0,
          kind: "rectangle",
          points: [
            [20, 30],
            [120, 30],
            [120, 80],
            [20, 80],
          ],
        },
      );
    });
    expect(mockedApi.replacePartLines).not.toHaveBeenCalled();
    expect(await screen.findByText("1 Segment")).toBeTruthy();
  });
  it("draws a polygon Segment and saves it as Line geometry for the document part", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    expect(await screen.findByText("ANNOTE PAGE WORKSPACE")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /polygon segment/i }));

    const canvas = screen.getByRole("group", { name: /page geometry canvas/i });
    fireEvent.click(canvas, { clientX: 40, clientY: 40 });
    fireEvent.click(canvas, { clientX: 160, clientY: 45 });
    fireEvent.click(canvas, { clientX: 150, clientY: 90 });
    fireEvent.click(canvas, { clientX: 35, clientY: 85 });
    fireEvent.doubleClick(canvas);

    await waitFor(() => {
      expect(mockedApi.createPartLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        {
          order: 0,
          kind: "polygon",
          points: [
            [40, 40],
            [160, 45],
            [150, 90],
            [35, 85],
          ],
        },
      );
    });
    expect(mockedApi.replacePartLines).not.toHaveBeenCalled();
    expect(await screen.findByText("1 Segment")).toBeTruthy();
  });

  it("deletes a selected Segment without replacing every line", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({ kind: "rectangle" }),
      line({
        id: "line-2",
        order: 1,
        points: [
          [80, 20],
          [120, 20],
          [120, 50],
          [80, 50],
        ],
      }),
    ]);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    fireEvent.click(screen.getByRole("button", { name: /delete selected/i }));

    await waitFor(() => {
      expect(mockedApi.deletePartLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-1",
      );
    });
    expect(confirmSpy).toHaveBeenCalled();
    expect(mockedApi.replacePartLines).not.toHaveBeenCalled();
    expect(mockedApi.getPagePairing).toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it("cancels whole-Segment delete when confirmation is declined", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([line({ kind: "rectangle" })]);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    fireEvent.click(screen.getByRole("button", { name: /delete selected/i }));

    await flushPageEditorEffects();
    expect(mockedApi.deletePartLine).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /^Segment 1/ })).toBeTruthy();
    confirmSpy.mockRestore();
  });

  it("removes only the selected vertex with Delete and supports Edit undo", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        points: [
          [10, 10],
          [80, 10],
          [80, 40],
          [10, 40],
        ],
      }),
    ]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    const vertex = await screen.findByLabelText(/Segment vertex 2/);
    fireEvent.pointerDown(vertex, { clientX: 80, clientY: 10, pointerId: 1 });
    fireEvent.pointerUp(vertex, { clientX: 80, clientY: 10, pointerId: 1 });
    fireEvent.mouseUp(window);

    expect(
      await screen.findByLabelText(/Segment vertex 2.*selected/i),
    ).toBeTruthy();

    fireEvent.keyDown(window, { key: "Delete" });

    await waitFor(() => {
      expect(mockedApi.patchPartLine).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-1",
        {
          points: [
            [10, 10],
            [80, 40],
            [10, 40],
          ],
        },
      );
    });
    expect(mockedApi.deletePartLine).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });

    await waitFor(() => {
      expect(mockedApi.patchPartLine).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-1",
        {
          points: [
            [10, 10],
            [80, 10],
            [80, 40],
            [10, 40],
          ],
        },
      );
    });
  });

  it("adds a vertex when clicking a Segment edge", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([
      line({
        points: [
          [0, 0],
          [100, 0],
          [100, 100],
          [0, 100],
        ],
      }),
    ]);

    renderPageEditor();

    const segment = await screen.findByRole("button", { name: /^Segment 1/ });
    fireEvent.click(segment);
    // Midpoint of top edge in image coords; SVG maps client → viewBox via getBoundingClientRect.
    Object.defineProperty(segment, "ownerSVGElement", {
      configurable: true,
      value: {
        getBoundingClientRect: () => ({
          left: 0,
          top: 0,
          width: 640,
          height: 900,
          right: 640,
          bottom: 900,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        }),
      },
    });
    fireEvent.click(segment, { clientX: 50, clientY: 0 });

    await waitFor(() => {
      expect(mockedApi.patchPartLine).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        "line-1",
        {
          points: [
            [0, 0],
            [50, 0],
            [100, 0],
            [100, 100],
            [0, 100],
          ],
        },
      );
    });
  });

  it("Escape commits selection chrome away without deleting the Segment", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([line({ kind: "rectangle" })]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));
    expect(await screen.findByLabelText(/Segment vertex 1/)).toBeTruthy();

    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByLabelText(/Segment vertex 1/)).toBeNull();
    });
    expect(mockedApi.deletePartLine).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /^Segment 1/ })).toBeTruthy();
  });
});
