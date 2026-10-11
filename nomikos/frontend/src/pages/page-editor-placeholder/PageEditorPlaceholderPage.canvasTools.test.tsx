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

/**
 * The editor used to carry a Layout/Transcription toggle that gated vertex
 * editing and the draw shortcuts, and a Select button whose active state was
 * tied to that toggle rather than to the armed tool. These cover what replaced
 * it: one tool state, shown honestly, with Space as the escape hatch.
 */
describe("PageEditorPlaceholderPage canvas tools", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await flushPageEditorEffects();
  });

  it("arms Select on load, so the button agrees with what a drag does", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    const select = await screen.findByRole("button", {
      name: /select and pan/i,
    });
    expect(select).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /rectangle segment/i }),
    ).toHaveAttribute("aria-pressed", "false");
  });

  it("moves the armed state to the tool that was picked and back again", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    const rectangle = await screen.findByRole("button", {
      name: /rectangle segment/i,
    });
    fireEvent.click(rectangle);
    expect(rectangle).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /select and pan/i }),
    ).toHaveAttribute("aria-pressed", "false");

    // Clicking the armed tool disarms it rather than doing nothing, which is
    // the second way back to panning besides the Select button.
    fireEvent.click(rectangle);
    expect(rectangle).toHaveAttribute("aria-pressed", "false");
    expect(
      screen.getByRole("button", { name: /select and pan/i }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("holding Space stops an armed drawing tool from claiming the pointer", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    fireEvent.click(
      await screen.findByRole("button", { name: /rectangle segment/i }),
    );

    fireEvent.keyDown(window, { code: "Space" });
    const canvas = screen.getByRole("group", {
      name: /page geometry canvas/i,
    });
    fireEvent.pointerDown(canvas, { clientX: 20, clientY: 30 });
    fireEvent.pointerMove(canvas, { clientX: 120, clientY: 80 });
    fireEvent.pointerUp(canvas, { clientX: 120, clientY: 80 });
    await flushPageEditorEffects();

    // The gesture belonged to the pan, so nothing was drawn or saved.
    expect(mockedApi.createPartLine).not.toHaveBeenCalled();

    // Releasing Space hands the pointer back to the still-armed tool.
    fireEvent.keyUp(window, { code: "Space" });
    fireEvent.pointerDown(canvas, { clientX: 20, clientY: 30 });
    fireEvent.pointerMove(canvas, { clientX: 120, clientY: 80 });
    fireEvent.pointerUp(canvas, { clientX: 120, clientY: 80 });

    await waitFor(() => {
      expect(mockedApi.createPartLine).toHaveBeenCalled();
    });
  });

  it("keeps the pan override off when Space is typed into a field", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();
    expect(await screen.findByText("ANNOTE PAGE WORKSPACE")).toBeTruthy();

    const host = globalThis.document.querySelector(".pe-canvas-host");
    expect(host).not.toBeNull();

    // Typing a space into any text field must reach the field. Panning the
    // page instead would make the transcription strip unusable.
    const field = globalThis.document.createElement("input");
    globalThis.document.body.append(field);
    fireEvent.keyDown(field, { code: "Space" });
    expect(host?.classList.contains("pe-canvas-host--panning")).toBe(false);

    // The same key outside a field does arm the override.
    fireEvent.keyDown(window, { code: "Space" });
    await waitFor(() => {
      expect(
        globalThis.document
          .querySelector(".pe-canvas-host")
          ?.classList.contains("pe-canvas-host--panning"),
      ).toBe(true);
    });

    fireEvent.keyUp(window, { code: "Space" });
    field.remove();
  });

  it("edits vertices whenever a segment is selected and no tool is armed", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue([line({ kind: "polygon" })]);

    renderPageEditor();

    fireEvent.click(await screen.findByRole("button", { name: /^Segment 1/ }));

    // Vertex handles used to require Layout mode. Selection is the only
    // condition now, so they are present without any mode being chosen.
    await waitFor(() => {
      expect(
        globalThis.document.querySelectorAll(".pe-vertex-handle").length,
      ).toBeGreaterThan(0);
    });
  });

  it("is one tab stop with arrow keys inside it, as role=toolbar promises", async () => {
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);

    renderPageEditor();

    const island = await screen.findByRole("toolbar", {
      name: /canvas tools/i,
    });
    const select = screen.getByRole("button", { name: /select and pan/i });
    const rectangle = screen.getByRole("button", {
      name: /rectangle segment/i,
    });

    // Eight buttons floating over the canvas must not cost eight presses to
    // get past, so exactly one of them is in the tab order at a time.
    const stops = [...island.querySelectorAll("button")].filter(
      (button) => button.tabIndex === 0,
    );
    expect(stops).toEqual([select]);

    fireEvent.keyDown(island, { key: "ArrowRight" });
    expect(globalThis.document.activeElement).toBe(rectangle);
    expect(rectangle.tabIndex).toBe(0);
    expect(select.tabIndex).toBe(-1);

    // Delete is greyed out with nothing selected, so the arrow walk steps over
    // it rather than parking the island's only tab stop somewhere unfocusable.
    fireEvent.keyDown(island, { key: "ArrowRight" });
    fireEvent.keyDown(island, { key: "ArrowRight" });
    expect(globalThis.document.activeElement).toBe(
      screen.getByRole("button", { name: /zoom out/i }),
    );

    fireEvent.keyDown(island, { key: "Home" });
    expect(globalThis.document.activeElement).toBe(select);
  });
});
