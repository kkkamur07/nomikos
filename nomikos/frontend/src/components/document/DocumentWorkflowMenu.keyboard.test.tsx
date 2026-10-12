import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

const listInferenceModels = vi.fn();
const enqueueDocumentSegment = vi.fn();
const enqueueDocumentTranscribe = vi.fn();
const listProjectModelBindings = vi.fn();
const createProjectModelBinding = vi.fn();
const deleteProjectModelBinding = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    createProjectModelBinding: (...args: unknown[]) =>
      createProjectModelBinding(...args),
    deleteProjectModelBinding: (...args: unknown[]) =>
      deleteProjectModelBinding(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), info: vi.fn(), error: vi.fn() },
}));

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 2 };
const CATALOG = [
  {
    id: "seg-a",
    task: "segment",
    name: "kraken",
    artifact_ref: "registry://blla-segment?tag=stable",
  },
  {
    id: "seg-b",
    task: "segment",
    name: "pp-ocr",
    artifact_ref: "registry://ppocr-segment?tag=stable",
  },
  { id: "htr-1", task: "transcribe", name: "htr" },
];

/**
 * The bindings endpoints as one small stateful server. A list that always
 * answers the same rows would let an undo look done without deleting
 * anything, so a write has to be visible to the next read.
 */
function serveBindings() {
  let rows: { id: string; task: string; model_id: string }[] = [];
  listProjectModelBindings.mockImplementation(async () => [...rows]);
  createProjectModelBinding.mockImplementation(async (_projectId, body) => {
    const row = { id: `binding-${body.task}`, ...body };
    rows = [...rows.filter((r) => r.task !== body.task), row];
    return row;
  });
  deleteProjectModelBinding.mockImplementation(async (_projectId, id) => {
    rows = rows.filter((r) => r.id !== id);
  });
}

/** The menu plus a focus target outside it, for focus-leave checks. */
function openMenu() {
  render(
    <>
      <button type="button">outside</button>
      <DocumentWorkflowMenu
        projectId="project-1"
        documentId="document-1"
        counts={COUNTS}
        onJobsQueued={() => {}}
      />
    </>,
  );
  fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
  return screen.getByRole("menu", { name: "Document workflow" });
}

function segmentSelect(): HTMLElement {
  return screen.getByRole("combobox", { name: "Segmentation model" });
}

describe("DocumentWorkflowMenu keyboard access", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(CATALOG);
    serveBindings();
  });

  it("keeps the popup open on Tab inside a picker", async () => {
    const menu = openMenu();
    const select = segmentSelect();
    (select as HTMLSelectElement).focus();
    fireEvent.keyDown(select, { key: "Tab" });
    expect(menu).toBeInTheDocument();
  });

  it("leaves select arrow keys to the select instead of walking the menu", async () => {
    openMenu();
    // The select is disabled until the catalog lands; focusing it earlier is
    // a no-op and would leave focus on the auto-focused first run item.
    await screen.findByRole("option", { name: "kraken" });
    const select = segmentSelect();
    (select as HTMLSelectElement).focus();
    expect(document.activeElement).toBe(select);
    fireEvent.keyDown(select, { key: "ArrowDown" });
    expect(document.activeElement).toBe(select);
    fireEvent.keyDown(select, { key: "ArrowUp" });
    expect(document.activeElement).toBe(select);
  });

  it("closes when focus leaves the popup, not when it moves inside", async () => {
    const menu = openMenu();
    const select = segmentSelect();
    (select as HTMLSelectElement).focus();
    fireEvent.blur(select, {
      relatedTarget: screen.getByRole("combobox", {
        name: "HTR transcription model",
      }),
    });
    expect(menu).toBeInTheDocument();

    fireEvent.blur(select, {
      relatedTarget: screen.getByRole("button", { name: "outside" }),
    });
    await waitFor(() =>
      expect(
        screen.queryByRole("menu", { name: "Document workflow" }),
      ).toBeNull(),
    );
  });

  it("closes on Escape and returns focus to the trigger", async () => {
    openMenu();
    const select = segmentSelect();
    (select as HTMLSelectElement).focus();
    fireEvent.keyDown(select, { key: "Escape" });
    await waitFor(() =>
      expect(
        screen.queryByRole("menu", { name: "Document workflow" }),
      ).toBeNull(),
    );
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /workflow/i }),
    );
  });

  it("still walks the run items with arrow keys, wrapping at the ends", async () => {
    openMenu();
    const first = screen.getByRole("menuitem", {
      name: /segment unsegmented pages/i,
    });
    const second = screen.getByRole("menuitem", {
      name: /re-segment every page/i,
    });
    (first as HTMLButtonElement).focus();
    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(document.activeElement).toBe(second);
    fireEvent.keyDown(second, { key: "ArrowUp" });
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(first, { key: "ArrowUp" });
    expect(document.activeElement).toBe(
      screen.getByRole("menuitem", { name: /transcribe unpaired pages/i }),
    );
  });

  it("keeps the popup open when a picker is changed from the keyboard", async () => {
    const menu = openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });
    const select = segmentSelect() as HTMLSelectElement;
    select.focus();

    // A keyboard change is a change event with focus still on the select; the
    // write that follows disables it for a moment, which must not read as
    // focus leaving the popup.
    fireEvent.keyDown(select, { key: "ArrowDown" });
    fireEvent.change(select, { target: { value: "seg-b" } });
    fireEvent.blur(select, { relatedTarget: null });

    await waitFor(() => expect(createProjectModelBinding).toHaveBeenCalled());
    expect(menu).toBeInTheDocument();
    expect(
      screen.getByRole("menu", { name: "Document workflow" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(segmentSelect()).not.toBeDisabled());
  });

  it("keeps the popup open when Undo is activated from the keyboard", async () => {
    const menu = openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });
    const select = segmentSelect() as HTMLSelectElement;
    select.focus();
    fireEvent.change(select, { target: { value: "seg-b" } });

    const undo = await screen.findByRole("button", { name: "Undo" });
    undo.focus();
    expect(document.activeElement).toBe(undo);
    // Enter on a plain button inside the popup is a click, and the menu's own
    // key handling must leave it alone.
    fireEvent.keyDown(undo, { key: "Enter" });
    fireEvent.click(undo);
    fireEvent.blur(undo, { relatedTarget: null });

    // Waiting for the Undo button to go is not enough: it is also absent
    // while the undo write runs, when the select still shows the pick. The
    // settled state is the select back on the restored model.
    await waitFor(() => {
      expect(segmentSelect()).toHaveValue("seg-a");
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    });
    expect(deleteProjectModelBinding).toHaveBeenCalledWith(
      "project-1",
      "binding-segment",
    );
    expect(menu).toBeInTheDocument();
  });

  it("orders pickers and run items in DOM order", async () => {
    const menu = openMenu();
    await screen.findByRole("combobox", { name: "HTR transcription model" });
    const tabbables = Array.from(
      menu.querySelectorAll("select, button:not([disabled])"),
    ).map(
      (element) => element.getAttribute("aria-label") ?? element.textContent,
    );
    expect(tabbables).toEqual([
      "Segmentation model",
      expect.stringMatching(/segment unsegmented pages/i),
      expect.stringMatching(/re-segment every page/i),
      "HTR transcription model",
      expect.stringMatching(/transcribe unpaired pages/i),
    ]);
  });
});
