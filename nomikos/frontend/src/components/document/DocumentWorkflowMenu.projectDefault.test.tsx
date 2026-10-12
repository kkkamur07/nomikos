import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { DocumentWorkflowMenu } from "./DocumentWorkflowMenu";

const listInferenceModels = vi.fn();
const listProjectModelBindings = vi.fn();
const createProjectModelBinding = vi.fn();
const updateProjectModelBinding = vi.fn();
const deleteProjectModelBinding = vi.fn();
const enqueueDocumentSegment = vi.fn();
const enqueueDocumentTranscribe = vi.fn();
const error = vi.fn();
const info = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listInferenceModels: (...args: unknown[]) => listInferenceModels(...args),
    listProjectModelBindings: (...args: unknown[]) =>
      listProjectModelBindings(...args),
    createProjectModelBinding: (...args: unknown[]) =>
      createProjectModelBinding(...args),
    updateProjectModelBinding: (...args: unknown[]) =>
      updateProjectModelBinding(...args),
    deleteProjectModelBinding: (...args: unknown[]) =>
      deleteProjectModelBinding(...args),
    enqueueDocumentSegment: (...args: unknown[]) =>
      enqueueDocumentSegment(...args),
    enqueueDocumentTranscribe: (...args: unknown[]) =>
      enqueueDocumentTranscribe(...args),
  },
}));

vi.mock("../ui/toast", () => ({
  toast: {
    success: vi.fn(),
    info: (...args: unknown[]) => info(...args),
    error: (...args: unknown[]) => error(...args),
  },
}));

/** What the menu says when an undo finds somebody else's choice. */
const UNDO_STALE =
  "The project default was changed in the meantime, nothing was undone.";

const COUNTS = { total: 3, reviewed: 0, unsegmented: 1, unpaired: 2 };
const CATALOG = [
  { id: "seg-a", task: "segment", name: "kraken" },
  { id: "seg-b", task: "segment", name: "pp-ocr" },
  { id: "htr-1", task: "transcribe", name: "greek" },
  { id: "htr-2", task: "transcribe", name: "syriac" },
];

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function openMenu() {
  render(
    <DocumentWorkflowMenu
      projectId="project-1"
      documentId="document-1"
      counts={COUNTS}
      onJobsQueued={() => {}}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
}

function segmentSelect(): HTMLSelectElement {
  return screen.getByRole("combobox", {
    name: "Segmentation model",
  }) as HTMLSelectElement;
}

function transcribeSelect(): HTMLSelectElement {
  return screen.getByRole("combobox", {
    name: "HTR transcription model",
  }) as HTMLSelectElement;
}

type StoredBinding = { id: string; task: string; model_id: string };

/**
 * The bindings endpoints as one small stateful server: a write is visible to
 * the next read, which is what `saveProjectDefault` and `clearProjectDefault`
 * both walk through before they touch anything.
 */
function serveBindings(initial: StoredBinding[]) {
  let rows = [...initial];
  listProjectModelBindings.mockImplementation(async () => [...rows]);
  createProjectModelBinding.mockImplementation(async (_projectId, body) => {
    const row = { id: `binding-${body.task}`, ...body };
    rows = [...rows.filter((r) => r.task !== body.task), row];
    return row;
  });
  updateProjectModelBinding.mockImplementation(
    async (_projectId, bindingId, body) => {
      const previous = rows.find((r) => r.id === bindingId);
      const row = {
        id: bindingId,
        task: previous ? previous.task : "segment",
        ...body,
      };
      rows = [...rows.filter((r) => r.id !== bindingId), row];
      return row;
    },
  );
  deleteProjectModelBinding.mockImplementation(
    async (_projectId, bindingId) => {
      rows = rows.filter((r) => r.id !== bindingId);
    },
  );
  const read = () => rows;
  /** What another member's write looks like from this side of the network. */
  read.set = (next: StoredBinding[]) => {
    rows = [...next];
  };
  return read;
}

/** Every write the four endpoints have seen, for asserting that none happened. */
function writeCounts() {
  return {
    create: createProjectModelBinding.mock.calls.length,
    update: updateProjectModelBinding.mock.calls.length,
    remove: deleteProjectModelBinding.mock.calls.length,
  };
}

describe("DocumentWorkflowMenu sets the project default", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listInferenceModels.mockResolvedValue(CATALOG);
    serveBindings([]);
    enqueueDocumentSegment.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
    enqueueDocumentTranscribe.mockResolvedValue({
      queued: 1,
      skipped: 0,
      jobs: [],
    });
  });

  it("saves the pick as the project default and offers the way back", async () => {
    const write = deferred<unknown>();
    createProjectModelBinding.mockReturnValue(write.promise);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    await waitFor(() => expect(screen.getByText("Saving…")).toBeTruthy());
    // One write at a time for the whole project, so both pickers are frozen.
    expect(segmentSelect()).toBeDisabled();
    expect(transcribeSelect()).toBeDisabled();

    write.resolve({ id: "binding-1", task: "segment", model_id: "seg-b" });

    await waitFor(() =>
      expect(screen.getByText("Saved as project default.")).toBeTruthy(),
    );
    expect(screen.getByRole("button", { name: "Undo" })).toBeTruthy();
    expect(createProjectModelBinding).toHaveBeenCalledWith("project-1", {
      task: "segment",
      model_id: "seg-b",
    });
    expect(segmentSelect()).toHaveValue("seg-b");
    expect(transcribeSelect()).not.toBeDisabled();
  });

  it("undoes the pick back to the binding it replaced", async () => {
    const rows = serveBindings([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    openMenu();
    await waitFor(() => expect(segmentSelect()).toHaveValue("seg-a"));

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));

    // One undo per pick, and the line speaks for the project again.
    await waitFor(() => {
      expect(segmentSelect()).toHaveValue("seg-a");
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
      expect(screen.getByText("Project default")).toBeTruthy();
    });
    expect(rows()).toEqual([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
  });

  it("undoes a first pick by clearing the binding it created", async () => {
    const rows = serveBindings([]);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));

    await waitFor(() =>
      expect(deleteProjectModelBinding).toHaveBeenCalledWith(
        "project-1",
        "binding-segment",
      ),
    );
    expect(rows()).toEqual([]);
    // Back to the model the picker showed before the pick, and no claim.
    await waitFor(() => {
      expect(segmentSelect()).toHaveValue("seg-a");
      expect(screen.queryByText("Project default")).toBeNull();
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    });
  });

  it("keeps the pick on screen until the undo write lands", async () => {
    serveBindings([]);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    const slowDelete = deferred<void>();
    deleteProjectModelBinding.mockReturnValue(slowDelete.promise);
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));

    // The offer goes as soon as the write starts, but the project still
    // stores the pick, so that is what the picker shows meanwhile. This is
    // the window a test that waits only for the button to go lands in.
    await waitFor(() => expect(screen.getByText("Saving…")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    expect(segmentSelect()).toHaveValue("seg-b");

    slowDelete.resolve();
    await waitFor(() => {
      expect(segmentSelect()).toHaveValue("seg-a");
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    });
  });

  it("keeps the stored state when the undo fails", async () => {
    serveBindings([{ id: "binding-1", task: "segment", model_id: "seg-a" }]);
    const write = updateProjectModelBinding.getMockImplementation()!;
    updateProjectModelBinding
      .mockImplementationOnce(write)
      .mockRejectedValueOnce(new ApiError("No access", 403));
    openMenu();
    await waitFor(() => expect(segmentSelect()).toHaveValue("seg-a"));

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));

    // The pick is what the project stores, so that is what the menu shows.
    await waitFor(() => {
      expect(error).toHaveBeenCalledWith("No access");
      expect(segmentSelect()).toHaveValue("seg-b");
      expect(screen.getByText("Project default")).toBeTruthy();
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    });
  });

  it("takes the undo offer away when the menu is closed", async () => {
    serveBindings([]);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    await screen.findByRole("button", { name: "Undo" });

    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));
    fireEvent.click(screen.getByRole("button", { name: /workflow/i }));

    await screen.findByRole("option", { name: "pp-ocr" });
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
      expect(screen.getByText("Project default")).toBeTruthy();
    });
  });

  it("leaves another member's newer default alone instead of undoing over it", async () => {
    const serve = serveBindings([]);
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    await screen.findByRole("button", { name: "Undo" });

    // Somebody else picked their own model for the project in the meantime.
    serve.set([{ id: "binding-other", task: "segment", model_id: "seg-a" }]);
    const before = writeCounts();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(info).toHaveBeenCalledWith(UNDO_STALE);
      // Their choice is what the project holds, so that is what the menu says.
      expect(segmentSelect()).toHaveValue("seg-a");
      expect(screen.getByText("Project default")).toBeTruthy();
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    });
    expect(writeCounts()).toEqual(before);
    expect(serve()).toEqual([
      { id: "binding-other", task: "segment", model_id: "seg-a" },
    ]);
    expect(error).not.toHaveBeenCalled();
  });

  it("writes nothing when another member cleared the default before the undo", async () => {
    const serve = serveBindings([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    openMenu();
    await waitFor(() => expect(segmentSelect()).toHaveValue("seg-a"));

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });
    await screen.findByRole("button", { name: "Undo" });

    // Somebody else cleared it: putting the old model back would be a choice
    // nobody made.
    serve.set([]);
    const before = writeCounts();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(info).toHaveBeenCalledWith(UNDO_STALE);
      expect(screen.queryByText("Project default")).toBeNull();
      expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
      // The project stores nothing, so the picker is back to a run choice.
      expect(segmentSelect()).toHaveValue("seg-a");
    });
    expect(writeCounts()).toEqual(before);
    expect(serve()).toEqual([]);
    expect(error).not.toHaveBeenCalled();
  });

  it("marks a stored default without writing anything", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "transcribe", model_id: "htr-2" },
    ]);
    openMenu();

    await waitFor(() => expect(transcribeSelect()).toHaveValue("htr-2"));
    expect(screen.getAllByText("Project default")).toHaveLength(1);
    expect(createProjectModelBinding).not.toHaveBeenCalled();
    expect(updateProjectModelBinding).not.toHaveBeenCalled();
  });

  it("returns the select to the stored default when the write fails", async () => {
    listProjectModelBindings.mockResolvedValue([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    updateProjectModelBinding.mockRejectedValue(new ApiError("No access", 403));
    openMenu();
    await waitFor(() => expect(segmentSelect()).toHaveValue("seg-a"));

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    await waitFor(() => {
      expect(error).toHaveBeenCalledWith("No access");
      expect(segmentSelect()).toHaveValue("seg-a");
    });
  });

  it("keeps the pick for the run when the project has no stored default", async () => {
    createProjectModelBinding.mockRejectedValue(new ApiError("No access", 403));
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    // Nothing stored to fall back to, so the run keeps the model it replaced.
    await waitFor(() => {
      expect(error).toHaveBeenCalledWith("No access");
      expect(segmentSelect()).toHaveValue("seg-a");
    });
    expect(screen.queryByText("Project default")).toBeNull();
  });

  it("writes no default and claims none while the bindings cannot be read", async () => {
    listProjectModelBindings.mockRejectedValue(new ApiError("offline", 503));
    openMenu();
    await screen.findByRole("option", { name: "pp-ocr" });

    fireEvent.change(segmentSelect(), { target: { value: "seg-b" } });

    // The pick still runs the job, it just does not speak for the project.
    fireEvent.click(
      screen.getByRole("menuitem", { name: /segment unsegmented pages/i }),
    );
    await waitFor(() =>
      expect(enqueueDocumentSegment).toHaveBeenCalledWith(
        "project-1",
        "document-1",
        { scope: "unsegmented", model_id: "seg-b" },
      ),
    );
    expect(createProjectModelBinding).not.toHaveBeenCalled();
    expect(updateProjectModelBinding).not.toHaveBeenCalled();
    expect(screen.queryByText("Project default")).toBeNull();
    expect(error).not.toHaveBeenCalled();
  });
});
