import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import {
  clearProjectDefault,
  saveProjectDefault,
  undoProjectDefault,
  useProjectModelDefaults,
  type BindingStore,
  type ProjectBinding,
} from "./projectModelDefaults";

function storeWith(bindings: ProjectBinding[]): BindingStore & {
  calls: { create: number; update: number; remove: number };
} {
  const calls = { create: 0, update: 0, remove: 0 };
  let rows = [...bindings];
  return {
    calls,
    list: async () => [...rows],
    create: async (_projectId, task, modelId) => {
      calls.create += 1;
      const created = {
        id: `binding-${task}`,
        task,
        model_id: modelId,
      };
      rows = [...rows.filter((row) => row.task !== task), created];
      return created;
    },
    update: async (_projectId, bindingId, modelId) => {
      calls.update += 1;
      const current = rows.find((row) => row.id === bindingId);
      if (!current) {
        throw new ApiError("Model binding not found", 404);
      }
      const updated = { ...current, model_id: modelId };
      rows = rows.map((row) => (row.id === bindingId ? updated : row));
      return updated;
    },
    remove: async (_projectId, bindingId) => {
      calls.remove += 1;
      if (!rows.some((row) => row.id === bindingId)) {
        throw new ApiError("Model binding not found", 404);
      }
      rows = rows.filter((row) => row.id !== bindingId);
    },
  };
}

describe("saveProjectDefault", () => {
  it("POSTs when the project has no binding for the task", async () => {
    const store = storeWith([]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "transcribe",
      "htr-syriac",
    );
    expect(saved).toMatchObject({ task: "transcribe", model_id: "htr-syriac" });
    expect(store.calls).toEqual({ create: 1, update: 0, remove: 0 });
  });

  it("PATCHes the existing binding instead of creating a second row", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
    ]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "transcribe",
      "htr-syriac",
    );
    expect(saved).toMatchObject({ id: "binding-1", model_id: "htr-syriac" });
    expect(store.calls).toEqual({ create: 0, update: 1, remove: 0 });
  });

  it("leaves a binding that already points at the model alone", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const saved = await saveProjectDefault(
      store,
      "project-1",
      "segment",
      "seg-a",
    );
    expect(saved).toMatchObject({ id: "binding-1" });
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 0 });
  });

  it("refetches and PATCHes when a collaborator wins the create race", async () => {
    const raced = {
      id: "binding-race",
      task: "segment",
      model_id: "seg-b",
    } as const;
    const store = storeWith([{ ...raced }]);
    const create = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("A binding already exists", 409));
    const racing: BindingStore = {
      ...store,
      create,
      list: async () => [{ ...raced }],
    };
    const saved = await saveProjectDefault(
      racing,
      "project-1",
      "segment",
      "seg-a",
    );
    expect(saved).toMatchObject({ id: "binding-race", model_id: "seg-a" });
    expect(store.calls.update).toBe(1);
  });

  it("rethrows a conflict when even the refetch finds no binding", async () => {
    const racing: BindingStore = {
      ...storeWith([]),
      create: async () => {
        throw new ApiError("A binding already exists", 409);
      },
    };
    await expect(
      saveProjectDefault(racing, "project-1", "segment", "seg-a"),
    ).rejects.toMatchObject({ status: 409 });
  });

  it("rethrows failures that are not a create race", async () => {
    const failing: BindingStore = {
      ...storeWith([]),
      create: async () => {
        throw new ApiError("No access", 403);
      },
    };
    await expect(
      saveProjectDefault(failing, "project-1", "segment", "seg-a"),
    ).rejects.toMatchObject({ status: 403 });
  });
});

describe("clearProjectDefault", () => {
  it("deletes the binding for the task", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
      { id: "binding-2", task: "segment", model_id: "seg-a" },
    ]);
    await clearProjectDefault(store, "project-1", "transcribe");
    expect(store.calls.remove).toBe(1);
    expect(await store.list("project-1")).toEqual([
      { id: "binding-2", task: "segment", model_id: "seg-a" },
    ]);
  });

  it("does nothing when the task has no binding", async () => {
    const store = storeWith([]);
    await clearProjectDefault(store, "project-1", "segment");
    expect(store.calls.remove).toBe(0);
  });

  it("treats a binding deleted in the meantime as cleared", async () => {
    let rows: ProjectBinding[] = [
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ];
    const racing: BindingStore = {
      ...storeWith([]),
      list: async () => [...rows],
      remove: async () => {
        // Someone else deleted it between the list and this call.
        rows = [];
        throw new ApiError("Model binding not found", 404);
      },
    };
    await expect(
      clearProjectDefault(racing, "project-1", "segment"),
    ).resolves.toBeUndefined();
  });

  it("deletes the row that replaced a stale id instead of reporting success", async () => {
    let rows: ProjectBinding[] = [
      { id: "binding-old", task: "segment", model_id: "seg-a" },
    ];
    const removed: string[] = [];
    const racing: BindingStore = {
      ...storeWith([]),
      list: async () => [...rows],
      remove: async (_projectId, bindingId) => {
        removed.push(bindingId);
        if (bindingId === "binding-old") {
          // A collaborator replaced the binding: same task, new id.
          rows = [{ id: "binding-new", task: "segment", model_id: "seg-b" }];
          throw new ApiError("Model binding not found", 404);
        }
        rows = rows.filter((row) => row.id !== bindingId);
      },
    };

    await expect(
      clearProjectDefault(racing, "project-1", "segment"),
    ).resolves.toBeUndefined();

    expect(removed).toEqual(["binding-old", "binding-new"]);
    expect(rows).toEqual([]);
  });

  it("gives up rather than call a default cleared that is still there", async () => {
    let created = 0;
    const racing: BindingStore = {
      ...storeWith([]),
      // Every list answers with a fresh row under a fresh id.
      list: async () => {
        created += 1;
        return [
          { id: `binding-${created}`, task: "segment", model_id: "seg-a" },
        ];
      },
      remove: async () => {
        throw new ApiError("Model binding not found", 404);
      },
    };
    await expect(
      clearProjectDefault(racing, "project-1", "segment"),
    ).rejects.toThrow(/keeps being replaced/);
  });

  it("rethrows a failure that is not a missing binding", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const failing: BindingStore = {
      ...store,
      remove: async () => {
        throw new ApiError("No access", 403);
      },
    };
    await expect(
      clearProjectDefault(failing, "project-1", "segment"),
    ).rejects.toMatchObject({ status: 403 });
  });
});

describe("undoProjectDefault", () => {
  const PICK = { modelId: "seg-b", bindingId: "binding-1" };

  it("puts the previous model back when the pick is still the default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-b" },
    ]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      "seg-a",
    );
    expect(outcome.stale).toBe(false);
    expect(outcome.binding).toMatchObject({
      id: "binding-1",
      model_id: "seg-a",
    });
    expect(store.calls).toEqual({ create: 0, update: 1, remove: 0 });
    expect(await store.list("project-1")).toEqual([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
  });

  it("deletes the binding when the pick created the project's first one", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-b" },
    ]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      null,
    );
    expect(outcome.stale).toBe(false);
    expect(outcome.binding).toBeNull();
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 1 });
    expect(await store.list("project-1")).toEqual([]);
  });

  it("writes nothing when another member set their own model in between", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-c" },
    ]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      "seg-a",
    );
    expect(outcome.stale).toBe(true);
    expect(outcome.binding).toMatchObject({ model_id: "seg-c" });
    expect(outcome.bindings).toEqual([
      { id: "binding-1", task: "segment", model_id: "seg-c" },
    ]);
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 0 });
  });

  it("writes nothing when another member cleared the default in between", async () => {
    const store = storeWith([]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      "seg-a",
    );
    expect(outcome.stale).toBe(true);
    expect(outcome.binding).toBeNull();
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 0 });
  });

  it("writes nothing when the same model sits under a new row", async () => {
    // A collaborator cleared the default and set the same model again: same
    // model id, new row. The pick that would be undone is gone all the same.
    const store = storeWith([
      { id: "binding-new", task: "segment", model_id: "seg-b" },
    ]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      "seg-a",
    );
    expect(outcome.stale).toBe(true);
    expect(store.calls).toEqual({ create: 0, update: 0, remove: 0 });
  });

  it("leaves the other tasks alone", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-b" },
      { id: "binding-2", task: "transcribe", model_id: "htr-greek" },
    ]);
    const outcome = await undoProjectDefault(
      store,
      "project-1",
      "segment",
      PICK,
      null,
    );
    expect(await store.list("project-1")).toEqual([
      { id: "binding-2", task: "transcribe", model_id: "htr-greek" },
    ]);
    // A reversal reports its own row only. The list it walked was read before
    // its write, so it is no news about the other task.
    expect(outcome.bindings).toBeUndefined();
  });

  it("adopts what is there when the row goes between the read and the write", async () => {
    let rows: ProjectBinding[] = [
      { id: "binding-1", task: "segment", model_id: "seg-b" },
    ];
    const racing: BindingStore = {
      ...storeWith([]),
      list: async () => [...rows],
      remove: async () => {
        rows = [{ id: "binding-new", task: "segment", model_id: "seg-c" }];
        throw new ApiError("Model binding not found", 404);
      },
    };
    const outcome = await undoProjectDefault(
      racing,
      "project-1",
      "segment",
      PICK,
      null,
    );
    expect(outcome.stale).toBe(true);
    expect(outcome.binding).toMatchObject({ model_id: "seg-c" });
  });

  it("rethrows a failure that is not a missing row", async () => {
    const failing: BindingStore = {
      ...storeWith([{ id: "binding-1", task: "segment", model_id: "seg-b" }]),
      update: async () => {
        throw new ApiError("No access", 403);
      },
    };
    await expect(
      undoProjectDefault(failing, "project-1", "segment", PICK, "seg-a"),
    ).rejects.toMatchObject({ status: 403 });
  });
});

describe("useProjectModelDefaults", () => {
  it("loads the bindings and reports each task default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-syriac" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac"),
    );
    expect(result.current.defaultModelId("segment")).toBeNull();
  });

  it("saves through the store and moves the default", async () => {
    const store = storeWith([]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await act(async () => {
      await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
    expect(store.calls.create).toBe(1);
  });

  it("reports a failed save and keeps the old default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "transcribe", model_id: "htr-greek" },
    ]);
    const failing: BindingStore = {
      ...store,
      update: async () => {
        throw new ApiError("No access", 403);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", failing),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("transcribe")).toBe("htr-greek"),
    );
    let saved: unknown;
    await act(async () => {
      saved = await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(saved).toMatchObject({ ok: false, message: "No access" });
    expect(result.current.error).toMatch(/No access/);
    expect(result.current.defaultModelId("transcribe")).toBe("htr-greek");
  });

  it("clears through the store and drops the default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("segment")).toBe("seg-a"),
    );
    await act(async () => {
      await result.current.clearDefault("segment");
    });
    expect(result.current.defaultModelId("segment")).toBeNull();
    expect(store.calls.remove).toBe(1);
  });

  it("reports a failed clear and keeps the stored default", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const failing: BindingStore = {
      ...store,
      remove: async () => {
        throw new ApiError("No access", 403);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", failing),
    );
    await waitFor(() =>
      expect(result.current.defaultModelId("segment")).toBe("seg-a"),
    );
    let cleared: unknown;
    await act(async () => {
      cleared = await result.current.clearDefault("segment");
    });
    expect(cleared).toMatchObject({ ok: false, message: "No access" });
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
  });

  it("keeps a save that lands while the initial list is still in flight", async () => {
    let releaseFirst!: (rows: ProjectBinding[]) => void;
    const firstGate = new Promise<ProjectBinding[]>((resolve) => {
      releaseFirst = resolve;
    });
    let calls = 0;
    const store = storeWith([]);
    const gated: BindingStore = {
      ...store,
      list: async () => {
        calls += 1;
        return calls === 1 ? firstGate : [];
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", gated),
    );
    // The initial list goes first; the save below must be the second call.
    await waitFor(() => expect(calls).toBe(1));
    await act(async () => {
      await result.current.saveDefault("transcribe", "htr-syriac");
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
    await act(async () => {
      releaseFirst([]);
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
  });

  it("refresh re-reads the bindings after an outside write", async () => {
    const store = storeWith([]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await store.create("project-1", "transcribe", "htr-syriac");
    expect(result.current.defaultModelId("transcribe")).toBeNull();
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-syriac");
  });

  it("fetches nothing without a project", () => {
    const store = storeWith([]);
    const list = vi.spyOn(store, "list");
    renderHook(() => useProjectModelDefaults(undefined, store));
    expect(list).not.toHaveBeenCalled();
  });

  it("reports a failed read as unknown rather than as no defaults", async () => {
    const store = storeWith([]);
    let listed = 0;
    const offline: BindingStore = {
      ...store,
      list: async () => {
        listed += 1;
        if (listed === 1) throw new ApiError("offline", 503);
        return [{ id: "binding-1", task: "segment", model_id: "seg-a" }];
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", offline),
    );

    await waitFor(() => expect(result.current.loadFailed).toBe(true));
    expect(result.current.loading).toBe(false);
    expect(result.current.known).toBe(false);
    expect(result.current.defaultModelId("segment")).toBeNull();

    // "Try again" is the way back, and a good read clears the doubt.
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.loadFailed).toBe(false);
    expect(result.current.known).toBe(true);
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
  });

  it("holds one saving state, so a write freezes both rows", async () => {
    const store = storeWith([]);
    let release!: (binding: ProjectBinding) => void;
    const gate = new Promise<ProjectBinding>((resolve) => {
      release = resolve;
    });
    const slow: BindingStore = { ...store, create: async () => gate };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", slow),
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    let pending!: Promise<unknown>;
    await act(async () => {
      pending = result.current.saveDefault("segment", "seg-a");
    });
    // One state for the project: the row being written names itself, and
    // every control the hook feeds is frozen, so no second write can start.
    expect(result.current.saving).toBe(true);
    expect(result.current.savingTask).toBe("segment");

    await act(async () => {
      release({ id: "binding-1", task: "segment", model_id: "seg-a" });
      await pending;
    });
    expect(result.current.saving).toBe(false);
    expect(result.current.savingTask).toBeNull();
  });

  it("drops a write that lands after the project changed", async () => {
    const rows: Record<string, ProjectBinding[]> = {
      "project-a": [],
      "project-b": [{ id: "binding-b", task: "segment", model_id: "seg-b" }],
    };
    let release!: (binding: ProjectBinding) => void;
    const gate = new Promise<ProjectBinding>((resolve) => {
      release = resolve;
    });
    const store: BindingStore = {
      ...storeWith([]),
      list: async (projectId) => [...(rows[projectId] ?? [])],
      create: async () => gate,
    };
    const { result, rerender } = renderHook(
      ({ projectId }) => useProjectModelDefaults(projectId, store),
      { initialProps: { projectId: "project-a" } },
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    let pending!: Promise<unknown>;
    await act(async () => {
      pending = result.current.saveDefault("segment", "seg-a");
    });

    rerender({ projectId: "project-b" });
    await waitFor(() =>
      expect(result.current.defaultModelId("segment")).toBe("seg-b"),
    );

    let stale: unknown;
    await act(async () => {
      release({ id: "binding-a", task: "segment", model_id: "seg-a" });
      stale = await pending;
    });

    // A's answer is about a project nobody is looking at any more.
    expect(stale).toMatchObject({ ok: false, superseded: true });
    expect(result.current.defaultModelId("segment")).toBe("seg-b");
    expect(result.current.loading).toBe(false);
    expect(result.current.known).toBe(true);
    expect(result.current.saving).toBe(false);
  });

  it("ends loading even when a write invalidates the list in flight", async () => {
    let releaseList!: (rows: ProjectBinding[]) => void;
    const listGate = new Promise<ProjectBinding[]>((resolve) => {
      releaseList = resolve;
    });
    let listed = 0;
    const store: BindingStore = {
      ...storeWith([]),
      list: async () => {
        listed += 1;
        return listed === 1 ? listGate : [];
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() => expect(listed).toBe(1));
    expect(result.current.loading).toBe(true);

    await act(async () => {
      await result.current.saveDefault("segment", "seg-a");
    });
    // The write knows the bindings, so it owns `loading` from here: the list
    // it invalidated must not leave the card waiting forever.
    expect(result.current.loading).toBe(false);
    expect(result.current.known).toBe(true);

    await act(async () => {
      releaseList([]);
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
  });
  it("undoes a pick through the hook and moves the default back", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-a" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    await act(async () => {
      await result.current.saveDefault("segment", "seg-b");
    });
    let undone!: Awaited<ReturnType<typeof result.current.undoDefault>>;
    await act(async () => {
      undone = await result.current.undoDefault(
        "segment",
        { modelId: "seg-b", bindingId: "binding-1" },
        "seg-a",
      );
    });
    expect(undone).toMatchObject({ ok: true, stale: false });
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
    expect(store.calls.remove).toBe(0);
  });

  it("adopts another member's default instead of undoing over it", async () => {
    const store = storeWith([
      { id: "binding-1", task: "segment", model_id: "seg-b" },
    ]);
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", store),
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    // Somebody else moved the project on after this session's pick.
    await store.update("project-1", "binding-1", "seg-c");
    const writesBefore = { ...store.calls };

    let undone!: Awaited<ReturnType<typeof result.current.undoDefault>>;
    await act(async () => {
      undone = await result.current.undoDefault(
        "segment",
        { modelId: "seg-b", bindingId: "binding-1" },
        "seg-a",
      );
    });
    expect(undone).toMatchObject({ ok: true, stale: true });
    expect(store.calls).toEqual(writesBefore);
    // The hook now speaks for the project as it is, not as the pick left it.
    expect(result.current.defaultModelId("segment")).toBe("seg-c");
    expect(result.current.saving).toBe(false);
  });
  it("does not carry an undo's older view of the other task into state", async () => {
    const store = storeWith([
      { id: "binding-s", task: "segment", model_id: "seg-b" },
      { id: "binding-t", task: "transcribe", model_id: "htr-x" },
    ]);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const gated: BindingStore = {
      ...store,
      update: async (projectId, bindingId, modelId) => {
        if (bindingId === "binding-s") await gate;
        return store.update(projectId, bindingId, modelId);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", gated),
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    let pending!: Promise<unknown>;
    await act(async () => {
      pending = result.current.undoDefault(
        "segment",
        { modelId: "seg-b", bindingId: "binding-s" },
        "seg-a",
      );
    });

    // While the undo waits on the server, somebody else moves the other task
    // and this session reads that.
    await store.update("project-1", "binding-t", "htr-y");
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-y");

    await act(async () => {
      release();
      await pending;
    });
    // The undo listed the bindings before it wrote, when transcribe still read
    // htr-x. A write speaks for its own task only, so that view stays out.
    expect(result.current.defaultModelId("segment")).toBe("seg-a");
    expect(result.current.defaultModelId("transcribe")).toBe("htr-y");
  });

  it("does not carry a save's older view of the other task into state", async () => {
    const store = storeWith([
      { id: "binding-s", task: "segment", model_id: "seg-a" },
      { id: "binding-t", task: "transcribe", model_id: "htr-x" },
    ]);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const gated: BindingStore = {
      ...store,
      update: async (projectId, bindingId, modelId) => {
        if (bindingId === "binding-s") await gate;
        return store.update(projectId, bindingId, modelId);
      },
    };
    const { result } = renderHook(() =>
      useProjectModelDefaults("project-1", gated),
    );
    await waitFor(() => expect(result.current.known).toBe(true));

    let pending!: Promise<unknown>;
    await act(async () => {
      pending = result.current.saveDefault("segment", "seg-b");
    });

    await store.update("project-1", "binding-t", "htr-y");
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.defaultModelId("transcribe")).toBe("htr-y");

    await act(async () => {
      release();
      await pending;
    });
    expect(result.current.defaultModelId("segment")).toBe("seg-b");
    expect(result.current.defaultModelId("transcribe")).toBe("htr-y");
  });
});
