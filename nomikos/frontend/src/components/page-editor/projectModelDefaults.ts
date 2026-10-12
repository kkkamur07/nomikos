import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  type InferenceTask,
  type ModelBindingResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";

/**
 * What a save or a clear answers with. The message travels back to the
 * caller rather than only into `error` state, so the toast at the call site
 * says what the API said instead of reading a value React has not set yet.
 *
 * `superseded` marks a response the caller must act on in no way at all: the
 * hook has moved to another project since the write started, so this outcome
 * is neither the project on screen nor news for the person.
 */
export type ProjectDefaultResult =
  | {
      ok: true;
      superseded?: false;
      /**
       * True when the write looked at the project's rows, found the one it
       * was about changed by somebody else, and therefore wrote nothing. The
       * hook still adopted what it read, so `binding` is what the project
       * holds now, not what this call wanted to put there.
       */
      stale?: boolean;
      binding: ProjectBinding | null;
    }
  | { ok: false; superseded?: false; message: string }
  | { ok: false; superseded: true };

/** How often a clear re-lists after a 404 before it gives up. */
const CLEAR_ATTEMPTS = 3;

/**
 * What one write leaves the hook to store. A write speaks for its own task and
 * for nothing else: `binding` replaces that task's row, or removes it when it
 * is null, and every other task keeps whatever the state holds.
 *
 * `adopt` is the one exception, for rows a write read after somebody else's
 * change: those are newer than the state, so they replace it whole. A list a
 * write read *before* its own write is not that, however fresh it looked at
 * the time: another member may have moved a different task since.
 */
type WriteOutcome = {
  binding: ProjectBinding | null;
  adopt?: ProjectBinding[];
  stale?: boolean;
};

export type ProjectBinding = Pick<
  ModelBindingResponse,
  "id" | "task" | "model_id"
>;

export type BindingStore = {
  list: (projectId: string) => Promise<ProjectBinding[]>;
  create: (
    projectId: string,
    task: InferenceTask,
    modelId: string,
  ) => Promise<ProjectBinding>;
  update: (
    projectId: string,
    bindingId: string,
    modelId: string,
  ) => Promise<ProjectBinding>;
  remove: (projectId: string, bindingId: string) => Promise<void>;
};

export function apiBindingStore(): BindingStore {
  return {
    list: (projectId) => api.listProjectModelBindings(projectId),
    create: (projectId, task, modelId) =>
      api.createProjectModelBinding(projectId, { task, model_id: modelId }),
    update: (projectId, bindingId, modelId) =>
      api.updateProjectModelBinding(projectId, bindingId, {
        model_id: modelId,
      }),
    remove: (projectId, bindingId) =>
      api.deleteProjectModelBinding(projectId, bindingId),
  };
}

/**
 * Saving a project default: POST when the project has no binding for the
 * task, PATCH the binding when one exists. A 409 from the create means a
 * collaborator saved one in the meantime, so refetch and PATCH that row
 * instead of failing.
 */
export async function saveProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
  modelId: string,
): Promise<ProjectBinding> {
  const bindings = await store.list(projectId);
  const existing = bindings.find((binding) => binding.task === task);
  if (existing) {
    if (existing.model_id === modelId) return existing;
    try {
      return await store.update(projectId, existing.id, modelId);
    } catch (err) {
      // Deleted between the list and the update; fall through and create.
      if (!(err instanceof ApiError) || err.status !== 404) throw err;
    }
  }
  try {
    return await store.create(projectId, task, modelId);
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 409) throw err;
    const fresh = await store.list(projectId);
    const raced = fresh.find((binding) => binding.task === task);
    if (!raced) throw err;
    return await store.update(projectId, raced.id, modelId);
  }
}

/**
 * Clearing a project default: delete the binding for the task if there is
 * one. Nothing to delete is success, not an error: a collaborator who
 * cleared it first left the project in exactly the wanted state.
 *
 * A 404 from the delete means the id went stale between the list and the
 * delete, not that the default is gone: a collaborator may have replaced the
 * binding with a new row for the same task. So re-list and delete what is
 * there now, and report the clear only once a list comes back without one.
 */
export async function clearProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
): Promise<void> {
  for (let attempt = 0; attempt < CLEAR_ATTEMPTS; attempt += 1) {
    const bindings = await store.list(projectId);
    const existing = bindings.find((binding) => binding.task === task);
    if (!existing) return;
    try {
      await store.remove(projectId, existing.id);
      return;
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 404) throw err;
    }
  }
  throw new Error(
    "Could not clear the project default: it keeps being replaced.",
  );
}

/**
 * What one pick wrote, so an undo can tell its own row from a newer one.
 * `bindingId` is null when the pick's answer carried no row, which only a
 * test double does; the model id alone is then the test.
 */
export type PickRecord = { modelId: string; bindingId: string | null };

/**
 * What an undo did. `stale` means it found somebody else's value under the
 * pick and left it alone.
 *
 * `bindings` comes with a stale answer only, and then it is the whole project
 * as the undo read it after that change: rows worth adopting. A reversal
 * reports its own row and nothing else, because the list it walked was read
 * before its write and may already be behind on the other tasks.
 */
export type UndoOutcome = {
  binding: ProjectBinding | null;
  bindings?: ProjectBinding[];
  stale: boolean;
};

/**
 * Taking a pick back, without taking anyone else's choice with it.
 *
 * A project default belongs to the whole project, so between the pick and the
 * Undo another member may have set their own. Reversing blindly would put the
 * older value back over theirs, or delete their row outright. So the undo
 * reads the project's rows first and only reverses while the row for this
 * task is still the one the pick wrote. Otherwise it writes nothing and
 * reports what it read, for the surface to adopt and say so.
 */
export async function undoProjectDefault(
  store: BindingStore,
  projectId: string,
  task: InferenceTask,
  pick: PickRecord,
  previousModelId: string | null,
): Promise<UndoOutcome> {
  const bindings = await store.list(projectId);
  const current = bindings.find((binding) => binding.task === task) ?? null;
  // Cleared by somebody else, so there is nothing of ours left to take back.
  if (!current) return { binding: null, bindings, stale: true };
  const ours =
    current.model_id === pick.modelId &&
    (pick.bindingId === null || current.id === pick.bindingId);
  if (!ours) return { binding: current, bindings, stale: true };
  try {
    if (previousModelId) {
      const binding = await store.update(
        projectId,
        current.id,
        previousModelId,
      );
      return { binding, stale: false };
    }
    await store.remove(projectId, current.id);
    return { binding: null, stale: false };
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 404) throw err;
    // The row went between the read and the write, so somebody else is
    // changing this default right now. Leave it to them and report theirs.
    const fresh = await store.list(projectId);
    return {
      binding: fresh.find((binding) => binding.task === task) ?? null,
      bindings: fresh,
      stale: true,
    };
  }
}

/**
 * The project bindings behind the project page's default models card, behind
 * the Workflow menu's pickers, and behind every picker's preselection.
 *
 * `loading` and `loadFailed` are part of the answer, not bookkeeping: an
 * empty `bindings` after a failed read means "we do not know", and a surface
 * that draws it as "No default" tells the researcher something about their
 * project that nobody checked. Callers wait, or say they could not look.
 *
 * There is one `saving` for the whole project, not one per task: every
 * control the hook feeds is disabled while a write runs, so a second write
 * cannot start and two writes can never race. `savingTask` is for the label
 * on the row being written, nothing else.
 */
export function useProjectModelDefaults(
  projectId: string | undefined,
  store?: BindingStore,
) {
  const resolvedStore = useMemo(() => store ?? apiBindingStore(), [store]);
  const [bindings, setBindings] = useState<ProjectBinding[]>([]);
  const [loading, setLoading] = useState<boolean>(Boolean(projectId));
  const [loadFailed, setLoadFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingTask, setSavingTask] = useState<InferenceTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * Which load is newest. A write that lands while a list is still in flight
   * must win over it: without this the late list would overwrite the
   * just-saved default with the server rows it left behind. The winner owns
   * `loading` from then on, so an invalidated load never leaves it true.
   */
  const generationRef = useRef(0);
  /**
   * The project the hook is on right now. A write started for one project
   * must not land after the surface has moved to another: its answer is about
   * a project nobody is looking at.
   */
  const projectIdRef = useRef<string | undefined>(projectId);

  const load = useCallback(
    async (id: string, cancelled?: () => boolean): Promise<void> => {
      const generation = ++generationRef.current;
      const isCurrent = () =>
        !cancelled?.() && generationRef.current === generation;
      setLoading(true);
      try {
        // Wrapped so a store that answers with nothing (a test double, an old
        // stub) reads as an empty list rather than crashing the picker.
        const rows = await resolvedStore.list(id);
        if (!isCurrent()) return;
        setBindings(Array.isArray(rows) ? rows : []);
        setLoadFailed(false);
        setLoading(false);
      } catch {
        if (!isCurrent()) return;
        setBindings([]);
        setLoadFailed(true);
        setLoading(false);
      }
    },
    [resolvedStore],
  );

  useEffect(() => {
    projectIdRef.current = projectId;
    // Nothing read from the last project survives the change of project, not
    // its bindings and not a write it had in flight.
    setBindings([]);
    setLoadFailed(false);
    setSaving(false);
    setSavingTask(null);
    setError(null);
    if (!projectId) {
      generationRef.current += 1;
      setLoading(false);
      return;
    }
    let cancelled = false;
    void load(projectId, () => cancelled);
    return () => {
      cancelled = true;
    };
  }, [projectId, load]);

  const defaultModelId = useCallback(
    (task: InferenceTask): string | null =>
      bindings.find((binding) => binding.task === task)?.model_id ?? null,
    [bindings],
  );

  /**
   * Re-read the project bindings, for the card's "Try again". A late read
   * never wins over newer state: every read carries the generation guard.
   */
  const refresh = useCallback(async (): Promise<void> => {
    if (!projectId) return;
    await load(projectId);
  }, [projectId, load]);

  /**
   * Runs one write and settles the hook's state on its answer. A write whose
   * project is no longer the hook's changes nothing at all.
   */
  const runWrite = useCallback(
    async (
      task: InferenceTask,
      write: (id: string) => Promise<WriteOutcome>,
      failureMessage: string,
    ): Promise<ProjectDefaultResult> => {
      const writeProjectId = projectId;
      if (!writeProjectId) return { ok: false, message: "No project" };
      setSaving(true);
      setSavingTask(task);
      setError(null);
      const settle = () => {
        if (projectIdRef.current !== writeProjectId) return false;
        setSaving(false);
        setSavingTask(null);
        return true;
      };
      try {
        const outcome = await write(writeProjectId);
        if (!settle()) return { ok: false, superseded: true };
        // The write outranks a list still in flight, and owns `loading` from
        // here: see generationRef.
        generationRef.current += 1;
        if (outcome.adopt) {
          // Rows read after somebody else's change, so newer than the state.
          setBindings(outcome.adopt);
        } else {
          // Only the written task moves. The other tasks belong to the state,
          // which may have learned a newer row while this write was in flight.
          setBindings((current) => [
            ...current.filter((binding) => binding.task !== task),
            ...(outcome.binding ? [outcome.binding] : []),
          ]);
        }
        setLoadFailed(false);
        setLoading(false);
        return {
          ok: true,
          stale: outcome.stale ?? false,
          binding: outcome.binding,
        };
      } catch (err) {
        if (!settle()) return { ok: false, superseded: true };
        const message = err instanceof Error ? err.message : failureMessage;
        setError(message);
        return { ok: false, message };
      }
    },
    [projectId],
  );

  const saveDefault = useCallback(
    (task: InferenceTask, modelId: string): Promise<ProjectDefaultResult> =>
      runWrite(
        task,
        async (id) => ({
          binding: await saveProjectDefault(resolvedStore, id, task, modelId),
        }),
        "Could not save the project default.",
      ),
    [runWrite, resolvedStore],
  );

  const clearDefault = useCallback(
    (task: InferenceTask): Promise<ProjectDefaultResult> =>
      runWrite(
        task,
        async (id) => {
          await clearProjectDefault(resolvedStore, id, task);
          return { binding: null };
        },
        "Could not clear the project default.",
      ),
    [runWrite, resolvedStore],
  );

  /**
   * Take one pick back, but only while the project still holds it: see
   * `undoProjectDefault`. A `stale` answer means somebody else's choice is
   * there now and was left alone.
   */
  const undoDefault = useCallback(
    (
      task: InferenceTask,
      pick: PickRecord,
      previousModelId: string | null,
    ): Promise<ProjectDefaultResult> =>
      runWrite(
        task,
        async (id) => {
          const outcome = await undoProjectDefault(
            resolvedStore,
            id,
            task,
            pick,
            previousModelId,
          );
          // `bindings` comes with a stale answer only, which is the one case
          // where the write knows the other tasks better than the state does.
          return {
            binding: outcome.binding,
            adopt: outcome.bindings,
            stale: outcome.stale,
          };
        },
        "Could not undo the project default.",
      ),
    [runWrite, resolvedStore],
  );

  return {
    bindings,
    loading,
    loadFailed,
    /** True once a list came back, so a surface may speak about the defaults. */
    known: Boolean(projectId) && !loading && !loadFailed,
    /** One write at a time, so this disables every control the hook feeds. */
    saving,
    /** Which row that write is about, for its own label. */
    savingTask,
    error,
    defaultModelId,
    saveDefault,
    clearDefault,
    undoDefault,
    refresh,
  };
}
