import { useEffect, useState } from "react";
import {
  api,
  type DocumentWorkflowCounts,
  type InferenceModelResponse,
  type InferenceTask,
} from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  ActionMenu,
  ActionMenuConfirm,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { PageEditorModelSelect } from "../page-editor/PageEditorModelSelect";
import {
  useProjectModelDefaults,
  type PickRecord,
} from "../page-editor/projectModelDefaults";
import { resolveSegmentModelId } from "../page-editor/segmentModelChoice";
import { resolveTranscribeModelId } from "../page-editor/transcribeModelChoice";
import { batchQueuedMessage, pageCountLabel } from "./documentActionCopy";

/** What one pick replaced, kept only as long as its Undo is on offer. */
type UndoEntry = {
  /** The binding the project had before the pick, or null when it had none. */
  storedModelId: string | null;
  /** What the picker showed before the pick, for a project with no binding. */
  runModelId: string | null;
  /** What the pick wrote, so the undo can recognise its own row later. */
  pick: PickRecord;
};

/** Said once, when an undo finds somebody else's choice where the pick was. */
const UNDO_STALE_MESSAGE =
  "The project default was changed in the meantime, nothing was undone.";

/** The accessible name of each section's select, for putting focus back. */
const MODEL_SELECT_LABEL: Record<InferenceTask, string> = {
  segment: "Segmentation model",
  transcribe: "HTR transcription model",
  binarize: "Binarization model",
};

type DocumentWorkflowMenuProps = {
  projectId: string;
  documentId: string;
  /** Null while the counts are still loading, which disables every item. */
  counts: DocumentWorkflowCounts | null;
  disabled?: boolean;
  /** Jobs were queued, so the counts and the parts list are both behind. */
  onJobsQueued: () => void;
};

/**
 * Segment and transcribe, for the whole document at once.
 *
 * These are document-level acts: they change every page, so they belong on the
 * document, not on whichever page happens to be open in the editor. The page
 * editor keeps its own single-page versions of the same two jobs.
 */
export function DocumentWorkflowMenu({
  projectId,
  documentId,
  counts,
  disabled = false,
  onJobsQueued,
}: DocumentWorkflowMenuProps) {
  const [confirmingResegment, setConfirmingResegment] = useState(false);
  const [running, setRunning] = useState(false);
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  /**
   * The researcher's own pick, when they made one. Null means "no explicit
   * choice": the selection below follows the project binding, then the
   * canonical row (segment) or the first row, so a saved default moves every
   * picker that nobody overrode.
   */
  const [explicitSegmentModelId, setExplicitSegmentModelId] = useState<
    string | null
  >(null);
  const [explicitTranscribeModelId, setExplicitTranscribeModelId] = useState<
    string | null
  >(null);
  /**
   * What the last pick in each section replaced, while its Undo is on offer.
   */
  const [undoable, setUndoable] = useState<
    Partial<Record<InferenceTask, UndoEntry>>
  >({});
  const projectDefaults = useProjectModelDefaults(projectId);

  /**
   * Always one of the catalog ids once the catalog loads; null only while
   * the list is empty or failed to load, when `model_id: null` is sent and
   * the backend resolves its own default.
   */
  const selectedSegmentModelId = resolveSegmentModelId(
    segmentModels,
    explicitSegmentModelId,
    projectDefaults.defaultModelId("segment"),
  );
  const selectedTranscribeModelId = resolveTranscribeModelId(
    transcribeModels,
    explicitTranscribeModelId,
    projectDefaults.defaultModelId("transcribe"),
  );

  // The catalog for both pickers. A failed request leaves empty lists and
  // does not block queueing: the requests then omit `model_id`.
  useEffect(() => {
    let cancelled = false;
    setExplicitSegmentModelId(null);
    setExplicitTranscribeModelId(null);
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (!cancelled) {
          setSegmentModels(catalog.filter((model) => model.task === "segment"));
          setTranscribeModels(
            catalog.filter((model) => model.task === "transcribe"),
          );
        }
      } catch {
        if (!cancelled) {
          setSegmentModels([]);
          setTranscribeModels([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, documentId]);

  const total = counts?.total ?? 0;
  const unsegmented = counts?.unsegmented ?? 0;
  const unpaired = counts?.unpaired ?? 0;
  const busy = disabled || running || counts === null;
  const selectedSegmentModelName =
    segmentModels.find((model) => model.id === selectedSegmentModelId)?.name ??
    null;

  /**
   * Picking a model here sets the project default, at once and with no
   * button: the same binding the project page's card writes, through the same
   * hook. Until the bindings are known, the pick still chooses the model for
   * this run but writes nothing, because a write would be guessing at what it
   * replaces.
   */
  async function chooseModel(
    task: InferenceTask,
    modelId: string | null,
    previousModelId: string | null,
    setExplicit: (modelId: string | null) => void,
  ) {
    setExplicit(modelId);
    if (!modelId || !projectDefaults.known) return;
    const storedModelId = projectDefaults.defaultModelId(task);
    const saved = await projectDefaults.saveDefault(task, modelId);
    restoreFocus(task);
    if (saved.superseded) return;
    if (saved.ok) {
      // What the pick replaced, so the line under it can offer the way back,
      // and what it wrote, so the way back knows its own row from a newer one.
      setUndoable((current) => ({
        ...current,
        [task]: {
          storedModelId,
          runModelId: previousModelId,
          pick: { modelId, bindingId: saved.binding?.id ?? null },
        },
      }));
      return;
    }
    forgetUndo(task);
    toast.error(saved.message);
    // Never leave the select showing a model the project did not save: back
    // to the stored default, or to the pick it replaced when there is none.
    setExplicit(projectDefaults.defaultModelId(task) ? null : previousModelId);
  }

  /**
   * Put the project back the way the pick found it: the binding it replaced,
   * or no binding at all when the project had none.
   *
   * Only while the pick is still what the project holds. This default belongs
   * to everyone in the project, and another member may have set their own
   * since; undoing over that would take their choice away without a word.
   */
  async function undoPick(
    task: InferenceTask,
    entry: UndoEntry,
    setExplicit: (modelId: string | null) => void,
  ) {
    const undone = await projectDefaults.undoDefault(
      task,
      entry.pick,
      entry.storedModelId,
    );
    restoreFocus(task);
    if (undone.superseded) return;
    // One undo per pick, whichever way it went: after this the line speaks
    // for the project again, not for what just happened.
    forgetUndo(task);
    if (!undone.ok) {
      toast.error(undone.message);
      setExplicit(
        projectDefaults.defaultModelId(task) ? null : entry.runModelId,
      );
      return;
    }
    if (undone.stale) {
      // Somebody else's choice stands. The picker follows the project again,
      // and falls back to the run choice when they cleared the default.
      toast.info(UNDO_STALE_MESSAGE);
      setExplicit(undone.binding ? null : entry.runModelId);
      return;
    }
    setExplicit(entry.storedModelId ? null : entry.runModelId);
  }

  function forgetUndo(task: InferenceTask) {
    setUndoable((current) => {
      if (!current[task]) return current;
      const next = { ...current };
      delete next[task];
      return next;
    });
  }

  /**
   * A write disables the picker and takes the Undo line away with it, so
   * whatever the person was standing on is gone by the time it lands. Put
   * them back on that section's select.
   */
  function restoreFocus(task: InferenceTask) {
    const select = globalThis.document.querySelector<HTMLSelectElement>(
      `select[aria-label="${MODEL_SELECT_LABEL[task]}"]`,
    );
    select?.focus();
  }

  /**
   * The one small line under a picker. It says only what is known: nothing
   * while the bindings are unread or unreadable, and nothing for a model that
   * is merely what the catalog offered first.
   */
  function defaultNote(task: InferenceTask, selectedModelId: string | null) {
    if (projectDefaults.savingTask === task) return "Saving…";
    if (!projectDefaults.known || !selectedModelId) return "";
    return projectDefaults.defaultModelId(task) === selectedModelId
      ? "Project default"
      : "";
  }

  /** The line under one picker: the undo offer if there is one, else the note. */
  function modelNote(
    task: InferenceTask,
    selectedModelId: string | null,
    setExplicit: (modelId: string | null) => void,
  ) {
    const entry = undoable[task];
    if (entry && projectDefaults.savingTask !== task) {
      return (
        <p className="action-menu__model-note">
          Saved as project default.{" "}
          <button
            type="button"
            className="action-menu__model-undo"
            disabled={projectDefaults.saving}
            onClick={() => void undoPick(task, entry, setExplicit)}
          >
            Undo
          </button>
        </p>
      );
    }
    return (
      <p className="action-menu__model-note">
        {defaultNote(task, selectedModelId)}
      </p>
    );
  }

  async function runSegment(scope: "unsegmented" | "all", close: () => void) {
    setRunning(true);
    try {
      const result = await api.enqueueDocumentSegment(projectId, documentId, {
        scope,
        model_id: selectedSegmentModelId,
      });
      toast.success(batchQueuedMessage(result));
      onJobsQueued();
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not queue segmentation",
      );
    } finally {
      setRunning(false);
      setConfirmingResegment(false);
    }
  }

  async function runTranscribe(close: () => void) {
    setRunning(true);
    try {
      const result = await api.enqueueDocumentTranscribe(
        projectId,
        documentId,
        { scope: "unpaired", model_id: selectedTranscribeModelId },
      );
      toast.success(batchQueuedMessage(result));
      onJobsQueued();
      close();
    } catch (err) {
      toast.error(
        err instanceof ApiError ? err.message : "Could not queue transcription",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <ActionMenu
      label="Workflow"
      menuLabel="Document workflow"
      wide
      onOpenChange={(open) => {
        if (open) return;
        setConfirmingResegment(false);
        // The offer belongs to the menu that was open, not to the next one.
        setUndoable({});
      }}
    >
      {(close) =>
        confirmingResegment ? (
          <ActionMenuConfirm
            destructive
            busy={running}
            question={`Re-segment every page (${pageCountLabel(total)})?`}
            detail={
              selectedSegmentModelName
                ? `Lines with approved text, a pairing or hand-drawn geometry stay. Every other line is redrawn and the model's unapproved text on it is discarded. There is no undo. Runs with ${selectedSegmentModelName}.`
                : "Lines with approved text, a pairing or hand-drawn geometry stay. Every other line is redrawn and the model's unapproved text on it is discarded. There is no undo."
            }
            confirmLabel={
              running ? "Queueing…" : `Yes, re-segment ${pageCountLabel(total)}`
            }
            onCancel={() => setConfirmingResegment(false)}
            onConfirm={() => void runSegment("all", close)}
          />
        ) : (
          <>
            <ActionMenuSection>Segment</ActionMenuSection>
            <div
              className="action-menu__model"
              role="group"
              aria-label="Segment"
            >
              <PageEditorModelSelect
                label="Model"
                ariaLabel={MODEL_SELECT_LABEL.segment}
                models={segmentModels}
                selectedModelId={selectedSegmentModelId}
                onSelectedModelIdChange={(modelId) =>
                  void chooseModel(
                    "segment",
                    modelId,
                    selectedSegmentModelId,
                    setExplicitSegmentModelId,
                  )
                }
                disabled={busy || projectDefaults.saving}
              />
              {modelNote(
                "segment",
                selectedSegmentModelId,
                setExplicitSegmentModelId,
              )}
            </div>
            <ActionMenuItem
              label="Segment unsegmented pages"
              meta={pageCountLabel(unsegmented)}
              quietMeta
              disabled={busy || unsegmented === 0}
              onSelect={() => void runSegment("unsegmented", close)}
            />
            <ActionMenuItem
              destructive
              label="Re-segment every page"
              detail="Discards unapproved machine text on untouched lines."
              meta={pageCountLabel(total)}
              quietMeta
              disabled={busy || total === 0}
              onSelect={() => setConfirmingResegment(true)}
            />
            <ActionMenuDivider />
            <ActionMenuSection>Transcribe</ActionMenuSection>
            <div
              className="action-menu__model"
              role="group"
              aria-label="Transcribe"
            >
              <PageEditorModelSelect
                label="Model"
                ariaLabel={MODEL_SELECT_LABEL.transcribe}
                models={transcribeModels}
                selectedModelId={selectedTranscribeModelId}
                onSelectedModelIdChange={(modelId) =>
                  void chooseModel(
                    "transcribe",
                    modelId,
                    selectedTranscribeModelId,
                    setExplicitTranscribeModelId,
                  )
                }
                disabled={busy || projectDefaults.saving}
              />
              {modelNote(
                "transcribe",
                selectedTranscribeModelId,
                setExplicitTranscribeModelId,
              )}
            </div>
            <ActionMenuItem
              label="Transcribe unpaired pages"
              meta={pageCountLabel(unpaired)}
              quietMeta
              disabled={busy || unpaired === 0}
              onSelect={() => void runTranscribe(close)}
            />
          </>
        )
      }
    </ActionMenu>
  );
}
