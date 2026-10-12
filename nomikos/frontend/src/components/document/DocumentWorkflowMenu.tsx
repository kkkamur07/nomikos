import { useEffect, useState } from "react";
import {
  api,
  type DocumentWorkflowCounts,
  type InferenceModelResponse,
} from "../../api/client";
import { ApiError } from "../../api/errors";
import {
  ActionMenu,
  ActionMenuCaption,
  ActionMenuConfirm,
  ActionMenuDivider,
  ActionMenuItem,
  ActionMenuSection,
  ActionMenuWarning,
} from "../ui/ActionMenu";
import { toast } from "../ui/toast";
import { PageEditorModelSelect } from "../page-editor/PageEditorModelSelect";
import { batchQueuedMessage, pageCountLabel } from "./documentActionCopy";

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
  /**
   * Null means "Default": `model_id: null` is sent and the backend resolves
   * the binding or the worker's own default, exactly as before this picker
   * existed.
   */
  const [selectedSegmentModelId, setSelectedSegmentModelId] = useState<
    string | null
  >(null);
  /**
   * The transcription model the batch route will resolve, by catalog name:
   * the document's binding, else the project's. Null when neither is set (or
   * the lookup failed): the backend then refuses a model-less transcribe, so
   * the menu offers the catalog instead and sends the chosen id.
   */
  const [transcribeModelName, setTranscribeModelName] = useState<string | null>(
    null,
  );
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const [selectedTranscribeModelId, setSelectedTranscribeModelId] = useState<
    string | null
  >(null);

  // The segment catalog for the picker. A failed request leaves only
  // "Default" and does not block segmenting.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (!cancelled) {
          setSegmentModels(catalog.filter((model) => model.task === "segment"));
          const htr = catalog.filter((model) => model.task === "transcribe");
          setTranscribeModels(htr);
          const remembered = readRememberedModel(projectId);
          setSelectedTranscribeModelId(
            htr.find((model) => model.id === remembered)?.id ??
              htr[0]?.id ??
              null,
          );
        }
        // The same order the backend resolves in for a whole document: a
        // document binding wins over the project one. Part bindings are per
        // page and are not named here.
        const [documentBindings, projectBindings] = await Promise.all([
          api.listDocumentModelBindings(projectId, documentId),
          api.listProjectModelBindings(projectId),
        ]);
        const binding = [...documentBindings, ...projectBindings].find(
          (candidate) => candidate.task === "transcribe",
        );
        const name =
          catalog.find((model) => model.id === binding?.model_id)?.name ?? null;
        if (!cancelled) setTranscribeModelName(name);
      } catch {
        if (!cancelled) setTranscribeModelName(null);
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
  function chooseTranscribeModel(modelId: string | null) {
    setSelectedTranscribeModelId(modelId);
    rememberModel(projectId, modelId);
  }

  const selectedSegmentModelName =
    segmentModels.find((model) => model.id === selectedSegmentModelId)?.name ??
    null;

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
        {
          scope: "unpaired",
          // A binding resolves on the server; otherwise the picker's choice.
          model_id: transcribeModelName ? null : selectedTranscribeModelId,
        },
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
        if (!open) setConfirmingResegment(false);
      }}
    >
      {(close) =>
        confirmingResegment ? (
          <ActionMenuConfirm
            destructive
            busy={running}
            question={`Re-segment every page (${pageCountLabel(total)})?`}
            detail={`Lines with approved text, a pairing or hand-drawn geometry stay. Every other line is redrawn and the model's unapproved text on it is discarded. There is no undo. Runs with ${selectedSegmentModelName ?? "the default model"}.`}
            confirmLabel={
              running ? "Queueing…" : `Yes, re-segment ${pageCountLabel(total)}`
            }
            onCancel={() => setConfirmingResegment(false)}
            onConfirm={() => void runSegment("all", close)}
          />
        ) : (
          <>
            <ActionMenuSection>Segment</ActionMenuSection>
            <PageEditorModelSelect
              label="Seg"
              ariaLabel="Segmentation model"
              models={segmentModels}
              selectedModelId={selectedSegmentModelId}
              onSelectedModelIdChange={setSelectedSegmentModelId}
              disabled={busy}
              includeDefaultOption
            />
            <ActionMenuItem
              label="Segment unsegmented pages"
              meta={String(unsegmented)}
              disabled={busy || unsegmented === 0}
              onSelect={() => void runSegment("unsegmented", close)}
            />
            <ActionMenuItem
              destructive
              label="Re-segment every page"
              meta={String(total)}
              disabled={busy || total === 0}
              onSelect={() => setConfirmingResegment(true)}
            />
            <ActionMenuWarning>
              Re-segmenting discards unapproved machine text on lines nobody has
              touched. Only the top item is safe.
            </ActionMenuWarning>
            <ActionMenuDivider />
            <ActionMenuSection>Transcribe</ActionMenuSection>
            {transcribeModelName ? (
              <ActionMenuCaption>
                Model <strong>{transcribeModelName}</strong>
              </ActionMenuCaption>
            ) : (
              <PageEditorModelSelect
                label="HTR"
                ariaLabel="Transcription model"
                models={transcribeModels}
                selectedModelId={selectedTranscribeModelId}
                onSelectedModelIdChange={chooseTranscribeModel}
                disabled={busy}
              />
            )}
            <ActionMenuItem
              label="Transcribe unpaired pages"
              meta={String(unpaired)}
              disabled={
                busy ||
                unpaired === 0 ||
                (!transcribeModelName && !selectedTranscribeModelId)
              }
              onSelect={() => void runTranscribe(close)}
            />
          </>
        )
      }
    </ActionMenu>
  );
}

/** The last transcription model picked for a project with no binding. */
function rememberedModelKey(projectId: string): string {
  return `nomikos.transcribeModel.${projectId}`;
}

function readRememberedModel(projectId: string): string | null {
  try {
    return window.localStorage.getItem(rememberedModelKey(projectId));
  } catch {
    return null;
  }
}

function rememberModel(projectId: string, modelId: string | null): void {
  try {
    if (modelId) {
      window.localStorage.setItem(rememberedModelKey(projectId), modelId);
    } else {
      window.localStorage.removeItem(rememberedModelKey(projectId));
    }
  } catch {
    // Private windows can refuse storage; the choice then lasts this visit.
  }
}
