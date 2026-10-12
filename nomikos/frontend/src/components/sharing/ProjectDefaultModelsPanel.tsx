import { useEffect, useState } from "react";
import { api, type InferenceModelResponse } from "../../api/client";
import { useProjectModelDefaults } from "../page-editor/projectModelDefaults";
import { ModelSelectRow } from "../ui/ModelSelectRow";
import { toast } from "../ui/toast";

type ProjectDefaultModelsPanelProps = {
  projectId: string;
};

const NO_DEFAULT = "";

/** The two tasks a project binds a model for. Binarization has no picker. */
type DefaultTask = "segment" | "transcribe";

const TASK_NOUN: Record<DefaultTask, string> = {
  segment: "segmentation",
  transcribe: "transcription",
};

/**
 * The one place a project's default models are seen and set.
 *
 * Every picker in the app starts on these bindings, so the setting belongs
 * where the project itself is configured rather than beside each run button,
 * where it competed with the action and told each researcher a different
 * story about what "default" meant.
 */
export function ProjectDefaultModelsPanel({
  projectId,
}: ProjectDefaultModelsPanelProps) {
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const defaults = useProjectModelDefaults(projectId);

  // A failed catalog read leaves the rows empty and disabled rather than an
  // error banner: nothing here is lost by trying again later.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const catalog = await api.listInferenceModels();
        if (cancelled) return;
        setSegmentModels(catalog.filter((model) => model.task === "segment"));
        setTranscribeModels(
          catalog.filter((model) => model.task === "transcribe"),
        );
      } catch {
        if (cancelled) return;
        setSegmentModels([]);
        setTranscribeModels([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  async function handleChange(
    task: DefaultTask,
    models: InferenceModelResponse[],
    value: string,
  ) {
    const noun = TASK_NOUN[task];
    if (value === NO_DEFAULT) {
      const cleared = await defaults.clearDefault(task);
      // A superseded write is not this row's state any more; the newer one
      // speaks for it.
      if (cleared.superseded) return;
      if (cleared.ok) toast.success(`Default ${noun} model cleared`);
      else toast.error(cleared.message);
      return;
    }
    const saved = await defaults.saveDefault(task, value);
    if (saved.superseded) return;
    if (saved.ok) {
      const name = models.find((model) => model.id === value)?.name ?? value;
      toast.success(`Default ${noun} model set to ${name}`);
    } else {
      toast.error(saved.message);
    }
  }

  function row(
    task: DefaultTask,
    label: string,
    models: InferenceModelResponse[],
  ) {
    const saving = defaults.savingTask === task;
    return (
      <ModelSelectRow
        key={task}
        id={`project-default-${task}`}
        label={label}
        // The select always shows the stored binding, so a failed save snaps
        // back to what the project actually has.
        value={defaults.defaultModelId(task) ?? NO_DEFAULT}
        options={models}
        emptyLabel="No default"
        // One write at a time, so a save on either row freezes both. Until a
        // list of the bindings comes back, no value here would be the
        // project's; an unknown default must not be offered as "No default".
        disabled={defaults.saving || !defaults.known}
        saving={saving}
        onChange={(value) => void handleChange(task, models, value)}
      />
    );
  }

  return (
    <div className="entity-panel__section">
      <h2 className="entity-panel__heading">Default models</h2>
      <p className="entity-panel__hint">
        New segmentation and transcription jobs in this project start with these
        models. Anyone can still pick another model for a single run.
      </p>
      {defaults.loadFailed && (
        <p className="model-row-message">
          Could not load the project defaults.{" "}
          <button
            type="button"
            className="model-row-retry"
            onClick={() => void defaults.refresh()}
          >
            Try again
          </button>
        </p>
      )}
      {row("segment", "Segmentation", segmentModels)}
      {row("transcribe", "Transcription", transcribeModels)}
    </div>
  );
}
