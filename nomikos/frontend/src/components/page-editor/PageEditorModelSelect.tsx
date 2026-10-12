import type { InferenceModelResponse } from "../../api/client";

type PageEditorModelSelectProps = {
  /** Short visible label, "HTR" or "Seg". */
  label: string;
  /** Accessible name for the select, "HTR transcription model" or similar. */
  ariaLabel: string;
  models: InferenceModelResponse[];
  selectedModelId: string | null;
  onSelectedModelIdChange: (modelId: string | null) => void;
  disabled?: boolean;
};

export function PageEditorModelSelect({
  label,
  ariaLabel,
  models,
  selectedModelId,
  onSelectedModelIdChange,
  disabled = false,
}: PageEditorModelSelectProps) {
  return (
    <label className="pe-model">
      <span className="pe-model__label">{label}</span>
      <select
        className="pe-model__select"
        aria-label={ariaLabel}
        value={selectedModelId ?? ""}
        disabled={disabled || models.length === 0}
        onChange={(event) =>
          onSelectedModelIdChange(event.target.value || null)
        }
      >
        {models.length === 0 ? (
          <option value="">No models</option>
        ) : (
          models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))
        )}
      </select>
    </label>
  );
}
