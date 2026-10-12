export type ModelSelectOption = {
  id: string;
  name: string;
};

type ModelSelectRowProps = {
  /** Ties the label to the select, so it must be unique on the page. */
  id: string;
  label: string;
  /** A model id, or the empty string for the first option. */
  value: string;
  options: ModelSelectOption[];
  /** The first option: what "no model chosen" reads as at this call site. */
  emptyLabel: string;
  disabled?: boolean;
  /** Draws a quiet note at the row's right without moving the select. */
  saving?: boolean;
  onChange: (value: string) => void;
};

/**
 * One labelled model select.
 *
 * The project page's default models card and the create-project dialog both
 * render this, so the rows are the same row in both places: one label column,
 * one select width, one gap. An empty catalog disables the select by itself,
 * because there is then nothing to choose.
 */
export function ModelSelectRow({
  id,
  label,
  value,
  options,
  emptyLabel,
  disabled = false,
  saving = false,
  onChange,
}: ModelSelectRowProps) {
  return (
    <div className="model-row">
      <label className="model-row__label" htmlFor={id}>
        {label}
      </label>
      <div className="model-row__control">
        <select
          id={id}
          className="model-row__select"
          value={value}
          disabled={disabled || options.length === 0}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">{emptyLabel}</option>
          {options.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
        {saving && (
          <span className="model-row__saving" role="status">
            Saving…
          </span>
        )}
      </div>
    </div>
  );
}
