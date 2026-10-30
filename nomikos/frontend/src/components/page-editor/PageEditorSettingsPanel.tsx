import {
  HOST_PREFERENCE_HINT,
  HOST_PREFERENCE_LABEL,
} from "../../inference/hostPreference";
import {
  WHEEL_ZOOM_SPEED_MAX,
  WHEEL_ZOOM_SPEED_MIN,
  type PageEditorCanvasSettings,
} from "./pageEditorSettings";

type PageEditorSettingsPanelProps = {
  settings: PageEditorCanvasSettings;
  onSettingsChange: (settings: PageEditorCanvasSettings) => void;
  /** The account-level **host preference**, not a per-job choice. */
  preferLocalInference: boolean;
  onPreferLocalInferenceChange: (preferLocal: boolean) => void;
  preferenceSaving: boolean;
  /** **Capacity** for this account's own computer, as the platform reports it. */
  hasLocalCapacity: boolean;
};

type NumericSetting =
  | "wheelZoomSpeed"
  | "overlayStrokeWidth"
  | "baselineStrokeWidth"
  | "segmentFillOpacity"
  | "handleSize";

type SliderRowProps = {
  id: string;
  label: string;
  setting: NumericSetting;
  format: (value: number) => string;
  min: number;
  max: number;
  step: number;
  /** What the low and the high end of the slider mean. */
  ends: [string, string];
  disabled?: boolean;
  settings: PageEditorCanvasSettings;
  onSettingsChange: (settings: PageEditorCanvasSettings) => void;
};

function SliderRow({
  id,
  label,
  setting,
  format,
  min,
  max,
  step,
  ends,
  disabled,
  settings,
  onSettingsChange,
}: SliderRowProps) {
  return (
    <div className="pe-dd-field pe-dd-field--stack">
      <label htmlFor={id}>
        {label} <strong>{format(settings[setting])}</strong>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={settings[setting]}
        disabled={disabled}
        onChange={(event) =>
          onSettingsChange({
            ...settings,
            [setting]: Number(event.target.value),
          })
        }
        onClick={(event) => event.stopPropagation()}
      />
      <div className="pe-dd-range-labels" aria-hidden="true">
        <span>{ends[0]}</span>
        <span>{ends[1]}</span>
      </div>
    </div>
  );
}

export function PageEditorSettingsPanel({
  settings,
  onSettingsChange,
  preferLocalInference,
  onPreferLocalInferenceChange,
  preferenceSaving,
  hasLocalCapacity,
}: PageEditorSettingsPanelProps) {
  return (
    <div
      className="pe-dropdown pe-dropdown--settings"
      role="dialog"
      aria-label="Editor settings"
    >
      <div className="pe-dd-section">Inference</div>
      {/* One account-level setting, and no per-job toggle: a researcher cannot
          know which host is faster for a given page, so the choice is made once
          and each job then says which host ran it (ADR 0002). */}
      <label className="pe-dd-check">
        <input
          id="pe-prefer-local-inference"
          type="checkbox"
          checked={preferLocalInference}
          disabled={preferenceSaving}
          onChange={(event) =>
            onPreferLocalInferenceChange(event.target.checked)
          }
          onClick={(event) => event.stopPropagation()}
        />
        {HOST_PREFERENCE_LABEL}
      </label>
      <p className="pe-dd-model">
        {`${HOST_PREFERENCE_HINT}${
          !preferLocalInference || hasLocalCapacity
            ? ""
            : " Nothing is running on this computer right now, so jobs go to the cloud."
        }`}
      </p>

      <div className="pe-dd-section">Navigation</div>
      <p className="pe-dd-model">
        How far the page zooms per mouse-wheel notch or trackpad step. Zoom
        buttons and pinch are unaffected.
      </p>
      <SliderRow
        id="pe-wheel-zoom-speed"
        label="Scroll zoom speed"
        setting="wheelZoomSpeed"
        format={(value) => `${value.toFixed(2)}×`}
        min={WHEEL_ZOOM_SPEED_MIN}
        max={WHEEL_ZOOM_SPEED_MAX}
        step={0.25}
        ends={["Precise", "Fast"]}
        settings={settings}
        onSettingsChange={onSettingsChange}
      />

      <div className="pe-dd-divider" />

      <div className="pe-dd-section">Canvas overlays</div>
      <p className="pe-dd-model">
        Stroke widths stay consistent while zooming. Baselines from Kraken
        layout use their own control.
      </p>
      <SliderRow
        id="pe-stroke-width"
        label="Segment stroke"
        setting="overlayStrokeWidth"
        format={(value) => `${value.toFixed(1)}×`}
        min={0.5}
        max={4}
        step={0.25}
        ends={["Thin", "Thick"]}
        settings={settings}
        onSettingsChange={onSettingsChange}
      />
      <SliderRow
        id="pe-baseline-width"
        label="Baseline width"
        setting="baselineStrokeWidth"
        format={(value) => `${value.toFixed(2)}×`}
        min={0.25}
        max={2.5}
        step={0.05}
        ends={["Thin", "Thick"]}
        disabled={!settings.showBaselines}
        settings={settings}
        onSettingsChange={onSettingsChange}
      />
      <SliderRow
        id="pe-segment-fill"
        label="Segment fill"
        setting="segmentFillOpacity"
        format={(value) => `${Math.round(value * 100)}%`}
        min={0}
        max={0.35}
        step={0.025}
        ends={["Clear", "Solid"]}
        settings={settings}
        onSettingsChange={onSettingsChange}
      />

      <SliderRow
        id="pe-handle-size"
        label="Pointer size"
        setting="handleSize"
        format={(value) => `${value.toFixed(2)}×`}
        min={0.4}
        max={2.5}
        step={0.05}
        ends={["Small", "Large"]}
        settings={settings}
        onSettingsChange={onSettingsChange}
      />

      <div className="pe-dd-divider" />

      <div className="pe-dd-section">Visibility</div>
      <p className="pe-dd-model">
        Hide Kraken layout overlays when you only want segment polygons and
        transcription.
      </p>
      <label className="pe-dd-check">
        <input
          type="checkbox"
          checked={settings.showLayoutBlocks}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              showLayoutBlocks: event.target.checked,
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        Show regions
      </label>
      <label className="pe-dd-check">
        <input
          type="checkbox"
          checked={settings.showBaselines}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              showBaselines: event.target.checked,
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        Show line baselines (Kraken layout)
      </label>
      <label className="pe-dd-check">
        <input
          type="checkbox"
          checked={settings.sideBySide}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              sideBySide: event.target.checked,
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        Side by side text panel
      </label>
      <label className="pe-dd-check">
        <input
          type="checkbox"
          checked={settings.transcriptFontFromMask}
          disabled={!settings.sideBySide}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              transcriptFontFromMask: event.target.checked,
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        Size transcript text by segment height
      </label>

      <div className="pe-dd-divider" />

      <div className="pe-dd-section">Polygon tool</div>
      <p className="pe-dd-model">
        Click to place corners one at a time. Double-click or press{" "}
        <strong>Enter</strong> to close the shape. <strong>Esc</strong> cancels.
      </p>
    </div>
  );
}
