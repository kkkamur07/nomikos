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
      <div className="pe-dd-field pe-dd-field--stack">
        <label htmlFor="pe-wheel-zoom-speed">
          Scroll zoom speed{" "}
          <strong>{settings.wheelZoomSpeed.toFixed(2)}×</strong>
        </label>
        <input
          id="pe-wheel-zoom-speed"
          type="range"
          min={WHEEL_ZOOM_SPEED_MIN}
          max={WHEEL_ZOOM_SPEED_MAX}
          step={0.25}
          value={settings.wheelZoomSpeed}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              wheelZoomSpeed: Number(event.target.value),
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        <div className="pe-dd-range-labels" aria-hidden="true">
          <span>Precise</span>
          <span>Fast</span>
        </div>
      </div>

      <div className="pe-dd-divider" />

      <div className="pe-dd-section">Canvas overlays</div>
      <p className="pe-dd-model">
        Stroke widths stay consistent while zooming. Baselines from Kraken
        layout use their own control.
      </p>
      <div className="pe-dd-field pe-dd-field--stack">
        <label htmlFor="pe-stroke-width">
          Segment stroke{" "}
          <strong>{settings.overlayStrokeWidth.toFixed(1)}×</strong>
        </label>
        <input
          id="pe-stroke-width"
          type="range"
          min={0.5}
          max={4}
          step={0.25}
          value={settings.overlayStrokeWidth}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              overlayStrokeWidth: Number(event.target.value),
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        <div className="pe-dd-range-labels" aria-hidden="true">
          <span>Thin</span>
          <span>Thick</span>
        </div>
      </div>
      <div className="pe-dd-field pe-dd-field--stack">
        <label htmlFor="pe-baseline-width">
          Baseline width{" "}
          <strong>{settings.baselineStrokeWidth.toFixed(2)}×</strong>
        </label>
        <input
          id="pe-baseline-width"
          type="range"
          min={0.25}
          max={2.5}
          step={0.05}
          value={settings.baselineStrokeWidth}
          disabled={!settings.showBaselines}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              baselineStrokeWidth: Number(event.target.value),
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        <div className="pe-dd-range-labels" aria-hidden="true">
          <span>Thin</span>
          <span>Thick</span>
        </div>
      </div>
      <div className="pe-dd-field pe-dd-field--stack">
        <label htmlFor="pe-segment-fill">
          Segment fill{" "}
          <strong>{Math.round(settings.segmentFillOpacity * 100)}%</strong>
        </label>
        <input
          id="pe-segment-fill"
          type="range"
          min={0}
          max={0.35}
          step={0.025}
          value={settings.segmentFillOpacity}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              segmentFillOpacity: Number(event.target.value),
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        <div className="pe-dd-range-labels" aria-hidden="true">
          <span>Clear</span>
          <span>Solid</span>
        </div>
      </div>

      <div className="pe-dd-field pe-dd-field--stack">
        <label htmlFor="pe-handle-size">
          Pointer size <strong>{settings.handleSize.toFixed(2)}×</strong>
        </label>
        <input
          id="pe-handle-size"
          type="range"
          min={0.4}
          max={2.5}
          step={0.05}
          value={settings.handleSize}
          onChange={(event) =>
            onSettingsChange({
              ...settings,
              handleSize: Number(event.target.value),
            })
          }
          onClick={(event) => event.stopPropagation()}
        />
        <div className="pe-dd-range-labels" aria-hidden="true">
          <span>Small</span>
          <span>Large</span>
        </div>
      </div>

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
        Show layout blocks
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

      <div className="pe-dd-divider" />

      <div className="pe-dd-section">Polygon tool</div>
      <p className="pe-dd-model">
        Click to place corners one at a time. Double-click or press{" "}
        <strong>Enter</strong> to close the shape. <strong>Esc</strong> cancels.
      </p>
    </div>
  );
}
