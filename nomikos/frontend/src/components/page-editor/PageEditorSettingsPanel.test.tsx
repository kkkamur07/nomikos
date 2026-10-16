import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PageEditorSettingsPanel } from "./PageEditorSettingsPanel";
import {
  DEFAULT_PAGE_EDITOR_SETTINGS,
  type PageEditorCanvasSettings,
} from "./pageEditorSettings";

function renderPanel(overrides: Partial<PageEditorCanvasSettings> = {}) {
  const onSettingsChange = vi.fn();
  render(
    <PageEditorSettingsPanel
      settings={{ ...DEFAULT_PAGE_EDITOR_SETTINGS, ...overrides }}
      onSettingsChange={onSettingsChange}
      preferLocalInference={false}
      onPreferLocalInferenceChange={() => {}}
      preferenceSaving={false}
      hasLocalCapacity={false}
    />,
  );
  return onSettingsChange;
}

describe("PageEditorSettingsPanel sliders", () => {
  it.each([
    ["Scroll zoom speed", "wheelZoomSpeed", "1.5"],
    ["Segment stroke", "overlayStrokeWidth", "2"],
    ["Baseline width", "baselineStrokeWidth", "1.2"],
    ["Segment fill", "segmentFillOpacity", "0.2"],
    ["Pointer size", "handleSize", "1.4"],
  ])("%s writes %s and keeps the rest", (label, setting, value) => {
    const onSettingsChange = renderPanel({ showBaselines: true });
    fireEvent.change(screen.getByLabelText(new RegExp(`^${label}`)), {
      target: { value },
    });
    expect(onSettingsChange).toHaveBeenCalledWith({
      ...DEFAULT_PAGE_EDITOR_SETTINGS,
      showBaselines: true,
      [setting]: Number(value),
    });
  });

  it("shows each value in its own unit", () => {
    renderPanel({
      overlayStrokeWidth: 1.5,
      baselineStrokeWidth: 0.75,
      segmentFillOpacity: 0.125,
      handleSize: 1,
      wheelZoomSpeed: 2,
    });
    expect(screen.getByLabelText(/^Segment stroke/)).toHaveAccessibleName(
      "Segment stroke 1.5×",
    );
    expect(screen.getByLabelText(/^Baseline width/)).toHaveAccessibleName(
      "Baseline width 0.75×",
    );
    expect(screen.getByLabelText(/^Segment fill/)).toHaveAccessibleName(
      "Segment fill 13%",
    );
    expect(screen.getByLabelText(/^Pointer size/)).toHaveAccessibleName(
      "Pointer size 1.00×",
    );
    expect(screen.getByLabelText(/^Scroll zoom speed/)).toHaveAccessibleName(
      "Scroll zoom speed 2.00×",
    );
  });

  it("locks the baseline width while baselines are hidden", () => {
    renderPanel({ showBaselines: false });
    expect(screen.getByLabelText(/^Baseline width/)).toBeDisabled();
    expect(screen.getByLabelText(/^Segment stroke/)).toBeEnabled();
  });
});
