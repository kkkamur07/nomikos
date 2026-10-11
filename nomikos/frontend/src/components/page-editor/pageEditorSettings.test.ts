import { beforeEach, describe, expect, it } from "vitest";

import {
  DEFAULT_PAGE_EDITOR_SETTINGS,
  loadPageEditorSettings,
  savePageEditorSettings,
} from "./pageEditorSettings";

describe("pageEditorSettings", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("falls back to the defaults when nothing is stored", () => {
    expect(loadPageEditorSettings()).toEqual(DEFAULT_PAGE_EDITOR_SETTINGS);
  });

  it("round-trips a saved wheel zoom speed", () => {
    savePageEditorSettings({
      ...DEFAULT_PAGE_EDITOR_SETTINGS,
      wheelZoomSpeed: 1.5,
    });
    expect(loadPageEditorSettings().wheelZoomSpeed).toBe(1.5);
  });

  it("shows the side-by-side text panel by default", () => {
    expect(DEFAULT_PAGE_EDITOR_SETTINGS.sideBySide).toBe(true);
    expect(loadPageEditorSettings().sideBySide).toBe(true);
  });

  it("round-trips a saved side-by-side choice", () => {
    savePageEditorSettings({
      ...DEFAULT_PAGE_EDITOR_SETTINGS,
      sideBySide: false,
    });
    expect(loadPageEditorSettings().sideBySide).toBe(false);
  });

  it("keeps settings saved before the side-by-side toggle existed", () => {
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ overlayStrokeWidth: 2 }),
    );
    expect(loadPageEditorSettings().sideBySide).toBe(true);
  });

  it("keeps settings saved before the wheel zoom speed existed", () => {
    // A profile written by an older build has no wheelZoomSpeed key at all.
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ overlayStrokeWidth: 2, showBaselines: true }),
    );
    const loaded = loadPageEditorSettings();
    expect(loaded.overlayStrokeWidth).toBe(2);
    expect(loaded.showBaselines).toBe(true);
    expect(loaded.wheelZoomSpeed).toBe(
      DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
    );
  });

  it.each([0, 4, "1"])(
    "ignores an out-of-range or non-numeric wheel zoom speed (%j)",
    (value) => {
      localStorage.setItem(
        "nomikos_page_editor_settings",
        JSON.stringify({ wheelZoomSpeed: value }),
      );
      expect(loadPageEditorSettings().wheelZoomSpeed).toBe(
        DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
      );
    },
  );

  it("defaults the split ratio to 0.55", () => {
    expect(DEFAULT_PAGE_EDITOR_SETTINGS.splitRatio).toBe(0.55);
    expect(loadPageEditorSettings().splitRatio).toBe(0.55);
  });

  it("round-trips a saved split ratio", () => {
    savePageEditorSettings({
      ...DEFAULT_PAGE_EDITOR_SETTINGS,
      splitRatio: 0.65,
    });
    expect(loadPageEditorSettings().splitRatio).toBe(0.65);
  });

  it.each([0.05, 0.95])(
    "clamps an out-of-range split ratio (%j) back to the range",
    (value) => {
      localStorage.setItem(
        "nomikos_page_editor_settings",
        JSON.stringify({ splitRatio: value }),
      );
      const loaded = loadPageEditorSettings().splitRatio;
      expect(loaded).toBeGreaterThanOrEqual(0.2);
      expect(loaded).toBeLessThanOrEqual(0.8);
    },
  );

  it("clamps 0.05 to 0.2 and 0.95 to 0.8", () => {
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ splitRatio: 0.05 }),
    );
    expect(loadPageEditorSettings().splitRatio).toBe(0.2);
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ splitRatio: 0.95 }),
    );
    expect(loadPageEditorSettings().splitRatio).toBe(0.8);
  });

  it("keeps a stored profile without the split ratio key on the default", () => {
    localStorage.setItem(
      "nomikos_page_editor_settings",
      JSON.stringify({ overlayStrokeWidth: 2 }),
    );
    expect(loadPageEditorSettings().splitRatio).toBe(
      DEFAULT_PAGE_EDITOR_SETTINGS.splitRatio,
    );
  });
});
