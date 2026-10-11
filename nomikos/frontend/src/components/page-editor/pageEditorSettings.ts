const STORAGE_KEY = "nomikos_page_editor_settings";

/**
 * Multiplier on the wheel zoom rate (see canvasZoom.ts). At 1× one mouse-wheel
 * notch zooms by about 20%, and a trackpad glides in small steps; the range
 * covers a quarter of that to three times it.
 */
export const WHEEL_ZOOM_SPEED_MIN = 0.25;
export const WHEEL_ZOOM_SPEED_MAX = 3;

export type PageEditorCanvasSettings = {
  /** Multiplier for segment/block overlay stroke width (0.5-4). */
  overlayStrokeWidth: number;
  /** Multiplier for Kraken/layout baseline stroke width (0.25-2.5). */
  baselineStrokeWidth: number;
  /** Segment polygon fill strength (0-0.35). */
  segmentFillOpacity: number;
  /** Multiplier for polygon corner handles (0.4-2.5). */
  handleSize: number;
  showLayoutBlocks: boolean;
  showBaselines: boolean;
  /** Multiplier on how far one wheel notch or trackpad step zooms (0.25-3). */
  wheelZoomSpeed: number;
  /** Show the text panel next to the canvas in a side-by-side layout. */
  sideBySide: boolean;
  /** Width share of the manuscript pane, 0.2 to 0.8. */
  splitRatio: number;
};

export const DEFAULT_PAGE_EDITOR_SETTINGS: PageEditorCanvasSettings = {
  overlayStrokeWidth: 1.25,
  baselineStrokeWidth: 0.75,
  segmentFillOpacity: 0.1,
  handleSize: 0.75,
  showLayoutBlocks: true,
  showBaselines: false,
  wheelZoomSpeed: 1,
  sideBySide: true,
  splitRatio: 0.55,
};

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  return typeof value === "number" && value >= min && value <= max
    ? value
    : fallback;
}

export function loadPageEditorSettings(): PageEditorCanvasSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PAGE_EDITOR_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<PageEditorCanvasSettings>;
    return {
      overlayStrokeWidth: clampNumber(
        parsed.overlayStrokeWidth,
        0.5,
        4,
        DEFAULT_PAGE_EDITOR_SETTINGS.overlayStrokeWidth,
      ),
      baselineStrokeWidth: clampNumber(
        parsed.baselineStrokeWidth,
        0.25,
        2.5,
        DEFAULT_PAGE_EDITOR_SETTINGS.baselineStrokeWidth,
      ),
      segmentFillOpacity: clampNumber(
        parsed.segmentFillOpacity,
        0,
        0.35,
        DEFAULT_PAGE_EDITOR_SETTINGS.segmentFillOpacity,
      ),
      handleSize: clampNumber(
        parsed.handleSize,
        0.4,
        2.5,
        DEFAULT_PAGE_EDITOR_SETTINGS.handleSize,
      ),
      showLayoutBlocks:
        typeof parsed.showLayoutBlocks === "boolean"
          ? parsed.showLayoutBlocks
          : DEFAULT_PAGE_EDITOR_SETTINGS.showLayoutBlocks,
      showBaselines:
        typeof parsed.showBaselines === "boolean"
          ? parsed.showBaselines
          : DEFAULT_PAGE_EDITOR_SETTINGS.showBaselines,
      wheelZoomSpeed: clampNumber(
        parsed.wheelZoomSpeed,
        WHEEL_ZOOM_SPEED_MIN,
        WHEEL_ZOOM_SPEED_MAX,
        DEFAULT_PAGE_EDITOR_SETTINGS.wheelZoomSpeed,
      ),
      sideBySide:
        typeof parsed.sideBySide === "boolean"
          ? parsed.sideBySide
          : DEFAULT_PAGE_EDITOR_SETTINGS.sideBySide,
      splitRatio:
        typeof parsed.splitRatio === "number" &&
        Number.isFinite(parsed.splitRatio)
          ? Math.min(0.8, Math.max(0.2, parsed.splitRatio))
          : DEFAULT_PAGE_EDITOR_SETTINGS.splitRatio,
    };
  } catch {
    return DEFAULT_PAGE_EDITOR_SETTINGS;
  }
}

export function savePageEditorSettings(
  settings: PageEditorCanvasSettings,
): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

/**
 * Whether the page rail is open, stored apart from the canvas settings.
 *
 * It is a layout choice rather than a rendering one, and it is toggled from the
 * toolbar rather than from the settings panel; folding it into the settings
 * object would make every rail toggle rewrite the whole canvas record.
 */
const PAGE_RAIL_KEY = "nomikos_page_editor_rail_open";

export function loadPageRailOpen(): boolean {
  try {
    // Open is the default: a researcher who has never touched the toggle
    // should find out the document is reachable from here without looking.
    return localStorage.getItem(PAGE_RAIL_KEY) !== "false";
  } catch {
    return true;
  }
}

export function savePageRailOpen(open: boolean): void {
  try {
    localStorage.setItem(PAGE_RAIL_KEY, String(open));
  } catch {
    // A browser refusing storage is not a reason to refuse the toggle.
  }
}
