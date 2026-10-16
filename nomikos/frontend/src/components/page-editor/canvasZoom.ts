/**
 * Wheel zoom maths for the page canvas.
 *
 * react-zoom-pan-pinch's own wheel zoom adds `smoothStep * |deltaY|` to the
 * scale per event and applies it instantly: one mouse notch jumps the view by
 * a fixed amount that feels huge at 50% and tiny at 500%, and the jump itself
 * is a hard cut. Here every notch multiplies the scale by the same ratio, and
 * the view glides to the target over a few frames, anchored to the cursor.
 */

/** Natural-log scale change per wheel delta unit at speed 1×. */
export const WHEEL_ZOOM_RATE = 0.0015;
/** Natural-log scale change per delta unit for a trackpad pinch (ctrl+wheel). */
export const PINCH_ZOOM_RATE = 0.01;
/** The largest delta one event may contribute; driver flings exceed a notch. */
export const MAX_WHEEL_DELTA = 250;
/** Time constant, in ms, of the glide toward the target scale. */
export const ZOOM_GLIDE_TAU_MS = 50;
/** Remaining log distance below which the glide snaps to its target. */
const ZOOM_SETTLE = 0.0005;

export function clampScale(scale: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, scale));
}

/** Wheel delta in pixels for any deltaMode (0 px, 1 lines, 2 pages). */
export function wheelDeltaPixels(deltaY: number, deltaMode: number): number {
  const pixels =
    deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY;
  return Math.max(-MAX_WHEEL_DELTA, Math.min(MAX_WHEEL_DELTA, pixels));
}

/**
 * The scale a wheel event asks for: exponential in the delta so that each notch
 * is the same ratio at any zoom level, clamped to the scale limits.
 */
export function wheelTargetScale(
  currentTarget: number,
  deltaPixels: number,
  rate: number,
  minScale: number,
  maxScale: number,
): number {
  return clampScale(
    currentTarget * Math.exp(-deltaPixels * rate),
    minScale,
    maxScale,
  );
}

/**
 * One animation frame of the glide: cover the fraction `1 - exp(-dt / tau)` of
 * the remaining distance in log space, and arrive exactly once close enough.
 */
export function nextZoomFrameScale(
  scale: number,
  target: number,
  dtMs: number,
): number {
  const remaining = Math.log(target / scale);
  if (Math.abs(remaining) < ZOOM_SETTLE) return target;
  const fraction = 1 - Math.exp(-Math.max(dtMs, 0) / ZOOM_GLIDE_TAU_MS);
  const next = scale * Math.exp(remaining * fraction);
  return Math.abs(Math.log(target / next)) < ZOOM_SETTLE ? target : next;
}

/**
 * Content position that keeps the image point under `cursor` (relative to the
 * wrapper's top-left) fixed while the scale changes from `scale` to `nextScale`.
 */
export function zoomAnchoredPosition(
  position: { x: number; y: number },
  scale: number,
  nextScale: number,
  cursor: { x: number; y: number },
): { x: number; y: number } {
  const ratio = nextScale / scale;
  return {
    x: cursor.x - (cursor.x - position.x) * ratio,
    y: cursor.y - (cursor.y - position.y) * ratio,
  };
}

/**
 * The transform that fits the whole folio into the visible canvas.
 *
 * Scales to the pane and keeps `reservedBottom` (the island's own strip) out
 * of the usable area, because the last line of a folio matters as much as the
 * first and must not land under the toolbar. Null when either size is unknown.
 */
export function fitTransform(
  view: { width: number; height: number },
  canvas: { width: number; height: number },
  limits: {
    padding: number;
    reservedBottom: number;
    minScale: number;
    maxScale: number;
  },
): { x: number; y: number; scale: number } | null {
  if (!view.width || !view.height || !canvas.width || !canvas.height) {
    return null;
  }
  const usableWidth = Math.max(1, view.width - limits.padding * 2);
  const usableHeight = Math.max(
    1,
    view.height - limits.reservedBottom - limits.padding,
  );
  const scale = clampScale(
    Math.min(usableWidth / canvas.width, usableHeight / canvas.height),
    limits.minScale,
    limits.maxScale,
  );
  return {
    x: (view.width - canvas.width * scale) / 2,
    y: (view.height - limits.reservedBottom - canvas.height * scale) / 2,
    scale,
  };
}
