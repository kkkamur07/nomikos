export type TextDirection = "ltr" | "rtl";

export type Script =
  "syriac" | "hebrew" | "arabic" | "greek" | "coptic" | "armenian" | "latin";

const RTL_SCRIPTS: ReadonlySet<Script> = new Set([
  "syriac",
  "hebrew",
  "arabic",
]);

/** Unicode blocks per script, as inclusive code point ranges. */
const SCRIPT_RANGES: readonly [number, number, Script][] = [
  [0x0041, 0x005a, "latin"],
  [0x0061, 0x007a, "latin"],
  [0x00c0, 0x024f, "latin"],
  [0x0370, 0x03e1, "greek"],
  [0x03e2, 0x03ef, "coptic"],
  [0x03f0, 0x03ff, "greek"],
  [0x0530, 0x058f, "armenian"],
  [0x0590, 0x05ff, "hebrew"],
  [0x0600, 0x06ff, "arabic"],
  [0x0700, 0x074f, "syriac"],
  [0x0750, 0x077f, "arabic"],
  [0x0860, 0x086f, "syriac"],
  [0x08a0, 0x08ff, "arabic"],
  [0x1f00, 0x1fff, "greek"],
  [0x2c80, 0x2cff, "coptic"],
  [0xfb1d, 0xfb4f, "hebrew"],
  [0xfb50, 0xfdff, "arabic"],
  [0xfe70, 0xfeff, "arabic"],
];

/** The script a code point belongs to, or null for digits, punctuation and marks. */
export function scriptOfCodePoint(codePoint: number): Script | null {
  for (const [start, end, script] of SCRIPT_RANGES) {
    if (codePoint >= start && codePoint <= end) return script;
  }
  return null;
}

/**
 * The script most letters of the text are written in, or null when the text
 * holds no letters of a known script. Accepts several texts so a whole page
 * can be weighed at once.
 */
export function dominantScript(
  texts: string | readonly string[],
): Script | null {
  const counts = new Map<Script, number>();
  for (const text of typeof texts === "string" ? [texts] : texts) {
    for (const char of text) {
      const script = scriptOfCodePoint(char.codePointAt(0) ?? 0);
      if (script !== null) counts.set(script, (counts.get(script) ?? 0) + 1);
    }
  }
  let best: Script | null = null;
  let bestCount = 0;
  for (const [script, count] of counts) {
    if (count > bestCount) {
      best = script;
      bestCount = count;
    }
  }
  return best;
}

export function directionOfScript(script: Script): TextDirection {
  return RTL_SCRIPTS.has(script) ? "rtl" : "ltr";
}

/**
 * The direction a text reads in, from its dominant script. Text without
 * letters (empty, digits only) takes the fallback.
 */
export function textDirection(
  texts: string | readonly string[],
  fallback: TextDirection = "ltr",
): TextDirection {
  const script = dominantScript(texts);
  return script === null ? fallback : directionOfScript(script);
}
