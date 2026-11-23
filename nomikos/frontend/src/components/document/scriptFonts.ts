/**
 * The font choices a document can make per script.
 *
 * A mirror of `SCRIPT_FONTS` in nomikos/backend/core/fonts.py, which is the
 * source of truth: the API validates `script_fonts` against it and the PDF and
 * Word exports draw from it. The browser faces for these ids live in
 * src/styles/script-fonts.css. tests/nomikos/unit/test_script_fonts_frontend.py
 * fails when the three disagree, so adding a font is data in all three places.
 */
export type ScriptFontChoice = { id: string; label: string };

export type ScriptFontEntry = {
  defaultChoice: string;
  choices: readonly ScriptFontChoice[];
};

export const SCRIPT_FONTS: Readonly<Record<string, ScriptFontEntry>> = {
  latin: {
    defaultChoice: "noto-sans",
    choices: [{ id: "noto-sans", label: "Noto Sans" }],
  },
  greek: {
    defaultChoice: "noto-sans",
    choices: [{ id: "noto-sans", label: "Noto Sans" }],
  },
  coptic: {
    defaultChoice: "noto-sans-coptic",
    choices: [{ id: "noto-sans-coptic", label: "Noto Sans Coptic" }],
  },
  armenian: {
    defaultChoice: "noto-sans-armenian",
    choices: [{ id: "noto-sans-armenian", label: "Noto Sans Armenian" }],
  },
  syriac: {
    defaultChoice: "estrangela",
    choices: [
      { id: "estrangela", label: "Estrangela" },
      { id: "eastern", label: "East Syriac" },
      { id: "western", label: "West Syriac" },
    ],
  },
};

export type ScriptFonts = Record<string, string>;

/** Scripts with more than one bundled face: the only ones worth a control. */
export function choosableScripts(): string[] {
  return Object.keys(SCRIPT_FONTS).filter(
    (script) => SCRIPT_FONTS[script].choices.length > 1,
  );
}

export function effectiveChoice(
  script: string,
  scriptFonts: ScriptFonts | null | undefined,
): string | undefined {
  const entry = SCRIPT_FONTS[script];
  if (!entry) return undefined;
  const chosen = scriptFonts?.[script];
  return entry.choices.some((choice) => choice.id === chosen)
    ? chosen
    : entry.defaultChoice;
}

/**
 * The `data-font-<script>` attributes for a subtree root, from a document's
 * `script_fonts`. Only known, non-default choices are emitted: the defaults
 * already come from :root, and an unknown id must never reach the DOM.
 */
export function scriptFontAttributes(
  scriptFonts: ScriptFonts | null | undefined,
): Record<string, string> {
  const attributes: Record<string, string> = {};
  for (const script of Object.keys(SCRIPT_FONTS)) {
    const choice = effectiveChoice(script, scriptFonts);
    if (choice && choice !== SCRIPT_FONTS[script].defaultChoice) {
      attributes[`data-font-${script}`] = choice;
    }
  }
  return attributes;
}
