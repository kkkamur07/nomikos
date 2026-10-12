import type {
  CharacterConfidence,
  LineTranscriptionResponse,
} from "../../api/client";

export type { CharacterConfidence };

/**
 * A transcription plus per-character scores. The platform API carries
 * `character_confidences` on `LineTranscriptionResponse` (null for
 * human-written rows and rows written before the column existed), so this is
 * a plain alias kept for its existing importers, declared once, here.
 */
export type LineTranscriptionWithCharacterConfidence =
  LineTranscriptionResponse;

export function confidenceTierClass(confidence: number): string {
  if (confidence > 0.9) return "ch-high";
  if (confidence > 0.5) return "ch-mid";
  return "ch-low";
}

/** Human-readable tier for aria / tooltips (matches strip legend). */
export function confidenceTierLabel(confidence: number): string {
  if (confidence > 0.9) return "high";
  if (confidence > 0.5) return "mid";
  return "low";
}

export function confidenceHighlightColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
}

export function confidenceLabelColor(confidence: number): string {
  if (confidence > 0.9) return "#059669";
  if (confidence > 0.7) return "#d97706";
  if (confidence > 0.5) return "#d97706";
  return "#dc2626";
}

export function formatConfidencePercent(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

/**
 * Length in code points: the API aligns `character_confidences` one to one
 * with the code points of the text, while UTF-16 `length` counts a combining
 * mark or an astral character differently and would reject a valid array.
 */
function codePoints(text: string): string[] {
  return Array.from(text);
}

export function characterConfidencesForTranscription(
  transcription: LineTranscriptionWithCharacterConfidence,
): CharacterConfidence[] {
  const explicit = transcription.character_confidences;
  if (explicit && explicit.length === codePoints(transcription.text).length) {
    return explicit;
  }
  if (transcription.confidence === null) {
    return codePoints(transcription.text).map((char) => ({
      char,
      confidence: 0,
    }));
  }
  return codePoints(transcription.text).map((char) => ({
    char,
    confidence: transcription.confidence as number,
  }));
}

export function hasDistinctCharacterConfidences(
  transcription: LineTranscriptionWithCharacterConfidence,
): boolean {
  const explicit = transcription.character_confidences;
  return Boolean(
    explicit && explicit.length === codePoints(transcription.text).length,
  );
}

/**
 * A letter of a cursive joining script: the font picks an initial, medial,
 * final or isolated glyph from the neighbours, and a browser only shapes
 * across characters that sit in the same inline box under the same font.
 *
 * It has to be a letter. The Arabic and Syriac blocks also hold punctuation
 * and Arabic-Indic digits, and a line carrying only those has nothing cursive
 * in it, so it keeps the ordinary padded spans. `Script_Extensions` rather
 * than `Script` so that the characters shared between these scripts, the
 * tatweel U+0640 above all, still count.
 */
const JOINING_SCRIPT_LETTER =
  /(?=[\p{Script_Extensions=Syriac}\p{Script_Extensions=Arabic}\p{Script_Extensions=Mandaic}\p{Script_Extensions=Nko}\p{Script_Extensions=Mongolian}])\p{L}/u;

export function containsJoiningScript(text: string): boolean {
  return JOINING_SCRIPT_LETTER.test(text);
}

/**
 * A code point that belongs to the cluster before it rather than opening one:
 * any combining mark, and the zero width non-joiner and joiner.
 */
const CONTINUES_CLUSTER = /[\p{M}\u200c\u200d]/u;

/**
 * Grapheme clusters, so a base letter keeps its combining marks (Syriac vowel
 * points and seyame U+0308, the Coptic supralinear stroke) and an astral
 * character keeps its surrogate pair.
 *
 * The fallback for a runtime without `Intl.Segmenter` iterates by code point,
 * which keeps a surrogate pair whole, and attaches marks itself. Plain
 * `Array.from` would leave every mark standing alone, and a mark scored into a
 * different tier than its base letter would then be rendered in its own span
 * and drift off the letter it belongs to.
 */
function graphemeClusters(text: string): string[] {
  const segmenter = (
    Intl as typeof Intl & { Segmenter?: typeof Intl.Segmenter }
  ).Segmenter;
  if (segmenter) {
    return Array.from(
      new segmenter(undefined, { granularity: "grapheme" }).segment(text),
      (entry) => entry.segment,
    );
  }
  const clusters: string[] = [];
  for (const char of text) {
    if (clusters.length > 0 && CONTINUES_CLUSTER.test(char)) {
      clusters[clusters.length - 1] += char;
    } else {
      clusters.push(char);
    }
  }
  return clusters;
}

/** One rendered span: consecutive grapheme clusters sharing a confidence tier. */
export type ConfidenceRun = {
  text: string;
  /** Lowest score in the run. Drives the tier, so a weak mark is never hidden. */
  confidence: number;
  /** Highest score in the run, for the tooltip when the run is not uniform. */
  maxConfidence: number;
};

/**
 * Collapse per-character scores into the fewest spans that still show every
 * tier. One span per character breaks a cursive script: the browser shapes
 * each box on its own, so every Syriac letter falls back to its isolated form
 * and the word reads as loose letters. Grouping also keeps a combining mark in
 * the same box as its base letter.
 */
export function groupConfidenceRuns(
  characterConfidences: CharacterConfidence[],
): ConfidenceRun[] {
  const text = characterConfidences.map((entry) => entry.char).join("");
  const runs: ConfidenceRun[] = [];
  let entryIndex = 0;
  let consumedUnits = 0;

  for (const cluster of graphemeClusters(text)) {
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    let remaining = cluster.length;

    while (remaining > 0 && entryIndex < characterConfidences.length) {
      const entry = characterConfidences[entryIndex];
      const available = entry.char.length - consumedUnits;
      const taken = Math.min(available, remaining);
      lowest = Math.min(lowest, entry.confidence);
      highest = Math.max(highest, entry.confidence);
      remaining -= taken;
      consumedUnits += taken;
      if (consumedUnits >= entry.char.length) {
        entryIndex += 1;
        consumedUnits = 0;
      }
    }

    const confidence = Number.isFinite(lowest) ? lowest : 0;
    const maxConfidence = Number.isFinite(highest) ? highest : confidence;
    const previous = runs[runs.length - 1];
    if (
      previous &&
      confidenceTierClass(previous.confidence) ===
        confidenceTierClass(confidence)
    ) {
      previous.text += cluster;
      previous.confidence = Math.min(previous.confidence, confidence);
      previous.maxConfidence = Math.max(previous.maxConfidence, maxConfidence);
    } else {
      runs.push({ text: cluster, confidence, maxConfidence });
    }
  }

  return runs;
}

/** Tooltip for a run: one score, or the span of scores it covers. */
export function confidenceRunTitle(run: ConfidenceRun): string {
  const tier = confidenceTierLabel(run.confidence);
  const lowest = formatConfidencePercent(run.confidence);
  if (
    Math.round(run.confidence * 1000) === Math.round(run.maxConfidence * 1000)
  ) {
    return `${lowest} confidence (${tier})`;
  }
  return `${lowest} to ${formatConfidencePercent(run.maxConfidence)} confidence (${tier})`;
}
