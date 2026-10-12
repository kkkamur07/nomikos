import { describe, expect, it } from "vitest";

import {
  characterConfidencesForTranscription,
  confidenceHighlightColor,
  confidenceLabelColor,
  confidenceRunTitle,
  containsJoiningScript,
  formatConfidencePercent,
  groupConfidenceRuns,
  hasDistinctCharacterConfidences,
  type CharacterConfidence,
  type LineTranscriptionWithCharacterConfidence,
} from "./characterConfidence";

const BASE_TRANSCRIPTION: LineTranscriptionWithCharacterConfidence = {
  id: "tx-1",
  transcription_id: "model-1",
  transcription_kind: "model",
  text: "abc",
  confidence: 0.82,
};

describe("characterConfidence", () => {
  it("maps confidence values to highlight and label colors", () => {
    expect(confidenceHighlightColor(0.95)).toBe("#059669");
    expect(confidenceHighlightColor(0.8)).toBe("#d97706");
    expect(confidenceHighlightColor(0.6)).toBe("#d97706");
    expect(confidenceHighlightColor(0.2)).toBe("#dc2626");
    expect(confidenceLabelColor(0.95)).toBe("#059669");
    expect(formatConfidencePercent(0.825)).toBe("82.5%");
  });

  it("uses explicit per-character confidences when aligned with text", () => {
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      character_confidences: [
        { char: "a", confidence: 0.99 },
        { char: "b", confidence: 0.55 },
        { char: "c", confidence: 0.71 },
      ],
    };

    expect(hasDistinctCharacterConfidences(transcription)).toBe(true);
    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "a", confidence: 0.99 },
      { char: "b", confidence: 0.55 },
      { char: "c", confidence: 0.71 },
    ]);
  });

  it("falls back to line confidence for each character when per-char scores are missing", () => {
    expect(hasDistinctCharacterConfidences(BASE_TRANSCRIPTION)).toBe(false);
    expect(characterConfidencesForTranscription(BASE_TRANSCRIPTION)).toEqual([
      { char: "a", confidence: 0.82 },
      { char: "b", confidence: 0.82 },
      { char: "c", confidence: 0.82 },
    ]);
  });

  it("aligns explicit scores by code point, not UTF-16 unit", () => {
    // "\u{10900}" is one code point in two UTF-16 units: text.length is 3
    // while the API's array holds 2 entries. Counting UTF-16 units would
    // reject the valid array and split the surrogate pair in the fallback.
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      text: "\u{10900}a",
      character_confidences: [
        { char: "\u{10900}", confidence: 0.91 },
        { char: "a", confidence: 0.42 },
      ],
    };

    expect(hasDistinctCharacterConfidences(transcription)).toBe(true);
    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "\u{10900}", confidence: 0.91 },
      { char: "a", confidence: 0.42 },
    ]);
  });

  it("falls back one entry per code point for text with a combining mark", () => {
    // Greek alpha with a combining acute: two code points, two entries, with
    // the accent kept as its own code point for the grouper to reattach.
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      text: "\u03B1\u0301",
      confidence: 0.7,
    };

    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "\u03B1", confidence: 0.7 },
      { char: "\u0301", confidence: 0.7 },
    ]);
  });

  it("falls back per code point when the array does not describe the text", () => {
    const transcription: LineTranscriptionWithCharacterConfidence = {
      ...BASE_TRANSCRIPTION,
      text: "ab",
      character_confidences: [{ char: "a", confidence: 0.99 }],
    };

    expect(hasDistinctCharacterConfidences(transcription)).toBe(false);
    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "a", confidence: 0.82 },
      { char: "b", confidence: 0.82 },
    ]);
  });
});

describe("groupConfidenceRuns", () => {
  const scored = (
    text: string,
    ...confidences: number[]
  ): CharacterConfidence[] =>
    Array.from(text, (char, index) => ({
      char,
      confidence: confidences[index] ?? confidences[confidences.length - 1],
    }));

  it("collapses a uniform line into one run", () => {
    expect(
      groupConfidenceRuns(scored("\u0720\u0721\u072A\u071D", 0.96)),
    ).toEqual([
      {
        text: "\u0720\u0721\u072A\u071D",
        confidence: 0.96,
        maxConfidence: 0.96,
      },
    ]);
  });

  it("splits only where the tier changes, not where the score changes", () => {
    const runs = groupConfidenceRuns(scored("abcd", 0.99, 0.95, 0.4, 0.42));

    expect(runs.map((run) => run.text)).toEqual(["ab", "cd"]);
    expect(runs[0]).toEqual({
      text: "ab",
      confidence: 0.95,
      maxConfidence: 0.99,
    });
    expect(runs[1]).toEqual({
      text: "cd",
      confidence: 0.4,
      maxConfidence: 0.42,
    });
  });

  it("keeps a combining mark in the run of its base letter", () => {
    // Scored on its own, the seyame would open a low run of one mark and the
    // dots would render away from the letter they belong to.
    const runs = groupConfidenceRuns(
      scored("\u071D\u0308\u0718", 0.95, 0.2, 0.95),
    );

    expect(runs.map((run) => run.text)).toEqual(["\u071D\u0308", "\u0718"]);
  });

  it("keeps an astral character whole", () => {
    // "".split("") would cut the surrogate pair in half.
    const runs = groupConfidenceRuns([
      { char: "\u{10900}", confidence: 0.9 },
      { char: "a", confidence: 0.9 },
    ]);

    expect(runs).toHaveLength(1);
    expect(runs[0].text).toBe("\u{10900}a");
  });

  it("scores an astral character and a following Syriac letter separately", () => {
    // The backend counts Python code points (one entry per code point), so an
    // astral character is one entry holding a surrogate pair. The cluster walk
    // consumes entries in matching UTF-16 units, so the pair stays in its own
    // cluster and each letter keeps its own score.
    const runs = groupConfidenceRuns([
      { char: "\u{10900}", confidence: 0.42 },
      { char: "\u0710", confidence: 0.96 },
    ]);

    expect(runs.map((run) => run.text)).toEqual(["\u{10900}", "\u0710"]);
    expect(runs[0].confidence).toBe(0.42);
    expect(runs[1].confidence).toBe(0.96);
  });

  it("returns nothing for empty input", () => {
    expect(groupConfidenceRuns([])).toEqual([]);
  });
});

describe("containsJoiningScript", () => {
  it("is true for a letter of a joining script", () => {
    expect(containsJoiningScript("\u0720\u0721")).toBe(true); // Syriac
    expect(containsJoiningScript("\u0645\u0631")).toBe(true); // Arabic
    expect(containsJoiningScript("\u0860")).toBe(true); // Syriac Supplement
    expect(containsJoiningScript("\u08A0")).toBe(true); // Arabic Extended-A
    expect(containsJoiningScript("\u0870")).toBe(true); // Arabic Extended-B
    expect(containsJoiningScript("\uFB50")).toBe(true); // presentation form
    expect(containsJoiningScript("\u07CA")).toBe(true); // N'Ko
    expect(containsJoiningScript("\u0840")).toBe(true); // Mandaic
    expect(containsJoiningScript("\u1820")).toBe(true); // Mongolian
    expect(containsJoiningScript("\u0640")).toBe(true); // tatweel, shared
  });

  it("is true for a line that holds one such letter among other scripts", () => {
    expect(containsJoiningScript("fol. 4v \u0720 line 22")).toBe(true);
  });

  it("is false for the non-joining scripts served", () => {
    expect(containsJoiningScript("\u1F10\u03BD")).toBe(false); // Greek
    expect(containsJoiningScript("\u2C81\u2C93")).toBe(false); // Coptic
    expect(containsJoiningScript("\u0531\u0561")).toBe(false); // Armenian
    expect(containsJoiningScript("\u05D0")).toBe(false); // Hebrew, not cursive
    expect(containsJoiningScript("")).toBe(false);
  });

  it("is false for punctuation and digits of those blocks with no letter", () => {
    // These would have switched the whole line to the joining style, dropping
    // the padding and the tier weights with nothing cursive to protect.
    expect(containsJoiningScript("\u060C")).toBe(false); // Arabic comma
    expect(containsJoiningScript("\u0660\u0661\u0662")).toBe(false); // Arabic-Indic digits
    expect(containsJoiningScript("\u0700\u070A")).toBe(false); // Syriac punctuation
    expect(containsJoiningScript("\u060C \u0661\u0662 \u0700")).toBe(false);
  });
});

describe("groupConfidenceRuns without Intl.Segmenter", () => {
  /** Runs the body on a runtime that has no `Intl.Segmenter`. */
  function withoutSegmenter<T>(body: () => T): T {
    const host = Intl as typeof Intl & { Segmenter?: typeof Intl.Segmenter };
    const original = host.Segmenter;
    delete host.Segmenter;
    try {
      return body();
    } finally {
      host.Segmenter = original;
    }
  }

  it("takes the fallback path only when Intl.Segmenter is gone", () => {
    expect(withoutSegmenter(() => "Segmenter" in Intl)).toBe(false);
    expect("Segmenter" in Intl).toBe(true);
  });

  it("keeps a combining mark with its base letter across a tier change", () => {
    // Scored on its own the seyame opens a low run of one mark, and the dots
    // render away from the letter they belong to.
    const runs = withoutSegmenter(() =>
      groupConfidenceRuns([
        { char: "\u071D", confidence: 0.95 },
        { char: "\u0308", confidence: 0.2 },
        { char: "\u0718", confidence: 0.95 },
      ]),
    );

    expect(runs.map((run) => run.text)).toEqual(["\u071D\u0308", "\u0718"]);
    expect(runs[0].confidence).toBe(0.2);
    expect(runs[0].maxConfidence).toBe(0.95);
  });

  it("keeps a zero width joiner and an astral character with their cluster", () => {
    const runs = withoutSegmenter(() =>
      groupConfidenceRuns([
        { char: "\u0645", confidence: 0.9 },
        { char: "\u200D", confidence: 0.3 },
        { char: "\u{10900}", confidence: 0.9 },
      ]),
    );

    expect(runs.map((run) => run.text)).toEqual(["\u0645\u200D", "\u{10900}"]);
  });

  it("gives the same runs as Intl.Segmenter for the lines we render", () => {
    const scored = (text: string) =>
      Array.from(text, (char, index) => ({
        char,
        confidence: index % 3 === 0 ? 0.4 : 0.95,
      }));

    for (const line of [
      "\u071D\u0308\u0718\u0721\u0710",
      "\u0720\u0721\u072A\u071D \u0725\u0720\u0721\u0710",
      "\u1F10\u03BD \u1F00\u03C1\u03C7\u1FC7",
      "\u2C81\u0305\u2C93\u2CA1",
    ]) {
      const withIntl = groupConfidenceRuns(scored(line));
      const fallback = withoutSegmenter(() =>
        groupConfidenceRuns(scored(line)),
      );
      expect(fallback).toEqual(withIntl);
    }
  });
});

describe("confidenceRunTitle", () => {
  it("names one score for a uniform run and a span of scores otherwise", () => {
    expect(
      confidenceRunTitle({ text: "ab", confidence: 0.96, maxConfidence: 0.96 }),
    ).toBe("96.0% confidence (high)");
    expect(
      confidenceRunTitle({ text: "ab", confidence: 0.92, maxConfidence: 0.99 }),
    ).toBe("92.0% to 99.0% confidence (high)");
  });
});
