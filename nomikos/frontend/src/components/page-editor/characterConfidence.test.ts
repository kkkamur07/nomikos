import { describe, expect, it } from "vitest";

import {
  characterConfidencesForTranscription,
  formatConfidencePercent,
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
  it("formats a confidence as a percentage", () => {
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

    expect(characterConfidencesForTranscription(transcription)).toEqual([
      { char: "a", confidence: 0.99 },
      { char: "b", confidence: 0.55 },
      { char: "c", confidence: 0.71 },
    ]);
  });

  it("falls back to line confidence for each character when per-char scores are missing", () => {
    expect(characterConfidencesForTranscription(BASE_TRANSCRIPTION)).toEqual([
      { char: "a", confidence: 0.82 },
      { char: "b", confidence: 0.82 },
      { char: "c", confidence: 0.82 },
    ]);
  });
});
