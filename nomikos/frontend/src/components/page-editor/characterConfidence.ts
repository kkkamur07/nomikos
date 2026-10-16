import type {
  CharacterConfidence,
  LineTranscriptionResponse,
} from "../../api/client";

export type { CharacterConfidence };

/**
 * A transcription plus per-character scores. The platform API has no such
 * field (`LineTranscriptionResponse` carries one confidence for the whole
 * line), so this is a client-side extension, declared once, here.
 *
 * Nothing populates it today: a local run's `character_confidences` are sent
 * to the server by `persistLocalTranscribe` and are not returned, so every
 * transcription the editor holds falls back to the per-line confidence below.
 */
export type LineTranscriptionWithCharacterConfidence =
  LineTranscriptionResponse & {
    character_confidences?: CharacterConfidence[] | null;
  };

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

export function formatConfidencePercent(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function characterConfidencesForTranscription(
  transcription: LineTranscriptionWithCharacterConfidence,
): CharacterConfidence[] {
  const explicit = transcription.character_confidences;
  if (explicit && explicit.length === transcription.text.length) {
    return explicit;
  }
  const confidence = transcription.confidence ?? 0;
  return transcription.text.split("").map((char) => ({ char, confidence }));
}
