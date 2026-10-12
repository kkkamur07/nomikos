import type { CharacterConfidence } from "./characterConfidence";
import {
  confidenceRunTitle,
  confidenceTierClass,
  confidenceTierLabel,
  containsJoiningScript,
  groupConfidenceRuns,
} from "./characterConfidence";

type CharacterConfidenceTextProps = {
  characterConfidences: CharacterConfidence[];
  ariaLabel: string;
};

export function CharacterConfidenceText({
  characterConfidences,
  ariaLabel,
}: CharacterConfidenceTextProps) {
  const runs = groupConfidenceRuns(characterConfidences);
  // Syriac and Arabic are cursive: the horizontal padding the tint carries
  // would put a gap at every span boundary and the letters would fall back to
  // their isolated forms. The modifier drops that padding for such a line.
  const joining = runs.some((run) => containsJoiningScript(run.text));
  const className = joining
    ? "pe-confidence-text pe-confidence-text--joining"
    : "pe-confidence-text";

  return (
    <span className={className} aria-label={ariaLabel}>
      {runs.map((run, index) => (
        <span
          key={`${index}-${run.text}`}
          className={confidenceTierClass(run.confidence)}
          data-conf={Math.round(run.confidence * 100)}
          data-tier={confidenceTierLabel(run.confidence)}
          title={confidenceRunTitle(run)}
        >
          {run.text}
        </span>
      ))}
    </span>
  );
}
