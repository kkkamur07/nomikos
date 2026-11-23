import { useLayoutEffect, useRef, type KeyboardEvent } from "react";
import type { LineTranscriptionResponse } from "../../api/client";
import { dominantScript, textDirection } from "../../utils/scriptDirection";
import { CharacterConfidenceText } from "./CharacterConfidenceText";
import {
  caretAfterFirstLowConfidence,
  characterConfidencesForTranscription,
} from "./characterConfidence";

type StripEditBoxProps = {
  segmentId: string | null;
  draft: string;
  /** The model reading on offer; coloured only while the box holds it. */
  modelReading: LineTranscriptionResponse | null;
  showsModelReading: boolean;
  ariaLabel: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
};

/**
 * The strip's one text box. While it holds the model's reading untouched, a
 * mirror stacked behind the transparent textarea tints each letter by the
 * model's confidence. The mirror copies the textarea's box, font and wrapping
 * and follows its scroll, so a tint stays under its letter; it renders nothing
 * once the text is the researcher's. Both layers take the bundled script font
 * the same way (``script-text`` plus ``data-script``, see
 * styles/script-fonts.css), so a font choice on the page moves them together.
 */
export function StripEditBox({
  segmentId,
  draft,
  modelReading,
  showsModelReading,
  ariaLabel,
  disabled,
  onChange,
  onKeyDown,
}: StripEditBoxProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const coloured = showsModelReading && modelReading !== null;
  const direction = textDirection(draft);
  // The script follows the text as typed, like a transcript row; an empty box
  // takes the model reading's, so the caret font matches what comes next.
  const script =
    dominantScript(draft) ?? dominantScript(modelReading?.text ?? "");
  const scriptAttribute = script ?? undefined;

  function syncScroll() {
    const textarea = textareaRef.current;
    const mirror = mirrorRef.current;
    if (!textarea || !mirror) return;
    mirror.scrollTop = textarea.scrollTop;
    mirror.scrollLeft = textarea.scrollLeft;
  }

  // A newly opened line holding the model reading puts the caret after its
  // first weak letter. This moves the selection only: focus stays where the
  // researcher left it, and a box that already has focus (after Cmd+Enter)
  // is ready to type into.
  const modelText = coloured ? modelReading.text : null;
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea || modelText === null || !modelReading) return;
    const caret = caretAfterFirstLowConfidence(modelReading);
    textarea.setSelectionRange(caret, caret);
    textarea.scrollTop = 0;
    syncScroll();
    // Runs per opened line and per new reading, not per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmentId, modelText]);

  return (
    <div
      className={
        showsModelReading
          ? "pe-strip__edit-box pe-strip__edit-box--model"
          : "pe-strip__edit-box"
      }
    >
      <div
        ref={mirrorRef}
        className="pe-strip__mirror script-text"
        aria-hidden="true"
        dir={direction}
        data-script={scriptAttribute}
        data-testid="strip-edit-mirror"
      >
        {coloured && (
          <CharacterConfidenceText
            characterConfidences={characterConfidencesForTranscription(
              modelReading,
            )}
            ariaLabel=""
          />
        )}
        {/* Keeps a trailing newline's line box, as the textarea has one. */}{" "}
      </div>
      <textarea
        ref={textareaRef}
        className="pe-strip__edit script-text"
        id="strip-edit"
        aria-label={ariaLabel}
        aria-describedby={coloured ? "confidence-legend" : undefined}
        spellCheck={false}
        dir={direction}
        data-script={scriptAttribute}
        placeholder="Type the transcription…"
        value={draft}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.target.value);
          syncScroll();
        }}
        onScroll={syncScroll}
        onKeyDown={onKeyDown}
      />
    </div>
  );
}
