import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CharacterConfidenceText } from "./CharacterConfidenceText";
import type { CharacterConfidence } from "./characterConfidence";

const SYRIAC_WORD = "ܠܡܪܝ"; // ܠܡܪܝ
const SYRIAC_LINE = "ܠܡܪܝ ܥܠܡܐ"; // ܠܡܪܝ ܥܠܡܐ
const SEYAME_WORD = "ܝ̈ܘܡܐ"; // ܝ̈ܘܡܐ, seyame on the first letter
const GREEK_LINE = "ἐν αρχῇ"; // ἐν ἀρχῇ

function uniform(text: string, confidence: number): CharacterConfidence[] {
  return Array.from(text, (char) => ({ char, confidence }));
}

function withDip(
  text: string,
  base: number,
  dipIndex: number,
  dip: number,
): CharacterConfidence[] {
  return Array.from(text, (char, index) => ({
    char,
    confidence: index === dipIndex ? dip : base,
  }));
}

function renderText(characterConfidences: CharacterConfidence[]): HTMLElement {
  const { container } = render(
    <CharacterConfidenceText
      characterConfidences={characterConfidences}
      ariaLabel="model output"
    />,
  );
  return container.querySelector<HTMLElement>(".pe-confidence-text")!;
}

function spans(wrapper: HTMLElement): HTMLElement[] {
  return Array.from(wrapper.querySelectorAll<HTMLElement>("[data-conf]"));
}

/** Zero width joiners and the other invisible marks U+200B to U+200F. */
function invisibleFormatCharacters(text: string): string[] {
  return Array.from(text).filter((char) => {
    const code = char.codePointAt(0) ?? 0;
    return code >= 0x200b && code <= 0x200f;
  });
}

/** What a selection copy yields: every text node the element renders. */
function copiedText(wrapper: HTMLElement): string {
  const walker = document.createTreeWalker(wrapper, NodeFilter.SHOW_TEXT);
  let out = "";
  let node = walker.nextNode();
  while (node) {
    out += node.nodeValue ?? "";
    node = walker.nextNode();
  }
  return out;
}

describe("CharacterConfidenceText", () => {
  it("renders a Syriac word of one confidence as a single span", () => {
    // One span per letter made every letter shape in isolation, which is the
    // reported "ܠ ܡ ܪ ܝ instead of ܠܡܪܝ".
    const wrapper = renderText(uniform(SYRIAC_WORD, 0.96));
    const rendered = spans(wrapper);

    expect(rendered).toHaveLength(1);
    expect(rendered[0].textContent).toBe(SYRIAC_WORD);
  });

  it("keeps a Syriac line joined when the confidence changes inside a word", () => {
    const wrapper = renderText(withDip(SYRIAC_LINE, 0.96, 2, 0.4));
    const rendered = spans(wrapper);

    // Three tiers in a row, so three spans and no more.
    expect(rendered).toHaveLength(3);
    expect(rendered.map((span) => span.className)).toEqual([
      "ch-high",
      "ch-low",
      "ch-high",
    ]);
    // The mechanism: the padding that would open a gap at the span boundary is
    // dropped for a joining script, and nothing is inserted into the text.
    expect(wrapper).toHaveClass("pe-confidence-text--joining");
    expect(wrapper.textContent).toBe(SYRIAC_LINE);
    expect(invisibleFormatCharacters(wrapper.textContent ?? "")).toEqual([]);
  });

  it("never splits a base letter from its combining mark", () => {
    // Seyame U+0308 in its own span drifts off its base letter.
    const wrapper = renderText(withDip(SEYAME_WORD, 0.95, 1, 0.3));
    const rendered = spans(wrapper);

    expect(rendered[0].textContent).toBe("ܝ̈");
    for (const span of rendered) {
      expect(span.textContent?.startsWith("̈")).toBe(false);
    }
    expect(wrapper.textContent).toBe(SEYAME_WORD);
  });

  it.each([
    ["Syriac", SYRIAC_LINE],
    ["Syriac with seyame", SEYAME_WORD],
    ["Greek", GREEK_LINE],
  ])("renders and copies %s exactly as stored", (_name, line) => {
    const wrapper = renderText(withDip(line, 0.93, 1, 0.45));

    expect(wrapper.textContent).toBe(line);
    expect(copiedText(wrapper)).toBe(line);
  });

  it("leaves a Greek line unchanged in text and keeps its padded spans", () => {
    const wrapper = renderText(withDip(GREEK_LINE, 0.97, 3, 0.6));

    expect(wrapper.textContent).toBe(GREEK_LINE);
    expect(wrapper).not.toHaveClass("pe-confidence-text--joining");
    expect(wrapper).toHaveClass("pe-confidence-text");
    expect(spans(wrapper).map((span) => span.className)).toEqual([
      "ch-high",
      "ch-mid",
      "ch-high",
    ]);
  });
});
