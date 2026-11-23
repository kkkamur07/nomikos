import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { LineTranscriptionResponse } from "../../api/client";
import { StripEditBox } from "./StripEditBox";

function reading(
  text: string,
  scores: number[] | null,
  confidence: number | null = 0.8,
): LineTranscriptionResponse {
  return {
    id: "tx-1",
    transcription_id: "model-1",
    transcription_kind: "model",
    text,
    confidence,
    character_confidences:
      scores === null
        ? null
        : Array.from(text, (char, index) => ({
            char,
            confidence: scores[index],
          })),
  } as LineTranscriptionResponse;
}

type Props = Parameters<typeof StripEditBox>[0];

function props(overrides: Partial<Props> = {}): Props {
  return {
    segmentId: "seg-1",
    draft: "abcde",
    modelReading: reading("abcde", [0.99, 0.7, 0.4, 0.99, 0.99]),
    showsModelReading: true,
    ariaLabel: "Edit transcription",
    disabled: false,
    onChange: vi.fn(),
    onKeyDown: vi.fn(),
    ...overrides,
  };
}

function setup(overrides: Partial<Props> = {}) {
  const view = render(<StripEditBox {...props(overrides)} />);
  const textarea = () =>
    view.getByLabelText("Edit transcription") as HTMLTextAreaElement;
  const mirror = () => view.getByTestId("strip-edit-mirror");
  return {
    ...view,
    textarea,
    mirror,
    again: (next: Partial<Props>) =>
      view.rerender(<StripEditBox {...props(next)} />),
  };
}

describe("StripEditBox mirror", () => {
  it("tints the model reading by confidence and hides the mirror from AT", () => {
    const { mirror } = setup();
    expect(mirror().getAttribute("aria-hidden")).toBe("true");
    const spans = Array.from(mirror().querySelectorAll("[data-conf]"));
    expect(spans.map((s) => s.className)).toEqual([
      expect.stringContaining("ch-high"),
      expect.stringContaining("ch-mid"),
      expect.stringContaining("ch-low"),
      expect.stringContaining("ch-high"),
    ]);
    expect(spans.map((s) => s.textContent)).toEqual(["a", "b", "c", "de"]);
  });

  it("drops the tint once the draft is no longer the model reading", () => {
    const { mirror, again } = setup();
    expect(mirror().querySelector("[data-conf]")).not.toBeNull();
    again({ draft: "abXde", showsModelReading: false });
    expect(mirror().querySelector("[data-conf]")).toBeNull();
  });

  it("shows no tint on a ground truth line", () => {
    const withReading = setup({ showsModelReading: false });
    expect(withReading.mirror().querySelector("[data-conf]")).toBeNull();
    withReading.unmount();
    const without = setup({ showsModelReading: false, modelReading: null });
    expect(without.mirror().querySelector("[data-conf]")).toBeNull();
  });
});

describe("StripEditBox caret", () => {
  it("puts the caret after the first low character", () => {
    const { textarea } = setup();
    expect(textarea().selectionStart).toBe(3);
    expect(textarea().selectionEnd).toBe(3);
  });

  it("puts the caret at the end when no character is low", () => {
    const { textarea } = setup({
      modelReading: reading("abcde", [0.99, 0.95, 0.7, 0.99, 0.99]),
    });
    expect(textarea().selectionStart).toBe(5);
    expect(textarea().selectionEnd).toBe(5);
  });

  it("puts the caret at the end with only a line-level confidence", () => {
    const { textarea } = setup({ modelReading: reading("abcde", null, 0.1) });
    expect(textarea().selectionStart).toBe(5);
    expect(textarea().selectionEnd).toBe(5);
  });

  it("moves the caret again for a new segment and reading", () => {
    const { textarea, again } = setup();
    expect(textarea().selectionStart).toBe(3);
    again({
      segmentId: "seg-2",
      draft: "xyz",
      modelReading: reading("xyz", [0.2, 0.99, 0.99]),
    });
    expect(textarea().selectionStart).toBe(1);
    expect(textarea().selectionEnd).toBe(1);
  });

  it("does not take focus", () => {
    const { textarea } = setup();
    expect(document.activeElement).not.toBe(textarea());
  });

  it("keeps a Coptic letter and its overline in one low span, caret after both", () => {
    const text = "ⲡⲉⲝ︦ⲥ";
    const { textarea, mirror } = setup({
      draft: text,
      modelReading: reading(text, [0.99, 0.99, 0.99, 0.3, 0.99]),
    });
    const low = mirror().querySelector(".ch-low");
    expect(low?.textContent).toBe("ⲝ︦");
    expect(textarea().selectionStart).toBe(4);
    expect(textarea().selectionEnd).toBe(4);
  });
});

describe("StripEditBox direction and keys", () => {
  it("sets rtl on both layers for Syriac and ltr for Greek", () => {
    const syriac = "ܠܡܪܝ ܥܠܡܐ";
    const rtl = setup({
      draft: syriac,
      modelReading: reading(syriac, null),
    });
    expect(rtl.textarea().getAttribute("dir")).toBe("rtl");
    expect(rtl.mirror().getAttribute("dir")).toBe("rtl");
    rtl.unmount();
    const ltr = setup({ draft: "ἐν ἀρχῇ", modelReading: null });
    expect(ltr.textarea().getAttribute("dir")).toBe("ltr");
    expect(ltr.mirror().getAttribute("dir")).toBe("ltr");
  });

  it("forwards key events such as Cmd+Enter", () => {
    const onKeyDown = vi.fn();
    const { textarea } = setup({ onKeyDown });
    fireEvent.keyDown(textarea(), { key: "Enter", metaKey: true });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
  });
});

describe("StripEditBox script font", () => {
  it.each([
    ["syriac", "ܫܠܡܐ ܥܠܡܐ"],
    ["coptic", "ⲡⲛⲟⲩⲧⲉ ⲡⲉ"],
  ])(
    "gives the textarea and mirror the same %s script font",
    (script, text) => {
      const { textarea, mirror } = setup({
        draft: text,
        modelReading: reading(text, null),
      });
      for (const layer of [textarea(), mirror()]) {
        expect(layer.classList.contains("script-text")).toBe(true);
        expect(layer.getAttribute("data-script")).toBe(script);
      }
    },
  );
});
