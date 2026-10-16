import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PageEditorSegmentNav } from "./PageEditorSegmentNav";

function renderNav(
  segmentNumber: number | null,
  totalSegments: number,
  disabled = false,
) {
  render(
    <PageEditorSegmentNav
      segmentNumber={segmentNumber}
      totalSegments={totalSegments}
      onPrevious={() => {}}
      onNext={() => {}}
      disabled={disabled}
    />,
  );
  const previous = screen.getByRole("button", { name: "Previous segment" });
  const next = screen.getByRole("button", { name: "Next segment" });
  return {
    label: screen.getByRole("navigation").textContent?.replace(/[‹›]/g, ""),
    previous: !(previous as HTMLButtonElement).disabled,
    next: !(next as HTMLButtonElement).disabled,
  };
}

describe("PageEditorSegmentNav", () => {
  it("moves both ways from a middle segment", () => {
    expect(renderNav(2, 3)).toEqual({
      label: "2 / 3",
      previous: true,
      next: true,
    });
  });

  it("stops at the first segment", () => {
    expect(renderNav(1, 3)).toMatchObject({ previous: false, next: true });
  });

  it("stops at the last segment", () => {
    expect(renderNav(3, 3)).toMatchObject({ previous: true, next: false });
  });

  it("shows the count but does not move with nothing selected", () => {
    expect(renderNav(null, 3)).toEqual({
      label: "- / 3",
      previous: false,
      next: false,
    });
  });

  it("shows a dash on a page with no segments", () => {
    expect(renderNav(null, 0)).toEqual({
      label: "-",
      previous: false,
      next: false,
    });
  });

  it("does not move while disabled", () => {
    expect(renderNav(2, 3, true)).toMatchObject({
      previous: false,
      next: false,
    });
  });
});
