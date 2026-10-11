import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { PublicLayoutResponse } from "../../api/client";
import { PublicTranscriptPanel } from "./PublicTranscriptPanel";

function layoutWithOrders(...orders: number[]): PublicLayoutResponse {
  return {
    blocks: [],
    lines: orders.map((order) => ({
      id: `line-${order}`,
      part_id: "part-1",
      order,
      points: [
        [0, 0],
        [10, 0],
        [10, 5],
      ],
      line_transcriptions: [],
    })),
  } as unknown as PublicLayoutResponse;
}

describe("PublicTranscriptPanel", () => {
  it("numbers lines by reading position, the way the editor does, across stored gaps", () => {
    render(
      <PublicTranscriptPanel
        partId="part-1"
        layout={layoutWithOrders(0, 1, 4, 9)}
        selectedLineIndex={2}
        onSelectLine={vi.fn()}
      />,
    );

    const list = screen.getByRole("list", {
      name: "Page transcription lines",
    });
    const numbers = within(list)
      .getAllByRole("button")
      .map((item) => item.querySelector(".pub-line-index__num")?.textContent);
    expect(numbers).toEqual(["1", "2", "3", "4"]);
    expect(screen.getByText("Line 3")).toBeInTheDocument();
  });
});
