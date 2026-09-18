/**
 * Picking several Segments on the canvas and folding them into one.
 *
 * The three things a reviewer does, end to end: shift-click to build a group,
 * press the action, and answer the one question the server can ask back. The
 * last is the one worth a page-level test rather than a hook test: the ids in
 * the 409 have to reach a dialog that shows the *text* of each, or the question
 * cannot be answered by anyone who is not reading the network tab.
 */
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import {
  DOCUMENT,
  flushPageEditorEffects,
  line,
  mockedApi,
  renderPageEditor,
  resetPageEditorApiMocks,
} from "./testSupport";

function segmentAt(order: number, top: number) {
  return line({
    id: `line-${order + 1}`,
    order,
    points: [
      [10, top],
      [50, top],
      [50, top + 20],
      [10, top + 20],
    ],
  });
}

function withGroundTruth(segment: ReturnType<typeof line>, text: string) {
  return {
    ...segment,
    line_transcriptions: [
      {
        id: `lt-${segment.id}`,
        transcription_id: "ground-truth-1",
        transcription_kind: "ground_truth",
        text,
        confidence: null,
      },
    ],
  };
}

const THREE = [segmentAt(0, 10), segmentAt(1, 40), segmentAt(2, 70)];

/** ``Segment 1`` alone, or with the ", paired" / ", selected for merge" tail. */
function segmentButton(number: number) {
  return screen.getByRole("button", {
    name: new RegExp(`^Segment ${number}(,|$)`),
  });
}

async function selectTwoSegments() {
  await screen.findByRole("button", { name: /^Segment 1(,|$)/ });
  fireEvent.click(segmentButton(1));
  fireEvent.click(segmentButton(2), { shiftKey: true });
}

describe("PageEditorPlaceholderPage merge", () => {
  beforeEach(() => {
    resetPageEditorApiMocks();
    mockedApi.getDocument.mockResolvedValue(DOCUMENT);
    mockedApi.listPartLines.mockResolvedValue(THREE);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await flushPageEditorEffects();
  });

  it("shows no merge action until a second Segment is shift-clicked", async () => {
    renderPageEditor();

    await screen.findByRole("button", { name: /^Segment 1(,|$)/ });
    fireEvent.click(segmentButton(1));
    expect(screen.queryByRole("button", { name: /^Merge \(/ })).toBeNull();

    fireEvent.click(segmentButton(2), { shiftKey: true });

    expect(
      await screen.findByRole("button", { name: "Merge (2)" }),
    ).toBeTruthy();
  });

  it("shift-clicking a chosen Segment again takes it back out of the group", async () => {
    renderPageEditor();
    await selectTwoSegments();
    expect(
      await screen.findByRole("button", { name: "Merge (2)" }),
    ).toBeTruthy();

    fireEvent.click(segmentButton(2), { shiftKey: true });

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^Merge \(/ })).toBeNull();
    });
  });

  it("merges the group in page order and replaces the canvas with the answer", async () => {
    mockedApi.mergePartLines.mockResolvedValue([THREE[0], THREE[2]]);
    renderPageEditor();
    await selectTwoSegments();

    fireEvent.click(await screen.findByRole("button", { name: "Merge (2)" }));

    await waitFor(() => {
      expect(mockedApi.mergePartLines).toHaveBeenCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        ["line-1", "line-2"],
        undefined,
      );
    });
    expect(await screen.findByText("2 Segments")).toBeTruthy();
  });

  it("merges from the keyboard with M", async () => {
    mockedApi.mergePartLines.mockResolvedValue([THREE[0], THREE[2]]);
    renderPageEditor();
    await selectTwoSegments();
    await screen.findByRole("button", { name: "Merge (2)" });

    fireEvent.keyDown(window, { key: "m" });

    await waitFor(() => {
      expect(mockedApi.mergePartLines).toHaveBeenCalled();
    });
  });

  it("asks which text to keep when the server refuses, and answers with it", async () => {
    const paired = [
      withGroundTruth(THREE[0], "alpha"),
      withGroundTruth(THREE[1], "beta"),
      THREE[2],
    ];
    mockedApi.listPartLines.mockResolvedValue(paired);
    mockedApi.mergePartLines
      .mockRejectedValueOnce(
        new ApiError("Several carry text", 409, null, {
          code: "MERGE_TEXT_CONFLICT",
          message: "Several carry text",
          line_ids: ["line-1", "line-2"],
        }),
      )
      .mockResolvedValue([paired[1], paired[2]]);
    renderPageEditor();
    await selectTwoSegments();

    fireEvent.click(await screen.findByRole("button", { name: "Merge (2)" }));

    // The words, not only the numbers: "keep 1 or 2?" is unanswerable.
    const choice = await screen.findByRole("radio", { name: /beta/ });
    fireEvent.click(choice);
    fireEvent.click(
      screen.getByRole("button", { name: /merge and keep this text/i }),
    );

    await waitFor(() => {
      expect(mockedApi.mergePartLines).toHaveBeenLastCalledWith(
        "project-1",
        "doc-1",
        "part-1",
        ["line-1", "line-2"],
        "line-2",
      );
    });
  });
});
