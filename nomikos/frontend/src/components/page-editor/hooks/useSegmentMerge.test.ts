/**
 * The merge action and the one question it can be answered with.
 *
 * Most of this is about the 409. It is not a failure to report and then forget:
 * it is the server saying "more than one of these carries somebody's text, you
 * choose", and the hook has to hold the ids it was given long enough for the
 * reviewer to answer with one of them. Losing them would leave the dialog
 * unanswerable and the merge impossible from the UI.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../../api/errors";
import { useSegmentMerge } from "./useSegmentMerge";

const mergePartLines = vi.fn();
const getPagePairing = vi.fn();
const partContentChanged = vi.fn();

vi.mock("../../../api/client", () => ({
  api: {
    mergePartLines: (...args: unknown[]) => mergePartLines(...args),
    getPagePairing: (...args: unknown[]) => getPagePairing(...args),
  },
}));

vi.mock("../../../api/resources", () => ({
  invalidateAfter: {
    partContentChanged: (...args: unknown[]) => partContentChanged(...args),
  },
}));

const FIRST = "11111111-1111-4111-8111-111111111111";
const SECOND = "22222222-2222-4222-8222-222222222222";
const THIRD = "33333333-3333-4333-8333-333333333333";

function line(id: string) {
  return { id, order: 0 } as never;
}

function conflictError(lineIds: string[]) {
  return new ApiError("Several carry text", 409, null, {
    code: "MERGE_TEXT_CONFLICT",
    message: "Several carry text",
    line_ids: lineIds,
  });
}

function setup() {
  const setLines = vi.fn();
  const setTextLines = vi.fn();
  const setPairingProgress = vi.fn();
  const onMerged = vi.fn();
  const view = renderHook(() =>
    useSegmentMerge({
      projectId: "project-1",
      documentId: "document-1",
      partId: "part-1",
      setLines,
      setTextLines,
      setPairingProgress,
      onMerged,
    }),
  );
  return { ...view, setLines, setTextLines, setPairingProgress, onMerged };
}

describe("useSegmentMerge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mergePartLines.mockResolvedValue([line(FIRST)]);
    getPagePairing.mockResolvedValue({
      text_lines: [{ order: 0, text: "alpha", paired_line_id: FIRST }],
      pairing_progress: { paired_lines: 1, total_lines: 1, percent: 100 },
    });
  });

  it("sends the chosen ids and replaces the canvas with the answer", async () => {
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND]);
    });

    expect(mergePartLines).toHaveBeenCalledWith(
      "project-1",
      "document-1",
      "part-1",
      [FIRST, SECOND],
      undefined,
    );
    expect(view.setLines).toHaveBeenCalledWith([line(FIRST)]);
    expect(view.onMerged).toHaveBeenCalledWith(FIRST);
  });

  it("re-reads the pairing, because the merge released the deleted lines'", async () => {
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND]);
    });

    expect(getPagePairing).toHaveBeenCalled();
    expect(view.setTextLines).toHaveBeenCalled();
    expect(view.setPairingProgress).toHaveBeenCalled();
  });

  it("refuses a group of one without asking the server", async () => {
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST]);
    });

    expect(mergePartLines).not.toHaveBeenCalled();
    expect(view.result.current.mergeError).toMatch(/at least two/i);
  });

  it("refuses more than the cap without asking the server", async () => {
    const view = setup();
    const tooMany = Array.from({ length: 21 }, (_, index) => `line-${index}`);

    await act(async () => {
      await view.result.current.mergeSegments(tooMany);
    });

    expect(mergePartLines).not.toHaveBeenCalled();
    expect(view.result.current.mergeError).toMatch(/20 segments/);
  });

  it("turns a 409 into a question rather than an error message", async () => {
    mergePartLines.mockRejectedValueOnce(conflictError([FIRST, THIRD]));
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND, THIRD]);
    });

    expect(view.result.current.conflict?.lineIds).toEqual([FIRST, THIRD]);
    expect(view.result.current.mergeError).toBeNull();
    expect(view.setLines).not.toHaveBeenCalled();
  });

  it("answers the conflict with the same ids plus the keeper", async () => {
    mergePartLines.mockRejectedValueOnce(conflictError([FIRST, THIRD]));
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND, THIRD]);
    });
    await act(async () => {
      view.result.current.resolveConflict(THIRD);
    });

    await waitFor(() => {
      expect(mergePartLines).toHaveBeenLastCalledWith(
        "project-1",
        "document-1",
        "part-1",
        [FIRST, SECOND, THIRD],
        THIRD,
      );
    });
    await waitFor(() => {
      expect(view.result.current.conflict).toBeNull();
    });
  });

  it("forgets the question when the reviewer backs out", async () => {
    mergePartLines.mockRejectedValueOnce(conflictError([FIRST, THIRD]));
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND, THIRD]);
    });
    act(() => {
      view.result.current.dismissConflict();
    });

    expect(view.result.current.conflict).toBeNull();
    act(() => {
      view.result.current.resolveConflict(THIRD);
    });
    expect(mergePartLines).toHaveBeenCalledTimes(1);
  });

  it("reports an ordinary refusal as a message and keeps the canvas", async () => {
    mergePartLines.mockRejectedValueOnce(new ApiError("Nope", 403));
    const view = setup();

    await act(async () => {
      await view.result.current.mergeSegments([FIRST, SECOND]);
    });

    expect(view.result.current.conflict).toBeNull();
    expect(view.result.current.mergeError).toMatch(/project members/i);
    expect(view.setLines).not.toHaveBeenCalled();
  });
});
