import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import type * as ClientModule from "../../api/client";
import { ApiError } from "../../api/errors";
import { FeedbackDialog } from "./FeedbackDialog";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<ClientModule>();
  return {
    ...actual,
    api: { ...actual.api, sendFeedback: vi.fn() },
  };
});

function openDialog(props?: { pagePath?: string; onSent?: () => void }) {
  const onClose = vi.fn();
  const onSent = props?.onSent ?? vi.fn();
  render(
    <FeedbackDialog
      open
      pagePath={props?.pagePath ?? "/projects/1"}
      onClose={onClose}
      onSent={onSent}
    />,
  );
  return { onClose, onSent };
}

beforeEach(() => {
  vi.mocked(api.sendFeedback).mockReset();
});

describe("FeedbackDialog", () => {
  it("sends the trimmed message with the page path", async () => {
    vi.mocked(api.sendFeedback).mockResolvedValue({
      id: "f-1",
      created_at: "2026-10-11T12:00:00Z",
      message: "fix the contrast",
      page_path: "/projects/1",
    });
    const { onSent } = openDialog();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "  fix the contrast  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(api.sendFeedback).toHaveBeenCalledWith({
        message: "fix the contrast",
        page_path: "/projects/1",
      }),
    );
    await waitFor(() => expect(onSent).toHaveBeenCalledTimes(1));
  });

  it("does not send a blank message", () => {
    openDialog();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(api.sendFeedback).not.toHaveBeenCalled();
  });

  it("shows the counter only past 1800 characters", () => {
    openDialog();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "x".repeat(100) },
    });
    expect(screen.queryByText(/\/ 2000/)).toBeNull();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "x".repeat(1801) },
    });
    expect(screen.getByText("1801 / 2000")).toBeTruthy();
  });

  it("explains the 429 and keeps the dialog open", async () => {
    vi.mocked(api.sendFeedback).mockRejectedValue(
      new ApiError("Too many feedback notes; try again later", 429),
    );
    const { onSent } = openDialog();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "one more" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Too many notes sent lately",
    );
    expect(onSent).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Feedback")).toBeTruthy();
  });

  it("closes on Cancel without sending", () => {
    const { onClose } = openDialog();

    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "never mind" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(api.sendFeedback).not.toHaveBeenCalled();
  });
});
