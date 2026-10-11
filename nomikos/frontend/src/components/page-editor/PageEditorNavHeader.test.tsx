import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import type * as ClientModule from "../../api/client";
import { PageEditorNavHeader } from "./PageEditorNavHeader";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<ClientModule>();
  return {
    ...actual,
    api: { ...actual.api, sendFeedback: vi.fn() },
  };
});

beforeEach(() => {
  vi.mocked(api.sendFeedback).mockReset();
});

describe("PageEditorNavHeader feedback", () => {
  it("renders the Feedback control", () => {
    render(<PageEditorNavHeader backHref="/projects/1/documents/2" />);

    expect(screen.getByRole("button", { name: "Feedback" })).toBeTruthy();
  });

  it("opens the dialog and shows Thanks after sending", async () => {
    vi.mocked(api.sendFeedback).mockResolvedValue({
      id: "f-1",
      created_at: "2026-10-11T12:00:00Z",
      message: "nice",
      page_path: window.location.pathname,
    });
    render(<PageEditorNavHeader backHref="/projects/1/documents/2" />);

    fireEvent.click(screen.getByRole("button", { name: "Feedback" }));
    fireEvent.change(screen.getByLabelText("Feedback"), {
      target: { value: "nice" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("button", { name: "Thanks" })).toBeTruthy();
    expect(api.sendFeedback).toHaveBeenCalledWith({
      message: "nice",
      page_path: window.location.pathname,
    });
  });
});
