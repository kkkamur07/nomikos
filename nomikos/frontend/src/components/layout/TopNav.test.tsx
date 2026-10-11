import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import type * as ClientModule from "../../api/client";
import { TopNav } from "./TopNav";

const auth = vi.hoisted(() => ({ isAdmin: null as boolean | null }));

vi.mock("../../auth/AuthProvider", () => ({
  useAuthSession: () => ({
    status: "authenticated",
    isAdmin: auth.isAdmin,
    establish: vi.fn(),
    logout: vi.fn(),
  }),
}));

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<ClientModule>();
  return {
    ...actual,
    api: { ...actual.api, sendFeedback: vi.fn() },
  };
});

describe("TopNav admin link", () => {
  beforeEach(() => {
    auth.isAdmin = null;
  });

  it("is offered to an admin", () => {
    auth.isAdmin = true;
    render(<TopNav currentLabel="Projects" />);

    expect(
      screen.getByRole("link", { name: "Admin" }).getAttribute("href"),
    ).toBe("/admin");
  });

  it.each([false, null])("is hidden when isAdmin is %s", (value) => {
    auth.isAdmin = value;
    render(<TopNav currentLabel="Projects" />);

    expect(screen.queryByRole("link", { name: "Admin" })).toBeNull();
    expect(screen.getByRole("link", { name: "Settings" })).toBeTruthy();
  });
});

describe("TopNav feedback entry", () => {
  beforeEach(() => {
    auth.isAdmin = false;
    vi.mocked(api.sendFeedback).mockReset();
  });

  it("is hidden when signed out", () => {
    render(<TopNav currentLabel="Projects" />);

    expect(screen.queryByRole("button", { name: "Feedback" })).toBeNull();
  });

  it("opens the dialog and shows Thanks after sending", async () => {
    vi.mocked(api.sendFeedback).mockResolvedValue({
      id: "f-1",
      created_at: "2026-10-11T12:00:00Z",
      message: "nice",
      page_path: "/projects",
    });
    render(<TopNav currentLabel="Projects" username="ada" />);

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
