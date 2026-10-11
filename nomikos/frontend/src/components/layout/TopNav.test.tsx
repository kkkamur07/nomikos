import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
