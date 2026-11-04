import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { ProjectSharingPanel } from "./ProjectSharingPanel";

vi.mock("../../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      listProjectCollaborators: vi.fn(),
      shareProject: vi.fn(),
      unshareProject: vi.fn(),
    },
  };
});

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const FRIEND = {
  id: "user-2",
  username: "friend",
  email: "friend@example.org",
};

describe("ProjectSharingPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listProjectCollaborators).mockResolvedValue([]);
    vi.mocked(api.shareProject).mockResolvedValue(undefined);
    vi.mocked(api.unshareProject).mockResolvedValue(undefined);
  });

  it("shares by email when the box holds an address, then shows the person", async () => {
    vi.mocked(api.listProjectCollaborators)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([FRIEND]);
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("Not shared with anyone yet.");

    fireEvent.change(screen.getByLabelText("Email or username"), {
      target: { value: " Friend@Example.org " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() =>
      expect(api.shareProject).toHaveBeenCalledWith("project-1", {
        identifier: "Friend@Example.org",
      }),
    );
    expect(await screen.findByText("friend")).toBeInTheDocument();
    expect(screen.getByText("friend@example.org")).toBeInTheDocument();
    expect(screen.getByLabelText("Email or username")).toHaveValue("");
  });

  it("shares by username", async () => {
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("Not shared with anyone yet.");

    fireEvent.change(screen.getByLabelText("Email or username"), {
      target: { value: "friend" },
    });
    fireEvent.submit(
      screen.getByLabelText("Email or username").closest("form")!,
    );

    await waitFor(() =>
      expect(api.shareProject).toHaveBeenCalledWith("project-1", {
        identifier: "friend",
      }),
    );
  });

  it("does not mistake a username containing an @ for an email address", async () => {
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("Not shared with anyone yet.");

    fireEvent.change(screen.getByLabelText("Email or username"), {
      target: { value: "greek@corpus" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    // Sent verbatim for the server to resolve. Classifying it here as an email
    // would fail EmailStr validation and lose a legitimate account.
    await waitFor(() =>
      expect(api.shareProject).toHaveBeenCalledWith("project-1", {
        identifier: "greek@corpus",
      }),
    );
  });

  it("surfaces the API's explanation when the email has no account", async () => {
    const { toast } = await import("../ui/toast");
    vi.mocked(api.shareProject).mockRejectedValue(
      new ApiError("No account is registered under nobody@example.org", 404),
    );
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("Not shared with anyone yet.");

    fireEvent.change(screen.getByLabelText("Email or username"), {
      target: { value: "nobody@example.org" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        "No account is registered under nobody@example.org",
        expect.objectContaining({ status: 404 }),
      ),
    );
    // The draft is kept so the address can be corrected instead of retyped.
    expect(screen.getByLabelText("Email or username")).toHaveValue(
      "nobody@example.org",
    );
  });

  it("removes a collaborator by id and drops them from the list", async () => {
    vi.mocked(api.listProjectCollaborators).mockResolvedValue([FRIEND]);
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("friend");

    fireEvent.click(screen.getByRole("button", { name: "Remove friend" }));

    // By id, not username: a username may contain a "/", which would split the
    // path segment and 404 however it is encoded.
    await waitFor(() =>
      expect(api.unshareProject).toHaveBeenCalledWith("project-1", "user-2"),
    );
    expect(
      await screen.findByText("Not shared with anyone yet."),
    ).toBeInTheDocument();
  });

  it("removes a collaborator whose username contains a slash", async () => {
    const odd = {
      id: "user-3",
      username: "scribe/anna",
      email: "anna@example.org",
    };
    vi.mocked(api.listProjectCollaborators).mockResolvedValue([odd]);
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("scribe/anna");

    fireEvent.click(screen.getByRole("button", { name: "Remove scribe/anna" }));

    await waitFor(() =>
      expect(api.unshareProject).toHaveBeenCalledWith("project-1", "user-3"),
    );
    expect(
      await screen.findByText("Not shared with anyone yet."),
    ).toBeInTheDocument();
  });

  it("puts the cursor in the box when the header asks it to", async () => {
    const { rerender } = render(
      <ProjectSharingPanel projectId="project-1" focusToken={0} />,
    );
    await screen.findByText("Not shared with anyone yet.");
    expect(screen.getByLabelText("Email or username")).not.toHaveFocus();

    rerender(<ProjectSharingPanel projectId="project-1" focusToken={1} />);

    expect(screen.getByLabelText("Email or username")).toHaveFocus();
  });

  it("re-focuses the box each time the header asks again", async () => {
    const { rerender } = render(
      <ProjectSharingPanel projectId="project-1" focusToken={1} />,
    );
    await screen.findByText("Not shared with anyone yet.");
    screen.getByLabelText("Email or username").blur();

    rerender(<ProjectSharingPanel projectId="project-1" focusToken={2} />);

    expect(screen.getByLabelText("Email or username")).toHaveFocus();
  });

  it("does not submit an empty box", async () => {
    render(<ProjectSharingPanel projectId="project-1" />);
    await screen.findByText("Not shared with anyone yet.");

    expect(screen.getByRole("button", { name: "Share" })).toBeDisabled();
    fireEvent.submit(
      screen.getByLabelText("Email or username").closest("form")!,
    );

    expect(api.shareProject).not.toHaveBeenCalled();
  });
});
