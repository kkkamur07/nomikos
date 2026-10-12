import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ProjectResponse } from "../../api/client";
import { ProjectsTable } from "./ProjectsTable";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

function project(
  id: string,
  ownerUsername: string | null | undefined,
): ProjectResponse {
  return {
    id,
    name: `Project ${id}`,
    slug: `project-${id}`,
    guidelines: null,
    owner_id: "d76a1fc0-0000-0000-0000-000000000000",
    owner_username: ownerUsername,
    document_count: 1,
    created_at: "2026-10-01T10:00:00Z",
    updated_at: "2026-10-01T10:00:00Z",
  };
}

describe("ProjectsTable Owner column", () => {
  it("names the owner, and says - when the name is unknown", () => {
    render(
      <ProjectsTable
        id="shared"
        caption="Shared with you"
        projects={[project("a", "alice"), project("b", undefined)]}
        userId="me"
        loading={false}
        emptyText="None"
        showOwner
      />,
    );

    expect(screen.getByText("alice")).toBeTruthy();
    expect(screen.getByText("-")).toBeTruthy();
    expect(screen.queryByText("d76a1fc0")).toBeNull();
  });
});
