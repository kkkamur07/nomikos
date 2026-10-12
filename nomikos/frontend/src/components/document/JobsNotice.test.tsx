import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JobsNotice } from "./JobsNotice";

const listProjectJobsPage = vi.fn();
const enqueueTestJob = vi.fn();

vi.mock("../../api/client", () => ({
  api: {
    listProjectJobsPage: (...args: unknown[]) => listProjectJobsPage(...args),
    enqueueTestJob: (...args: unknown[]) => enqueueTestJob(...args),
  },
}));

vi.mock("../../hooks/useJobPolling", () => ({
  useJobPolling: () => {},
}));

vi.mock("../ui/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

function job(id: string, documentId: string, status: string) {
  return {
    id,
    type: "transcribe",
    status,
    error: null,
    document_id: documentId,
    document_part_id: "part-1",
    execution_target: "cloud",
    preferred_execution_target: "cloud",
  };
}

describe("JobsNotice", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    listProjectJobsPage.mockResolvedValue({ items: [], next_cursor: null });
  });

  it("lists a job queued for this document after the Workflow menu runs", async () => {
    const { rerender } = render(
      <JobsNotice
        projectId="project-1"
        documentId="document-1"
        refreshKey={0}
      />,
    );
    await waitFor(() => expect(listProjectJobsPage).toHaveBeenCalled());
    expect(
      screen.getByText("Segment, transcribe, or edit layout"),
    ).toBeTruthy();

    listProjectJobsPage.mockResolvedValue({
      items: [
        job("aaaaaaaa-1111", "document-1", "pending"),
        job("bbbbbbbb-2222", "other-document", "running"),
      ],
      next_cursor: null,
    });
    rerender(
      <JobsNotice
        projectId="project-1"
        documentId="document-1"
        refreshKey={1}
      />,
    );

    expect(await screen.findByText("1 running")).toBeTruthy();
    expect(screen.getByText("aaaaaaaa")).toBeTruthy();
    expect(screen.getByText(/transcribe · pending/)).toBeTruthy();
    expect(screen.queryByText("bbbbbbbb")).toBeNull();
  });

  it("still tracks the test job it starts itself", async () => {
    enqueueTestJob.mockResolvedValue({ job_id: "cccccccc-3333" });
    render(
      <JobsNotice
        projectId="project-1"
        documentId="document-1"
        refreshKey={0}
        enableTestJobs
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Run job" }));

    expect(await screen.findByText("cccccccc")).toBeTruthy();
  });
});
