import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EnqueueJobResponse } from "../api/client";
import { JobWaitTimeoutError } from "../utils/jobSubscription";
import {
  BackgroundJobsProvider,
  useBackgroundJobs,
} from "./BackgroundJobsContext";

vi.mock("../api/client", () => ({
  api: { cancelJob: vi.fn() },
  waitForJob: vi.fn(),
}));
vi.mock("../hooks/useJobPolling", () => ({
  useJobPolling: () => undefined,
}));

import { waitForJob } from "../api/client";

const ENQUEUED: EnqueueJobResponse = {
  job_id: "job-1",
  execution_target: "cloud",
  preferred_execution_target: "cloud",
  execution_target_substituted: false,
};

function TrackButton() {
  const { jobs, trackAndWait } = useBackgroundJobs();
  return (
    <>
      <button
        onClick={() =>
          void trackAndWait(ENQUEUED, {
            label: "Segment",
            kind: "segment",
          }).catch(() => {})
        }
      >
        track
      </button>
      <output data-testid="jobs">
        {jobs.map((job) => `${job.status}:${job.progressLabel}`).join(",")}
      </output>
    </>
  );
}

describe("trackAndWait client timeout", () => {
  beforeEach(() => {
    vi.mocked(waitForJob).mockReset();
  });

  it("keeps the job active with a lost-contact label instead of failing it", async () => {
    vi.mocked(waitForJob).mockRejectedValue(new JobWaitTimeoutError());
    render(
      <BackgroundJobsProvider>
        <TrackButton />
      </BackgroundJobsProvider>,
    );

    screen.getByRole("button", { name: "track" }).click();
    await waitFor(() =>
      expect(screen.getByTestId("jobs")).toHaveTextContent(
        "pending:Waiting for the server",
      ),
    );
  });

  it("marks a genuinely failed wait as failed", async () => {
    vi.mocked(waitForJob).mockRejectedValue(new Error("Job failed."));
    render(
      <BackgroundJobsProvider>
        <TrackButton />
      </BackgroundJobsProvider>,
    );

    screen.getByRole("button", { name: "track" }).click();
    await waitFor(() =>
      expect(screen.getByTestId("jobs")).toHaveTextContent("failed:Failed"),
    );
  });
});
