import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EnqueueJobResponse, JobResponse } from "../api/client";
import type * as ClientModule from "../api/client";
import { waitForJob } from "../api/client";
import { JobWaitTimeoutError } from "../utils/jobSubscription";
import type { BackgroundJobsContextValue } from "./BackgroundJobsContext";
import {
  BackgroundJobsProvider,
  useBackgroundJobs,
} from "./BackgroundJobsContext";

vi.mock("../hooks/useJobPolling", () => ({ useJobPolling: vi.fn() }));
vi.mock("../api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof ClientModule>()),
  waitForJob: vi.fn(),
}));

const mockWaitForJob = vi.mocked(waitForJob);

const ENQUEUED: EnqueueJobResponse = {
  job_id: "11111111-1111-4111-8111-111111111111",
  execution_target: "local",
  preferred_execution_target: "local",
  execution_target_substituted: false,
};

function doneResponse(): JobResponse {
  return {
    id: ENQUEUED.job_id,
    status: "done",
    error: null,
    type: "test",
    payload: {},
    result: {},
    document_id: null,
    document_part_id: null,
    execution_target: "local",
    preferred_execution_target: "local",
    execution_target_substituted: false,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
  };
}
function probe(): () => BackgroundJobsContextValue {
  let current!: BackgroundJobsContextValue;
  function Probe() {
    const value = useBackgroundJobs();
    useEffect(() => {
      current = value;
    }, [value]);
    return null;
  }
  render(
    <BackgroundJobsProvider>
      <Probe />
    </BackgroundJobsProvider>,
  );
  return () => current;
}

describe("BackgroundJobsProvider all-terminal auto-dismiss", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.resetAllMocks();
  });

  it("clears the collapsed pill 8 s after every job goes terminal", async () => {
    mockWaitForJob.mockResolvedValue(doneResponse());
    const get = probe();

    await act(async () => {
      await get().trackAndWait(ENQUEUED, {
        label: "Segment page",
        kind: "segmentation",
      });
    });
    expect(get().jobs).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(7_999);
    });
    expect(get().jobs).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(get().jobs).toHaveLength(0);
  });

  it("never auto-dismisses while a job is still running", () => {
    mockWaitForJob.mockReturnValue(new Promise<JobResponse>(() => {}));
    const get = probe();

    act(() => {
      void get().trackAndWait(ENQUEUED, {
        label: "Segment page",
        kind: "segmentation",
      });
    });
    expect(get().jobs).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(30_000);
    });
    expect(get().jobs).toHaveLength(1);
  });

  it("never auto-dismisses while the panel is expanded", async () => {
    mockWaitForJob.mockResolvedValue(doneResponse());
    const get = probe();

    await act(async () => {
      await get().trackAndWait(ENQUEUED, {
        label: "Segment page",
        kind: "segmentation",
      });
    });
    act(() => {
      get().setPanelExpanded(true);
    });
    // Past the 8 s pill timer but before the pre-existing 10 s per-job
    // removal: still here means the expanded state blocked the new timer.
    act(() => {
      vi.advanceTimersByTime(9_000);
    });
    expect(get().jobs).toHaveLength(1);
  });
});

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
