import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "../ui/toast";
import { api, type JobResponse } from "../../api/client";
import { ApiError } from "../../api/errors";
import { useJobPolling } from "../../hooks/useJobPolling";
import { isTerminalJobStatus } from "../page-editor/jobProgress";
import {
  jobExecution,
  type JobExecution,
} from "../../inference/executionTarget";
import { JobExecutionAnnouncement } from "../JobExecutionAnnouncement";

type TrackedJob = {
  jobId: string;
  status: JobResponse["status"];
  type?: JobResponse["type"];
  error?: string | null;
  /** Which **inference host** runs it; `null` until the platform has answered. */
  execution: JobExecution | null;
};

function shortId(jobId: string): string {
  return jobId.slice(0, 8);
}

export type JobsNoticeProps = {
  projectId: string;
  documentId: string;
  /**
   * Bumped by the page when it queues jobs (the Workflow menu), so the bar
   * re-reads this document's jobs instead of waiting for its own button.
   */
  refreshKey: number;
  enableTestJobs?: boolean;
};

export function JobsNotice({
  projectId,
  documentId,
  refreshKey,
  enableTestJobs = false,
}: JobsNoticeProps) {
  const [trackedJobs, setTrackedJobs] = useState<TrackedJob[]>([]);
  const [enqueueing, setEnqueueing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const failedNotifiedRef = useRef<Set<string>>(new Set());

  const trackJob = useCallback((jobId: string) => {
    setTrackedJobs((prev) => {
      if (prev.some((j) => j.jobId === jobId)) return prev;
      return [...prev, { jobId, status: "pending", execution: null }];
    });
    setExpanded(true);
  }, []);

  // This document's unfinished jobs, from the project jobs list: the ones the
  // Workflow menu just queued, and any started from another tab or the editor.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const page = await api.listProjectJobsPage(projectId);
        if (cancelled) return;
        const mine = page.items.filter(
          (job) =>
            job.document_id === documentId && !isTerminalJobStatus(job.status),
        );
        if (mine.length === 0) return;
        setTrackedJobs((prev) => {
          const known = new Set(prev.map((row) => row.jobId));
          const added = mine
            .filter((job) => !known.has(job.id))
            .map((job) => ({
              jobId: job.id,
              status: job.status,
              type: job.type,
              error: job.error,
              execution: jobExecution(job),
            }));
          return added.length > 0 ? [...prev, ...added] : prev;
        });
        setExpanded(true);
      } catch {
        // The bar is a convenience; the project jobs panel stays the record.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, documentId, refreshKey]);

  const activeJobIds = trackedJobs
    .filter((j) => !isTerminalJobStatus(j.status))
    .map((j) => j.jobId);

  useJobPolling(activeJobIds, (jobs) => {
    setTrackedJobs((prev) => {
      const next = prev.map((row) => ({ ...row }));
      for (const job of jobs) {
        const idx = next.findIndex((r) => r.jobId === job.id);
        if (idx < 0) continue;
        next[idx] = {
          jobId: job.id,
          status: job.status,
          type: job.type,
          error: job.error,
          execution: jobExecution(job),
        };
        if (
          job.status === "failed" &&
          job.error &&
          !failedNotifiedRef.current.has(job.id)
        ) {
          failedNotifiedRef.current.add(job.id);
          toast.error(`Job failed: ${job.error}`);
        }
      }
      return next;
    });
  });

  const handleRunTestJob = async () => {
    setEnqueueing(true);
    try {
      const { job_id } = await api.enqueueTestJob();
      trackJob(job_id);
      toast.success(`Test job enqueued (${shortId(job_id)})`);
    } catch (err) {
      const msg =
        err instanceof ApiError ? err.message : "Failed to enqueue test job";
      toast.error(msg);
    } finally {
      setEnqueueing(false);
    }
  };

  const activeCount = trackedJobs.filter(
    (j) => !isTerminalJobStatus(j.status),
  ).length;
  const statusHint =
    activeCount > 0
      ? `${activeCount} running`
      : trackedJobs.length > 0
        ? "All jobs complete"
        : "Segment, transcribe, or edit layout";

  return (
    <div>
      <div className="notice-inline" role="status">
        <strong>Jobs</strong>
        <span>{statusHint}</span>
        <span className="spacer" />
        {enableTestJobs && (
          <button
            type="button"
            className="btn btn-outline btn-xs"
            onClick={() => void handleRunTestJob()}
            disabled={enqueueing}
          >
            {enqueueing ? "Running…" : "Run job"}
          </button>
        )}
        {trackedJobs.length > 0 && (
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? "Hide" : "Details"}
          </button>
        )}
      </div>
      {expanded && trackedJobs.length > 0 && (
        <ul
          className="part-list"
          style={{ marginBottom: 20 }}
          aria-label="Job status"
        >
          {trackedJobs.map((job) => (
            <li
              key={job.jobId}
              className="part-row"
              style={{ listStyle: "none" }}
            >
              <div className="part-info">
                <div className="part-num">{shortId(job.jobId)}</div>
                <div className="part-desc">
                  {job.type ?? "job"} · {job.status}
                  {job.error ? `: ${job.error}` : ""}
                </div>
                <JobExecutionAnnouncement
                  execution={job.execution}
                  className="part-desc"
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
