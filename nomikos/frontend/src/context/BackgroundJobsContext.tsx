import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  waitForJob,
  type EnqueueJobResponse,
  type JobResponse,
  type JobStatus,
} from "../api/client";
import { toast } from "../components/ui/toast";
import { useJobPolling } from "../hooks/useJobPolling";
import { JobWaitTimeoutError } from "../utils/jobSubscription";
import {
  isTerminalJobStatus,
  jobStatusLabel,
  type PageEditorJobKind,
} from "../components/page-editor/jobProgress";
import {
  enqueuedExecution,
  jobExecution,
  type JobExecution,
} from "../inference/executionTarget";

export type TrackedBackgroundJob = {
  id: string;
  label: string;
  kind: PageEditorJobKind;
  status: JobStatus;
  error: string | null;
  progressLabel: string;
  finishedAt: number | null;
  /**
   * Which **inference host** the platform fixed for this job, and which one the
   * account asked for. Known from the enqueue response onwards, so a tracked
   * job never has a moment without a host to show.
   */
  execution: JobExecution;
};

/**
 * What a job leaving the "done" state announces to anyone that was not the
 * one waiting on it.
 *
 * `trackAndWait`'s caller already gets the full `JobResponse` back from its
 * own promise. This is for the other case: a page editor instance that is
 * mounted right now but did not start the job (or started it but was
 * unmounted, backgrounded, or navigated away when the promise continuation
 * would have run). `documentPartId` is how a listener decides the result is
 * its own to react to, without the context knowing anything about pages.
 */
export type JobCompletionEvent = {
  jobId: string;
  kind: PageEditorJobKind;
  documentPartId: string | null;
  status: JobStatus;
};

type BackgroundJobsContextValue = {
  jobs: TrackedBackgroundJob[];
  activeCount: number;
  panelExpanded: boolean;
  setPanelExpanded: (expanded: boolean) => void;
  /**
   * Takes the whole enqueue response rather than the id alone: the 202 already
   * names the job's **execution target**, and that is what the panel shows
   * from the first render, before any status update arrives.
   */
  trackAndWait: (
    enqueued: EnqueueJobResponse,
    meta: { label: string; kind: PageEditorJobKind },
    options?: { timeoutMs?: number },
  ) => Promise<JobResponse>;
  cancelJob: (jobId: string) => Promise<void>;
  dismissCompleted: () => void;
  /**
   * Registers a listener for `JobCompletionEvent`s and returns the function
   * that unregisters it. Fires for every job that reaches "done", regardless
   * of who started it or whether that caller is still around to see it.
   */
  subscribeToJobCompletion: (
    listener: (event: JobCompletionEvent) => void,
  ) => () => void;
};

const COMPLETED_TTL_MS = 10_000;

const BackgroundJobsContext = createContext<BackgroundJobsContextValue | null>(
  null,
);

function patchTrackedJob(
  job: TrackedBackgroundJob,
  latest: JobResponse,
): TrackedBackgroundJob {
  return {
    ...job,
    status: latest.status,
    error: latest.error,
    progressLabel: jobStatusLabel(latest),
    finishedAt: isTerminalJobStatus(latest.status) ? Date.now() : null,
    // Re-read on every poll rather than only at submission: the host is fixed,
    // but the *status* is part of the announcement ("Failed on your computer").
    execution: jobExecution(latest),
  };
}

export function BackgroundJobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<TrackedBackgroundJob[]>([]);
  const [panelExpanded, setPanelExpanded] = useState(false);
  const timersRef = useRef<Map<string, number>>(new Map());
  // Read by applyJobUpdate to find the kind of a job that just went terminal,
  // without making that callback depend on (and re-create on) `jobs`.
  const jobsRef = useRef<TrackedBackgroundJob[]>(jobs);
  jobsRef.current = jobs;
  const completionListenersRef = useRef<
    Set<(event: JobCompletionEvent) => void>
  >(new Set());

  const subscribeToJobCompletion = useCallback(
    (listener: (event: JobCompletionEvent) => void) => {
      completionListenersRef.current.add(listener);
      return () => {
        completionListenersRef.current.delete(listener);
      };
    },
    [],
  );

  const announceJobCompletion = useCallback(
    (jobId: string, kind: PageEditorJobKind, latest: JobResponse) => {
      // Only "done" says new content landed. Failed and cancelled are terminal
      // too, but there is nothing for a listener to go re-fetch.
      if (latest.status !== "done") return;
      const event: JobCompletionEvent = {
        jobId,
        kind,
        documentPartId: latest.document_part_id,
        status: latest.status,
      };
      for (const listener of completionListenersRef.current) {
        listener(event);
      }
    },
    [],
  );

  const scheduleRemoval = useCallback((jobId: string, status?: JobStatus) => {
    // Keep cancelled jobs visible until the user clears them - they are part
    // of recent history, unlike done/failed which auto-dismiss after a TTL.
    if (status === "cancelled") return;
    const existing = timersRef.current.get(jobId);
    if (existing) window.clearTimeout(existing);
    const timer = window.setTimeout(() => {
      setJobs((current) => current.filter((job) => job.id !== jobId));
      timersRef.current.delete(jobId);
    }, COMPLETED_TTL_MS);
    timersRef.current.set(jobId, timer);
  }, []);

  useEffect(
    () => () => {
      for (const timer of timersRef.current.values()) {
        window.clearTimeout(timer);
      }
      timersRef.current.clear();
    },
    [],
  );

  const applyJobUpdate = useCallback(
    (jobId: string, latest: JobResponse) => {
      // Read before the patch: `trackAndWait` calls this once from
      // `waitForJob`'s onUpdate and once more with its own resolved value, so
      // the same terminal status arrives here twice. Comparing against the
      // status this job had *before* the patch is what makes the announcement
      // fire on the transition into "done" rather than on every repeat of it.
      const previous = jobsRef.current.find((job) => job.id === jobId);
      setJobs((current) =>
        current.map((job) =>
          job.id === jobId ? patchTrackedJob(job, latest) : job,
        ),
      );
      if (isTerminalJobStatus(latest.status)) {
        scheduleRemoval(jobId, latest.status);
        if (previous && !isTerminalJobStatus(previous.status)) {
          announceJobCompletion(jobId, previous.kind, latest);
        }
      }
    },
    [scheduleRemoval, announceJobCompletion],
  );

  const activeJobIds = useMemo(
    () =>
      jobs
        .filter((job) => !isTerminalJobStatus(job.status))
        .map((job) => job.id),
    [jobs],
  );

  useJobPolling(activeJobIds, (updates) => {
    setJobs((current) => {
      const next = current.map((job) => ({ ...job }));
      for (const update of updates) {
        const index = next.findIndex((job) => job.id === update.id);
        if (index < 0) continue;
        const previousStatus = next[index].status;
        const kind = next[index].kind;
        next[index] = patchTrackedJob(next[index], update);
        if (isTerminalJobStatus(update.status)) {
          scheduleRemoval(update.id, update.status);
          // This is the path a backgrounded tab or a remounted component
          // relies on: the SSE stream that would have carried the update to
          // the instance that started the job dropped, this poll is what
          // catches it instead, and the announcement is what makes the catch
          // visible to a *different* mounted instance.
          if (!isTerminalJobStatus(previousStatus)) {
            announceJobCompletion(update.id, kind, update);
          }
        }
      }
      return next;
    });
  });

  const trackAndWait = useCallback(
    async (
      enqueued: EnqueueJobResponse,
      meta: { label: string; kind: PageEditorJobKind },
      options?: { timeoutMs?: number },
    ): Promise<JobResponse> => {
      const jobId = enqueued.job_id;
      setJobs((current) => {
        if (current.some((job) => job.id === jobId)) return current;
        return [
          ...current,
          {
            id: jobId,
            label: meta.label,
            kind: meta.kind,
            status: "pending" as JobStatus,
            error: null,
            progressLabel: "Queued",
            finishedAt: null,
            // Announced from the response that created the job: the target is
            // fixed at submission (ADR 0002), so nothing here is a guess, and
            // a substituted host is stated before the first poll, not after.
            execution: enqueuedExecution(enqueued),
          },
        ];
      });
      setPanelExpanded(false);

      try {
        const job = await waitForJob(jobId, {
          timeoutMs: options?.timeoutMs,
          onUpdate: (latest) => applyJobUpdate(jobId, latest),
        });
        applyJobUpdate(jobId, job);
        return job;
      } catch (err) {
        if (err instanceof JobWaitTimeoutError) {
          // The client stopped waiting, not the server. The job stays
          // active, so the poll loop keeps watching it, and the panel says
          // contact was lost rather than reporting the job as failed.
          setJobs((current) =>
            current.map((job) =>
              job.id === jobId
                ? {
                    ...job,
                    error: "Lost contact with the server; still checking.",
                    progressLabel: "Waiting for the server",
                  }
                : job,
            ),
          );
          throw err;
        }
        const message = err instanceof Error ? err.message : "Job failed";
        setJobs((current) =>
          current.map((job) =>
            job.id === jobId
              ? {
                  ...job,
                  status: "failed",
                  error: message,
                  progressLabel: "Failed",
                  finishedAt: Date.now(),
                  // A failed job says which host it failed on, so the host it
                  // was given has to travel with the new status.
                  execution: { ...job.execution, status: "failed" },
                }
              : job,
          ),
        );
        scheduleRemoval(jobId);
        throw err;
      }
    },
    [applyJobUpdate, scheduleRemoval],
  );

  const dismissCompleted = useCallback(() => {
    setJobs((current) =>
      current.filter((job) => !isTerminalJobStatus(job.status)),
    );
  }, []);

  const cancelJob = useCallback(
    async (jobId: string) => {
      try {
        const latest = await api.cancelJob(jobId);
        applyJobUpdate(jobId, latest);
        toast.success("Job cancelled");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Could not cancel that job",
        );
        throw err;
      }
    },
    [applyJobUpdate],
  );

  const activeCount = jobs.filter(
    (job) => !isTerminalJobStatus(job.status),
  ).length;

  const value = useMemo(
    () => ({
      jobs,
      activeCount,
      panelExpanded,
      setPanelExpanded,
      trackAndWait,
      cancelJob,
      dismissCompleted,
      subscribeToJobCompletion,
    }),
    [
      jobs,
      activeCount,
      panelExpanded,
      trackAndWait,
      cancelJob,
      dismissCompleted,
      subscribeToJobCompletion,
    ],
  );

  return (
    <BackgroundJobsContext.Provider value={value}>
      {children}
    </BackgroundJobsContext.Provider>
  );
}

export function useBackgroundJobs(): BackgroundJobsContextValue {
  const context = useContext(BackgroundJobsContext);
  if (!context) {
    throw new Error(
      "useBackgroundJobs must be used within BackgroundJobsProvider",
    );
  }
  return context;
}
