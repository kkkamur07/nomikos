/**
 * Job endpoints and the job waiter.
 *
 * `waitForJob` lives here, next to the job routes, rather than in the
 * `client` barrel: `utils/jobSubscription` owns the stream, this module owns
 * the routes it falls back to, and the barrel only re-exports. That keeps the
 * dependency one-way (`jobs` -> `jobSubscription` -> `http`).
 */
import { JOB_WAIT_POLL_INTERVAL_MS } from "../utils/jobPolling";
import { waitForSubscribedJob } from "../utils/jobSubscription";
import { collectCursorPages } from "./cursorPagination";
import { API_BASE_URL, apiRequest } from "./http";
import { cursorQuery } from "./paths";
import type { JobResponse, ListPageOptions, PageResponse } from "./types";

export const jobsApi = {
  getJob: (jobId: string) =>
    apiRequest<JobResponse>(`/jobs/${encodeURIComponent(jobId)}`),
  cancelJob: (jobId: string) =>
    apiRequest<JobResponse>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    }),

  /**
   * Remove finished jobs (done / failed / cancelled) from a project's history.
   * Active jobs are left alone by the server.
   */
  clearProjectJobHistory: (projectId: string) =>
    apiRequest<{ deleted: number }>(
      `/jobs/history?project_id=${encodeURIComponent(projectId)}`,
      { method: "DELETE" },
    ),

  listProjectJobsPage: (projectId: string, options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<JobResponse>>(
      `/projects/${encodeURIComponent(projectId)}/jobs${query ? `?${query}` : ""}`,
      { signal: options.signal },
    );
  },

  listProjectJobs: (
    projectId: string,
    options?: { maxPages?: number; signal?: AbortSignal },
  ) =>
    collectCursorPages(
      (pageOptions) => jobsApi.listProjectJobsPage(projectId, pageOptions),
      options,
    ),
};

export async function waitForJob(
  jobId: string,
  options?: { timeoutMs?: number; onUpdate?: (job: JobResponse) => void },
): Promise<JobResponse> {
  return waitForSubscribedJob(jobId, {
    ...options,
    eventsUrl: `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/events`,
    getJob: jobsApi.getJob,
    intervalMs: JOB_WAIT_POLL_INTERVAL_MS,
  });
}
