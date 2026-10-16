import { useEffect, useRef } from "react";
import { API_BASE_URL, api, type JobResponse } from "../api/client";
import { JOB_NOTICE_POLL_INTERVAL_MS } from "../utils/jobPolling";
import { subscribeToJob } from "../utils/jobSubscription";

export function useJobPolling(
  jobIds: string[],
  onUpdate: (jobs: JobResponse[]) => void,
  options?: { intervalMs?: number; enabled?: boolean },
): void {
  const intervalMs = options?.intervalMs ?? JOB_NOTICE_POLL_INTERVAL_MS;
  const enabled = options?.enabled ?? jobIds.length > 0;
  const onUpdateRef = useRef(onUpdate);
  onUpdateRef.current = onUpdate;
  const activeKey = jobIds.join(",");

  useEffect(() => {
    if (!enabled || !activeKey) return;

    const cleanups = Array.from(new Set(activeKey.split(","))).map((jobId) =>
      subscribeToJob(jobId, {
        eventsUrl: `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}`,
        getJob: api.getJob,
        intervalMs,
        onUpdate: (job) => onUpdateRef.current([job]),
      }),
    );

    return () => {
      for (const cleanup of cleanups) {
        cleanup();
      }
    };
  }, [activeKey, enabled, intervalMs]);
}
