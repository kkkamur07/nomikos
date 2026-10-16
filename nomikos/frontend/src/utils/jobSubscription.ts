import { fetchWithAuthRecovery } from "../api/http";
import type { JobResponse } from "../api/types";

const JOB_NO_EVENT_TIMEOUT_MS = 12_000;

type GetJob = (jobId: string) => Promise<JobResponse>;
type JobListener = (job: JobResponse) => void;

export type JobSubscriptionOptions = {
  eventsUrl: string;
  getJob: GetJob;
  onUpdate: JobListener;
  intervalMs: number;
  noEventTimeoutMs?: number;
};

type JobSubscriptionOwner = {
  jobId: string;
  eventsUrl: string;
  getJob: GetJob;
  intervalMs: number;
  noEventTimeoutMs: number;
  listeners: Set<JobListener>;
  controller: AbortController | null;
  heartbeatTimer: number | null;
  pollingTimer: number | null;
  polling: boolean;
  stopped: boolean;
  lastJob: JobResponse | null;
};

const owners = new Map<string, JobSubscriptionOwner>();

function isTerminal(job: JobResponse): boolean {
  return (
    job.status === "done" ||
    job.status === "failed" ||
    job.status === "cancelled"
  );
}

function parseSseChunk(buffer: string): {
  events: string[];
  remainder: string;
} {
  const parts = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const remainder = parts.pop() ?? "";
  const events = parts
    .map((part) =>
      part
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n"),
    )
    .filter(Boolean);
  return { events, remainder };
}

function stop(owner: JobSubscriptionOwner): void {
  if (owner.stopped) return;
  owner.stopped = true;
  owner.controller?.abort();
  if (owner.heartbeatTimer !== null) window.clearTimeout(owner.heartbeatTimer);
  if (owner.pollingTimer !== null) window.clearInterval(owner.pollingTimer);
  if (owners.get(owner.jobId) === owner) owners.delete(owner.jobId);
}

function emit(owner: JobSubscriptionOwner, job: JobResponse): void {
  const previous = owner.lastJob;
  if (
    previous &&
    previous.status === job.status &&
    previous.updated_at === job.updated_at
  )
    return;
  owner.lastJob = job;
  for (const listener of owner.listeners) listener(job);
  if (isTerminal(job)) stop(owner);
}

function startPolling(owner: JobSubscriptionOwner): void {
  if (owner.stopped || owner.pollingTimer !== null) return;
  owner.controller?.abort();

  const poll = async () => {
    if (owner.stopped || owner.polling) return;
    owner.polling = true;
    try {
      emit(owner, await owner.getJob(owner.jobId));
    } catch {
      // Keep the next scheduled poll available while a transient request fails.
    } finally {
      owner.polling = false;
    }
  };

  void poll();
  owner.pollingTimer = window.setInterval(() => void poll(), owner.intervalMs);
}

function touchHeartbeat(owner: JobSubscriptionOwner): void {
  if (owner.stopped || owner.pollingTimer !== null) return;
  if (owner.heartbeatTimer !== null) window.clearTimeout(owner.heartbeatTimer);
  owner.heartbeatTimer = window.setTimeout(
    () => startPolling(owner),
    owner.noEventTimeoutMs,
  );
}

async function openEventStream(
  owner: JobSubscriptionOwner,
  controller: AbortController,
): Promise<ReadableStreamDefaultReader<Uint8Array> | null> {
  // The Authorization header always carries the current token:
  // `fetchWithAuthRecovery` sets it from storage on every attempt and
  // refreshes it after a 401, so a token captured at subscribe time would
  // only ever be older than what is sent.
  const response = await fetchWithAuthRecovery(owner.eventsUrl, {
    headers: new Headers({ Accept: "text/event-stream" }),
    signal: controller.signal,
    credentials: "include",
  });
  if (!response.ok || !response.body) {
    startPolling(owner);
    return null;
  }
  return response.body.getReader();
}

async function pumpEventStream(
  owner: JobSubscriptionOwner,
  controller: AbortController,
  reader: ReadableStreamDefaultReader<Uint8Array>,
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = "";
  while (!owner.stopped && !controller.signal.aborted) {
    const { done, value } = await reader.read();
    if (done) break;
    touchHeartbeat(owner);
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.remainder;
    for (const event of parsed.events) {
      emit(owner, JSON.parse(event) as JobResponse);
      if (owner.stopped) return;
    }
  }
  if (!owner.stopped && !controller.signal.aborted) startPolling(owner);
}

async function startSse(owner: JobSubscriptionOwner): Promise<void> {
  if (typeof fetch === "undefined" || typeof ReadableStream === "undefined") {
    startPolling(owner);
    return;
  }

  const controller = new AbortController();
  owner.controller = controller;
  touchHeartbeat(owner);

  try {
    const reader = await openEventStream(owner, controller);
    if (!reader) return;
    touchHeartbeat(owner);
    await pumpEventStream(owner, controller, reader);
  } catch (error) {
    const aborted =
      error instanceof DOMException && error.name === "AbortError";
    if (!owner.stopped && !aborted) startPolling(owner);
  }
}

/**
 * Opens at most one stream or polling loop for each job ID. Every subscriber
 * receives the same update stream and the owner is removed after its final
 * listener unsubscribes or the job reaches a terminal state.
 */
export function subscribeToJob(
  jobId: string,
  options: JobSubscriptionOptions,
): () => void {
  let owner = owners.get(jobId);
  if (!owner) {
    owner = {
      jobId,
      eventsUrl: options.eventsUrl,
      getJob: options.getJob,
      intervalMs: options.intervalMs,
      noEventTimeoutMs: options.noEventTimeoutMs ?? JOB_NO_EVENT_TIMEOUT_MS,
      listeners: new Set(),
      controller: null,
      heartbeatTimer: null,
      pollingTimer: null,
      polling: false,
      stopped: false,
      lastJob: null,
    };
    owners.set(jobId, owner);
    void startSse(owner);
  }

  owner.listeners.add(options.onUpdate);
  return () => {
    owner?.listeners.delete(options.onUpdate);
    if (owner && owner.listeners.size === 0) stop(owner);
  };
}

/**
 * The client stopped waiting, not the server: the job may still be running,
 * so callers must not report it as failed. Lets `trackAndWait` tell the two
 * apart without matching on the message text.
 */
export class JobWaitTimeoutError extends Error {
  constructor() {
    super("Job timed out.");
    this.name = "JobWaitTimeoutError";
  }
}

export function waitForSubscribedJob(
  jobId: string,
  options: Omit<JobSubscriptionOptions, "onUpdate"> & {
    timeoutMs?: number;
    onUpdate?: JobListener;
  },
): Promise<JobResponse> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      unsubscribe();
      reject(new JobWaitTimeoutError());
    }, options.timeoutMs ?? 120_000);

    const unsubscribe = subscribeToJob(jobId, {
      ...options,
      onUpdate: (job) => {
        options.onUpdate?.(job);
        if (job.status === "done" || job.status === "cancelled") {
          window.clearTimeout(timeout);
          unsubscribe();
          resolve(job);
        } else if (job.status === "failed") {
          window.clearTimeout(timeout);
          unsubscribe();
          reject(new Error(job.error ?? "Job failed."));
        }
      },
    });
  });
}
