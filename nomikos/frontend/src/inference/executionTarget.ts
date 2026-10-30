/**
 * How a job announces its **execution target**, and how a refused submission
 * explains itself.
 *
 * The announcement line is not cosmetic - it is the entire user interface for
 * this feature (ADR 0002). It belongs on the job rather than in a toast,
 * because a researcher who looks away must still be able to read where their
 * job went. It says which **inference host** runs the job; when the preferred
 * host had no **capacity** it says so plainly rather than substituting in
 * silence; and a failed job names the host it failed on.
 *
 * Composing the sentence is kept here, apart from any component, so the four
 * states a job can announce are one testable function rather than four pieces
 * of markup.
 */
import { ApiError } from "../api/errors";
import type {
  EnqueueJobResponse,
  ExecutionTarget,
  JobResponse,
  JobStatus,
} from "../api/client";

/** Reads after a preposition: "Running **on your computer**". */
const INFERENCE_HOST_PHRASE: Record<ExecutionTarget, string> = {
  local: "on your computer",
  cloud: "in the cloud",
};

/** Reads as a subject: "**Your computer** had no capacity…". */
const INFERENCE_HOST_NOUN: Record<ExecutionTarget, string> = {
  local: "your computer",
  cloud: "the cloud",
};

/** Short label for a column or badge, where a sentence does not fit. */
export const INFERENCE_HOST_LABEL: Record<ExecutionTarget, string> = {
  local: "Your computer",
  cloud: "Cloud",
};

/**
 * The three fields every job carries, and nothing else. Typed structurally so
 * a `JobResponse` satisfies it without a cast, and so the announcement can be
 * composed for a job that a component is still holding in flight.
 */
export type JobExecution = {
  execution_target: ExecutionTarget;
  preferred_execution_target: ExecutionTarget;
  execution_target_substituted: boolean;
  status?: JobStatus;
};

export function jobExecution(job: JobResponse): JobExecution {
  return {
    execution_target: job.execution_target,
    preferred_execution_target: job.preferred_execution_target,
    execution_target_substituted: job.execution_target_substituted ?? false,
    status: job.status,
  };
}

/**
 * The announcement a job carries from the moment it is submitted.
 *
 * The 202 from an enqueue route names the host the platform fixed for the
 * job, so the announcement is composed here, before any status update: there
 * is no window in which a submitted job shows no host. The status is
 * "pending" because that is what a job is until an agent claims it, and the
 * first update replaces the whole value rather than patching it.
 */
export function enqueuedExecution(enqueued: EnqueueJobResponse): JobExecution {
  return {
    execution_target: enqueued.execution_target,
    preferred_execution_target: enqueued.preferred_execution_target,
    execution_target_substituted: enqueued.execution_target_substituted,
    status: "pending",
  };
}

function hostClause(target: ExecutionTarget, status?: JobStatus): string {
  const where = INFERENCE_HOST_PHRASE[target];
  if (status === "failed") return `Failed ${where}.`;
  if (status === "cancelled") return `Cancelled ${where}.`;
  if (status === "done") return `Ran ${where}.`;
  return `Running ${where}.`;
}

/**
 * The announcement for one job: which host runs it, and - when the account
 * preference could not be honoured - that the preferred host had no capacity.
 *
 * Four states, one sentence each:
 * - chosen:      "Running on your computer."
 * - substituted: "Running in the cloud. You asked for your computer, which had
 *                 no capacity when this job was submitted."
 * - failed:      "Failed on your computer."
 * - refused:     no job exists, so no announcement - see
 *                {@link submissionRefusalExplanation}.
 */
export function executionAnnouncement(job: JobExecution): string {
  const chosen = hostClause(job.execution_target, job.status);
  if (!job.execution_target_substituted) return chosen;
  return `${chosen} You asked for ${
    INFERENCE_HOST_NOUN[job.preferred_execution_target]
  }, which had no capacity when this job was submitted.`;
}

/**
 * One sentence for a batch of jobs queued together, or `null` when there is
 * no one sentence to say.
 *
 * Every job in a batch was decided against the same capacity reading and the
 * same account preference, so they nearly always share a host, and then the
 * batch says it once. When they do not (a model bound to one page is eligible
 * for one host only) each job still announces itself on the jobs list, and
 * the batch says nothing rather than something that is only mostly true.
 * `null` for an empty batch too: nothing was submitted, so nothing runs.
 */
export function batchExecutionAnnouncement(
  jobs: readonly JobExecution[],
): string | null {
  const [first] = jobs;
  if (!first) return null;
  if (jobs.length === 1) return executionAnnouncement(first);
  const oneHost = jobs.every(
    (job) =>
      job.execution_target === first.execution_target &&
      job.preferred_execution_target === first.preferred_execution_target &&
      job.execution_target_substituted === first.execution_target_substituted,
  );
  if (!oneHost) return null;
  const where = INFERENCE_HOST_PHRASE[first.execution_target];
  if (!first.execution_target_substituted) return `Running ${where}.`;
  return `Running ${where}. You asked for ${
    INFERENCE_HOST_NOUN[first.preferred_execution_target]
  }, which had no capacity when these jobs were submitted.`;
}

/**
 * Whether a submission was refused because no **inference host** had capacity.
 *
 * The platform answers 409 with a message naming the situation, so there is
 * nothing to reconstruct here - the check exists so the caller can route it to
 * a standing explanation instead of a transient error toast.
 */
export function isSubmissionRefusal(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409;
}

/**
 * The refusal, as something a researcher can act on, or `null` when the error
 * is an ordinary failure that belongs on the usual error path.
 */
export function submissionRefusalExplanation(error: unknown): string | null {
  if (!isSubmissionRefusal(error)) return null;
  const message = (error as ApiError).message?.trim();
  return message
    ? message
    : "No inference host had capacity, so this job was not submitted.";
}
