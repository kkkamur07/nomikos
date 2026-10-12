import type { DocumentBatchJobResponse } from "../../api/client";
import {
  batchExecutionAnnouncement,
  enqueuedExecution,
} from "../../inference/executionTarget";

export function pageCountLabel(count: number): string {
  return `${count} page${count === 1 ? "" : "s"}`;
}

/**
 * What a 202 from a batch route says out loud.
 *
 * `skipped` is reported rather than swallowed. "Queued 4, skipped 14" and
 * "queued 4" describe different documents, and a person who expected all 18
 * pages to run needs to be told which one they are looking at.
 *
 * The host comes from the response too. The target of every job in the batch
 * was fixed at submission (ADR 0002), so "Queued 4 pages. Running in the
 * cloud." is a report, not a prediction, and a substituted host is said here,
 * at the click, and not left for the jobs list to reveal later.
 */
export function batchQueuedMessage(result: DocumentBatchJobResponse): string {
  if (result.queued === 0) {
    return result.skipped > 0
      ? `Nothing to run. ${pageCountLabel(result.skipped)} skipped.`
      : "Nothing to run.";
  }
  const queued =
    result.skipped > 0
      ? `Queued ${pageCountLabel(result.queued)}. Skipped ${result.skipped}.`
      : `Queued ${pageCountLabel(result.queued)}.`;
  const host = batchExecutionAnnouncement(result.jobs.map(enqueuedExecution));
  return host ? `${queued} ${host}` : queued;
}

/**
 * The line the publish confirm shows: "18 pages, 6 reviewed, 12 not".
 *
 * Publishing is the one action here whose blast radius is other people, so the
 * split between checked and unchecked pages is put in front of the owner
 * before they agree to it rather than after.
 */
export function publishConfirmSummary(total: number, reviewed: number): string {
  const unreviewed = Math.max(total - reviewed, 0);
  return `${pageCountLabel(total)}, ${reviewed} reviewed, ${unreviewed} not`;
}

/** "updated 2 hours ago", in the coarsest unit that is still true. */
export function relativeUpdatedLabel(
  iso: string,
  now: Date = new Date(),
): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "updated recently";
  const seconds = Math.round((now.getTime() - then) / 1000);
  if (seconds < 60) return "updated just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60)
    return `updated ${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `updated ${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `updated ${days} day${days === 1 ? "" : "s"} ago`;
  return `updated ${new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  })}`;
}
