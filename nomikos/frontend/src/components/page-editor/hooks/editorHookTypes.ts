import type { Dispatch, SetStateAction } from "react";
import type {
  EnqueueJobResponse,
  JobResponse,
  PageTranscriptionTextLineResponse,
  PagePairingResponse,
  PairingProgressResponse,
} from "../../../api/client";
import { submissionRefusalExplanation } from "../../../inference";
import { ApiError, isAbortError } from "../../../api/errors";
import type { PageEditorJobKind } from "../jobProgress";

export type SetState<T> = Dispatch<SetStateAction<T>>;

export type TextLinesSetter = SetState<PageTranscriptionTextLineResponse[]>;

export type PairingProgressSetter = SetState<PairingProgressResponse>;

export type TrackJobAndWait = (
  enqueued: EnqueueJobResponse,
  meta: { label: string; kind: PageEditorJobKind },
  options?: { timeoutMs?: number },
) => Promise<JobResponse>;

/** The text an error banner shows: the error's own message, or `fallback`. */
export function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

/** The banner for a failed layout write: a 403 says who may edit layout. */
export function layoutMutationMessage(
  error: unknown,
  fallback = "Layout update failed.",
): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Only project members can edit layout.";
  }
  return errorMessage(error, fallback);
}

/**
 * A job cancelled from the jobs panel resolves rather than rejects, so it is
 * turned into the same AbortError a cancelled request raises: the panel has
 * already reported it, and the runner must neither report success nor fail.
 */
export function throwIfCancelled(job: JobResponse): JobResponse {
  if (job.status === "cancelled") {
    throw new DOMException("The job was cancelled.", "AbortError");
  }
  return job;
}

/**
 * The catch tail every job runner shares: a cancellation stays quiet, a
 * refused submission gets its standing explanation, anything else is an error.
 */
export function reportJobFailure(
  error: unknown,
  fallback: string,
  report: {
    setSubmissionRefusal: SetState<string | null>;
    setError: SetState<string | null>;
  },
): void {
  if (isAbortError(error)) return;
  const refusal = submissionRefusalExplanation(error);
  if (refusal) {
    report.setSubmissionRefusal(refusal);
    return;
  }
  report.setError(errorMessage(error, fallback));
}

/** Puts a pairing read on screen: the text lines and the progress count. */
export function showPairing(
  pairing: Pick<PagePairingResponse, "text_lines" | "pairing_progress">,
  setTextLines: TextLinesSetter,
  setPairingProgress: PairingProgressSetter,
): void {
  setTextLines(pairing.text_lines);
  setPairingProgress(pairing.pairing_progress);
}
