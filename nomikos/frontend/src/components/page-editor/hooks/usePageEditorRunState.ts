import { useMemo } from "react";
import { useBackgroundJobs } from "../../../context/BackgroundJobsContext";

export type PageEditorProcessingKind =
  "segmentation" | "transcription-page" | "transcription-segment";

const PROCESSING_LABELS: Record<PageEditorProcessingKind, string> = {
  segmentation: "Segmentation in progress",
  "transcription-page": "Full-page transcription in progress",
  "transcription-segment": "Segment transcription in progress",
};

type PageEditorRunStateInput = {
  segmenting: boolean;
  ocrRunning: boolean;
  ocrScope: "segment" | "page" | null;
};

type PageEditorRunState = {
  /** Which operation the page is busy with, or `null` when it is idle. */
  processingKind: PageEditorProcessingKind | null;
  /** Sentence for the canvas hint, or `null` when idle. */
  processingLabel: string | null;
  /** Whether the app-wide queue is working on anything at all. */
  backgroundJobsActive: boolean;
};

/**
 * The single answer to "what is this page running right now".
 *
 * `segmenting` / `ocrRunning` stay owned by the hooks that drive the operations:
 * they cover the whole call, including the request that enqueues the job and the
 * reloads that follow it, neither of which the job queue can see. Everything the
 * editor merely *derives* from them is derived here, once, instead of being
 * re-expressed at each use site.
 */
export function usePageEditorRunState({
  segmenting,
  ocrRunning,
  ocrScope,
}: PageEditorRunStateInput): PageEditorRunState {
  const { activeCount } = useBackgroundJobs();

  return useMemo(() => {
    const processingKind: PageEditorProcessingKind | null = segmenting
      ? "segmentation"
      : ocrRunning
        ? ocrScope === "page"
          ? "transcription-page"
          : "transcription-segment"
        : null;
    return {
      processingKind,
      processingLabel: processingKind && PROCESSING_LABELS[processingKind],
      backgroundJobsActive: activeCount > 0,
    };
  }, [activeCount, ocrRunning, ocrScope, segmenting]);
}
