import { useState } from "react";
import { toast } from "../components/ui/toast";
import { api } from "../api/client";
import { ApiError } from "../api/errors";
import {
  prepareDirectUpload,
  type DirectUploadPayload,
} from "../utils/encodePartImage";
import { renderPdfToPageFiles } from "../utils/pdfToPages";
import { compareFilenames, isPdfFile } from "../utils/uploadBatch";

/** Uploads run three at a time: serial was slow, unbounded hammers storage. */
const UPLOAD_CONCURRENCY = 3;

type PageUploadResult =
  { ok: true } | { ok: false; name: string; message: string };

async function finalizeWithRetry(
  projectId: string,
  documentId: string,
  partId: string,
  imageKey: string,
  payload: DirectUploadPayload,
): Promise<void> {
  const body = {
    image_key: imageKey,
    width: payload.width ?? null,
    height: payload.height ?? null,
  };
  try {
    await api.finalizePartUpload(projectId, documentId, partId, body);
  } catch (err) {
    // The bytes are already in storage; a transient failure here must not
    // orphan them. A 4xx (409, 422) answers the same way twice, so only a
    // network failure or a 5xx earns the one retry, then the error stands.
    if (err instanceof ApiError && err.status < 500) throw err;
    await api.finalizePartUpload(projectId, documentId, partId, body);
  }
}

async function uploadOnePart(
  projectId: string,
  documentId: string,
  file: File,
): Promise<void> {
  // Prefer the direct-to-storage path: get a presigned Supabase URL, PUT the
  // bytes straight to storage, then finalize. This bypasses Vercel's 4.5 MB
  // function-body cap that a manuscript scan would hit. The payload is never
  // lossy: natively displayable formats upload as the user's original bytes,
  // everything else is transcoded to lossless PNG. If the backend cannot
  // presign (local storage) or the browser cannot decode the file at all,
  // fall back to the legacy multipart upload.
  let payload: DirectUploadPayload | undefined;
  try {
    payload = await prepareDirectUpload(file);
  } catch {
    payload = undefined;
  }

  if (payload) {
    const begin = await api.beginPartUpload(projectId, documentId, {
      filename: payload.filename,
      size: payload.blob.size,
    });
    if (begin.upload_url && begin.part_id) {
      let put: Response;
      try {
        put = await fetch(begin.upload_url, {
          method: "PUT",
          body: payload.blob,
          headers: { "Content-Type": payload.contentType },
        });
      } catch {
        // The fetch itself failed - a CSP connect-src block shows up here as
        // a TypeError, never as a status - so the presigned host is
        // unreachable and the multipart path takes over. `begin` already
        // committed a pending row, which the document would show as a broken
        // page until the media GC sweeps it, so discard it first. Best
        // effort: if the delete fails the sweeper still reaps the row after
        // the presign window, and the user's file matters more.
        try {
          await api.deletePart(projectId, documentId, begin.part_id);
        } catch {
          // The sweeper reaps the pending row; the upload goes on.
        }
        await api.uploadPart(projectId, documentId, file);
        return;
      }
      if (!put.ok) {
        throw new Error("Storage upload failed");
      }
      await finalizeWithRetry(
        projectId,
        documentId,
        begin.part_id,
        begin.image_key,
        payload,
      );
      return;
    }
    // upload_url is null -> the backend is local storage; use the multipart path.
  }

  await api.uploadPart(projectId, documentId, file);
}

async function splitPdfPages(
  ordered: File[],
  onProgress: (label: string) => void,
): Promise<{ pages: File[]; unreadable: string[] }> {
  const pages: File[] = [];
  const unreadable: string[] = [];
  for (const file of ordered) {
    if (!isPdfFile(file)) {
      pages.push(file);
      continue;
    }
    try {
      onProgress(`Splitting ${file.name}…`);
      pages.push(
        ...(await renderPdfToPageFiles(file, (done, total) =>
          onProgress(`Splitting ${file.name} · page ${done}/${total}`),
        )),
      );
    } catch {
      unreadable.push(file.name);
    }
  }
  return { pages, unreadable };
}

async function attemptPageUpload(
  projectId: string,
  documentId: string,
  file: File,
): Promise<PageUploadResult> {
  try {
    await uploadOnePart(projectId, documentId, file);
    return { ok: true };
  } catch (err) {
    return {
      ok: false,
      name: file.name,
      message: err instanceof ApiError ? err.message : "Upload failed",
    };
  }
}

async function uploadPages(
  projectId: string,
  documentId: string,
  pages: File[],
  onStart: (index: number, total: number) => void,
): Promise<PageUploadResult[]> {
  const results: PageUploadResult[] = new Array(pages.length);
  let next = 0;
  const workers = Array.from(
    { length: Math.min(UPLOAD_CONCURRENCY, pages.length) },
    async () => {
      while (next < pages.length) {
        const index = next;
        next += 1;
        onStart(index, pages.length);
        results[index] = await attemptPageUpload(
          projectId,
          documentId,
          pages[index],
        );
      }
    },
  );
  await Promise.all(workers);
  return results;
}

function summarizeResults(results: PageUploadResult[]): {
  uploaded: number;
  failed: string[];
  firstFailure: string | null;
} {
  let uploaded = 0;
  const failed: string[] = [];
  let firstFailure: string | null = null;
  for (const result of results) {
    if (result.ok) {
      uploaded += 1;
    } else {
      failed.push(result.name);
      firstFailure ??= result.message;
    }
  }
  return { uploaded, failed, firstFailure };
}

function reportUploadOutcome(
  summary: { failed: string[]; firstFailure: string | null },
  unreadable: string[],
): void {
  const { failed, firstFailure } = summary;
  for (const name of unreadable) {
    toast.error(`Could not read ${name} as a PDF`);
  }
  if (failed.length === 1) {
    toast.error(firstFailure ?? "Upload failed");
  } else if (failed.length > 1) {
    toast.error(`Upload failed for ${failed.length} pages`);
  }
}

export function usePartUpload(options: {
  projectId: string | undefined;
  documentId: string | undefined;
  onUploaded: () => Promise<void>;
}): {
  uploading: boolean;
  uploadProgress: string | null;
  handleUpload: (files: File[]) => Promise<void>;
} {
  const { projectId, documentId, onUploaded } = options;
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);

  const handleUpload = async (files: File[]) => {
    if (!projectId || !documentId || files.length === 0) return;
    setUploading(true);
    try {
      // Page order comes from filenames; a PDF holds the slot its name sorts
      // into and expands to its pages in document order.
      const ordered = [...files].sort((a, b) =>
        compareFilenames(a.name, b.name),
      );
      const { pages, unreadable } = await splitPdfPages(
        ordered,
        setUploadProgress,
      );

      const results = await uploadPages(
        projectId,
        documentId,
        pages,
        (index, total) =>
          setUploadProgress(
            total > 1 ? `Uploading page ${index + 1}/${total}` : "Uploading…",
          ),
      );
      const summary = summarizeResults(results);
      if (summary.uploaded > 0) {
        toast.success(
          summary.uploaded === 1
            ? "Part uploaded"
            : `${summary.uploaded} pages uploaded`,
        );
        await onUploaded();
      }
      reportUploadOutcome(summary, unreadable);
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  return { uploading, uploadProgress, handleUpload };
}
