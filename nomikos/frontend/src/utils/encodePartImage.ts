/**
 * Client-side payload preparation for direct-to-storage part uploads.
 *
 * The production API is serverless, and Vercel Functions cap a request body at
 * 4.5 MB - far below a manuscript page scan. Instead of POSTing the bytes through
 * the API (which would be rejected), the browser PUTs them straight to object
 * storage using a presigned URL the API minted. The API still validates and
 * re-reads dimensions on finalize, so this is an optimization that moves bytes
 * around, not a place where validation is skipped.
 *
 * Nothing on this path may lose information: the stored image is the ground truth
 * researchers annotate against. A format the browser can display natively is
 * uploaded byte-for-byte as the user provided it; anything else is decoded and
 * re-encoded as PNG, which is lossless, before upload. Lossy re-encoding is never
 * performed here - the earlier WebP-at-q0.95 step silently degraded every scan.
 *
 * The server's own hard bound is 200 MP (see ``MAX_DECODE_PIXELS``); we keep the
 * client bound tighter so a page the user can still upload is one the server will
 * accept, and so an enormous scan fails fast before a canvas allocation.
 */

const MAX_PART_DECODE_PIXELS = 200_000_000;

/** Formats every browser displays natively; their original bytes upload as-is. */
const NATIVE_UPLOAD_EXTENSIONS: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

type EncodedPartImage = {
  /** Lossless PNG bytes. */
  data: Blob;
  /** Source width in pixels. */
  width: number;
  /** Source height in pixels. */
  height: number;
};

/** What the direct-upload flow PUTs to the presigned URL. */
export type DirectUploadPayload = {
  /** The exact bytes to store. */
  blob: Blob;
  contentType: string;
  /** A filename whose extension matches the bytes; the object key inherits it. */
  filename: string;
  /** Dimension hints, known only when the browser decoded the image itself. */
  width?: number;
  height?: number;
};

class UnsupportedImageError extends Error {
  constructor(message = "Uploaded file is not a valid image") {
    super(message);
    this.name = "UnsupportedImageError";
  }
}

/** Swap (or append) the filename's extension so it matches the uploaded bytes. */
function withExtension(filename: string, extension: string): string {
  const stem = filename.replace(/\.[^./\\]+$/, "");
  return `${stem || filename}.${extension}`;
}

/**
 * Prepare the bytes a direct upload should PUT, without ever losing information.
 *
 * A natively displayable format (JPEG, PNG, WebP) is passed through untouched -
 * the storage object is the user's file, bit for bit. Everything else must be
 * transcoded to display in the page canvas at all, and that transcode targets
 * PNG because it is lossless everywhere ``canvas.toBlob`` exists. Rejects with
 * ``UnsupportedImageError`` when the browser cannot decode the file; the caller
 * then falls back to the multipart upload and the server's own conversion.
 */
export async function prepareDirectUpload(
  file: File,
): Promise<DirectUploadPayload> {
  const nativeExtension = NATIVE_UPLOAD_EXTENSIONS[file.type];
  if (nativeExtension) {
    return {
      blob: file,
      contentType: file.type,
      filename: withExtension(file.name, nativeExtension),
    };
  }
  const encoded = await encodePartImage(file);
  return {
    blob: encoded.data,
    contentType: "image/png",
    filename: withExtension(file.name, "png"),
    width: encoded.width,
    height: encoded.height,
  };
}

/**
 * Decode a File to a bitmap and re-encode it as a lossless PNG with dimensions.
 *
 * ``createImageBitmap`` decodes in the background; browsers that lack it (older
 * Safari) reject here, and the caller falls back to the multipart upload. The
 * re-encode draws the bitmap to a canvas at its natural resolution - no resize -
 * then asks for ``image/png`` output, the one ``canvas.toBlob`` format that is
 * lossless in every browser.
 */
async function encodePartImage(file: File): Promise<EncodedPartImage> {
  const bitmap = await createImageBitmap(file).catch(() => {
    throw new UnsupportedImageError();
  });

  const width = bitmap.width;
  const height = bitmap.height;
  if (width * height > MAX_PART_DECODE_PIXELS) {
    bitmap.close();
    throw new UnsupportedImageError("Uploaded image is too large");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    throw new UnsupportedImageError();
  }

  // A transparent or palette-with-alpha source must stay RGBA; everything else
  // is flattened to opaque RGB so a scanned page has no alpha channel in storage.
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((result) => {
      if (result) resolve(result);
      else reject(new UnsupportedImageError("Could not encode image to PNG"));
    }, "image/png");
  });

  return { data: blob, width, height };
}
