export type CursorPage<T> = {
  items: T[];
  next_cursor: string | null;
};

export type CursorPageOptions = {
  cursor?: string | null;
  limit?: number;
  signal?: AbortSignal;
};

type CollectCursorPagesOptions = {
  maxPages?: number;
  signal?: AbortSignal;
};

const DEFAULT_MAX_PAGES = 10;

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("The request was cancelled.", "AbortError");
  }
}

/**
 * Bounded compatibility helper for callers that still need a complete list.
 * New UIs should render a page and opt into subsequent cursors explicitly.
 */
export async function collectCursorPages<T>(
  fetchPage: (options: CursorPageOptions) => Promise<CursorPage<T>>,
  options: CollectCursorPagesOptions = {},
): Promise<T[]> {
  const maxPages = options.maxPages ?? DEFAULT_MAX_PAGES;
  if (!Number.isInteger(maxPages) || maxPages < 1) {
    throw new Error("maxPages must be a positive integer.");
  }

  const items: T[] = [];
  const seenCursors = new Set<string>();
  let cursor: string | null = null;

  for (let pageNumber = 0; pageNumber < maxPages; pageNumber += 1) {
    throwIfAborted(options.signal);
    const page = await fetchPage({ cursor, signal: options.signal });
    throwIfAborted(options.signal);
    items.push(...page.items);

    const nextCursor = page.next_cursor;
    if (!nextCursor) return items;
    if (seenCursors.has(nextCursor)) {
      throw new Error("Cursor pagination repeated a cursor.");
    }
    seenCursors.add(nextCursor);
    cursor = nextCursor;
  }

  throw new Error(`Cursor pagination exceeded the ${maxPages}-page limit.`);
}
