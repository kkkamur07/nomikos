import { QueryClient, type Query } from "@tanstack/react-query";

/** Identity of one cached read. */
export type ResourceKey = readonly (string | number | boolean)[];

/** What a read depends on, and therefore what a write can make stale. */
export type ResourceTag = string;

/**
 * How long a settled read stays servable without going back to the server.
 *
 * The app invalidates its own writes explicitly, so this window only bounds how
 * stale another client's write can leave the page - long enough that paging back
 * and forth costs nothing, short enough that a collaborator's change shows up
 * without a reload.
 */
const RESOURCE_FRESH_MS = 30_000;

/**
 * A module singleton rather than a value created inside a component, because
 * three callers are not components: `setAccessToken`/`clearAccessToken` clear it
 * at a session boundary, `invalidateAfter` drops tags after a write, and the page
 * editor fetches a document imperatively while assembling a page.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: RESOURCE_FRESH_MS,
      // Pages map a failure onto their own banner text immediately; retrying
      // would delay that.
      retry: false,
      // Respects `staleTime`, so returning to a tab within the freshness
      // window still costs nothing.
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
    },
  },
});

/**
 * What a read depends on travels in `meta`, not in the query key.
 *
 * Most reads are composites - a dashboard loads its project and that project's
 * documents in one pass - so a write has to be able to make a read stale without
 * knowing its key. Tags are that indirection: a write names what it dirtied and
 * never enumerates the reads that happen to depend on it.
 */
type ResourceMeta = { tags: readonly ResourceTag[] };

export function taggedMeta(tags: readonly ResourceTag[]): ResourceMeta {
  return { tags };
}

function queryTags(query: Query): readonly ResourceTag[] {
  const meta = query.meta as ResourceMeta | undefined;
  return meta?.tags ?? [];
}

/**
 * Drop every cached read carrying one of `tags` and refetch whoever is showing it.
 *
 * Awaitable: a mutation that just wrote keeps its busy flag up across the
 * reload instead of firing a second explicit refetch at the same key.
 */
export function invalidateResourceTags(
  tags: readonly ResourceTag[],
): Promise<void> {
  if (tags.length === 0) return Promise.resolve();
  const dirty = new Set(tags);
  return queryClient
    .invalidateQueries({
      predicate: (query) => queryTags(query).some((tag) => dirty.has(tag)),
    })
    .then(() => undefined);
}
