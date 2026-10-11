import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api, type AdminOverview } from "../api/client";
import * as session from "../auth/session";
import { AdminDashboardPage, windowDays } from "./AdminDashboardPage";

const auth = vi.hoisted(() => ({ isAdmin: true as boolean | null }));
const navigation = vi.hoisted(() => ({ notFound: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => "/admin",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: navigation.notFound,
}));

vi.mock("../auth/AuthProvider", () => ({
  useAuthSession: () => ({
    status: "authenticated",
    isAdmin: auth.isAdmin,
    establish: vi.fn(),
    logout: vi.fn(),
  }),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, adminOverview: vi.fn() },
  };
});

const [, yesterday, today] = windowDays(3);

const overview: AdminOverview = {
  days: 30,
  generated_at: "2026-10-11T12:00:00Z",
  users: {
    total: 12,
    admins: 1,
    new_in_window: 3,
    active_7d: 4,
    active_30d: 7,
    active_in_window: 7,
    new_per_week: [{ week: "2026-10-05", count: 3 }],
    top_users: [
      {
        id: "u-1",
        email: "ada@example.edu",
        username: "ada",
        is_admin: true,
        created_at: "2026-09-01T00:00:00Z",
        jobs: 41,
        lines_approved: 230,
        last_seen: "2026-10-11T09:30:00Z",
      },
    ],
  },
  activity: {
    jobs: [
      { day: yesterday, kind: "segment", status: "done", count: 5 },
      { day: today, kind: "transcribe", status: "failed", count: 2 },
    ],
    pages: [{ day: today, count: 6 }],
    lines: [{ day: today, count: 461 }],
    annotations: [{ day: today, corrections: 3, approved: 18 }],
  },
  models: [
    {
      name: "byzantine-calamari-v3",
      task: "transcribe",
      provider: "calamari",
      jobs: 9,
      done: 6,
      failed: 2,
      success_rate: 0.75,
      median_seconds: 12.4,
    },
  ],
  client_failures: {
    started_at: "2026-10-11T08:00:00Z",
    per_day: [{ day: today, count: 1 }],
    recent: [
      {
        at: "2026-10-11T10:00:00Z",
        path: "/projects/p/documents/d",
        status: 500,
        source: "ui",
        message: "Upload failed",
      },
    ],
  },
  database: {
    name: "nomikos",
    size_bytes: 10_485_760,
    numbackends: 3,
    active_queries: 1,
    xact_commit: 1200,
    xact_rollback: 30,
    blks_hit: 990,
    blks_read: 10,
    tup_returned: 5000,
    tup_fetched: 4000,
    tup_inserted: 300,
    tup_updated: 20,
    tup_deleted: 5,
    stats_reset: null,
    pool_status: "Pool size: 5  Connections in pool: 1",
    busiest_tables: [
      {
        name: "auth_rate_limit_attempts",
        inserts: 812,
        updates: 0,
        deletes: 700,
        seq_scan: 2,
        idx_scan: 900,
        live_rows: 112,
        total_bytes: 81920,
      },
    ],
    seq_scan_tables: [],
    largest_tables: [],
  },
  requests: {
    started_at: "2026-10-11T08:00:00Z",
    total_requests: 57,
    routes: [
      {
        method: "GET",
        route: "/projects/{project_id}/documents",
        status_class: "2xx",
        count: 40,
        avg_ms: 12.5,
        max_ms: 80.1,
      },
    ],
  },
};

describe("AdminDashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    auth.isAdmin = true;
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.mocked(api.adminOverview).mockResolvedValue(overview);
  });

  it("renders every card from one overview response", async () => {
    render(<AdminDashboardPage />);

    expect(await screen.findByText("ada@example.edu")).toBeTruthy();
    expect(api.adminOverview).toHaveBeenCalledTimes(1);
    expect(api.adminOverview).toHaveBeenCalledWith(30);

    const stats = screen.getByLabelText("Headline numbers");
    expect(within(stats).getByText("12")).toBeTruthy();

    // Jobs chart: one legend entry per status, and the day's total above its bar.
    const jobs = screen.getByRole("figure", { name: "Jobs by status" });
    expect(within(jobs).getByText("Done")).toBeTruthy();
    expect(within(jobs).getByText("Failed")).toBeTruthy();
    expect(within(jobs).getByText("5")).toBeTruthy();

    expect(screen.getByText("byzantine-calamari-v3")).toBeTruthy();
    expect(screen.getByText("75.0%")).toBeTruthy();
    expect(screen.getAllByText("auth_rate_limit_attempts").length).toBe(1);
    expect(screen.getByText("/projects/{project_id}/documents")).toBeTruthy();
    expect(screen.getByText("Upload failed")).toBeTruthy();
    expect(navigation.notFound).not.toHaveBeenCalled();
  });

  it("shows the not-found page to a non-admin and never asks for the overview", () => {
    auth.isAdmin = false;

    render(<AdminDashboardPage />);

    expect(navigation.notFound).toHaveBeenCalled();
    expect(api.adminOverview).not.toHaveBeenCalled();
    expect(screen.queryByText("Admin")).toBeNull();
  });
});
