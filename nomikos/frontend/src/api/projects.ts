/**
 * Project and account endpoints, behind the shared transport.
 *
 * Each module in this directory owns one resource's routes and imports only
 * `http`, `paths`, `types` and `cursorPagination` - never the `client`
 * barrel - so the dependency runs one way and `utils/jobSubscription` can
 * share the transport without a cycle.
 */
import { collectCursorPages } from "./cursorPagination";
import {
  apiRequest,
  postCsrfProtected,
  refreshAccessToken,
  rememberCsrfToken,
} from "./http";
import { cursorQuery } from "./paths";
import type {
  EnqueueTestJobRequest,
  EnqueueTestJobResponse,
  ExecutionPreferenceResponse,
  ListPageOptions,
  LoginRequest,
  PageResponse,
  ProjectCollaboratorResponse,
  ProjectCreateRequest,
  ProjectResponse,
  ProjectUpdateRequest,
  RegisterRequest,
  ShareUserRequest,
  TokenResponse,
  UserResponse,
} from "./types";

export const authApi = {
  login: (body: LoginRequest) =>
    apiRequest<TokenResponse>("/auth/login", {
      method: "POST",
      body,
      skipAuth: true,
    }).then(rememberCsrfToken),

  register: (body: RegisterRequest) =>
    apiRequest<TokenResponse>("/auth/register", {
      method: "POST",
      body,
      skipAuth: true,
    }).then(rememberCsrfToken),

  refresh: refreshAccessToken,

  // Revoking the session server-side is CSRF-checked too, so it gets the same
  // cookie retry: a tab whose token another tab rotated must still be able to
  // sign out for real, not just forget its own copy of the credentials.
  logout: () => postCsrfProtected<void>("/auth/logout"),

  me: () => apiRequest<UserResponse>("/me"),
};

export const projectsApi = {
  /**
   * The account-level **host preference** - "use my computer when it is
   * available" - plus what it resolves to right now.
   *
   * There is deliberately no per-job variant of either call: a researcher
   * cannot know which host is faster for a given page, so the choice is made
   * once for the account and announced on each job (ADR 0002).
   */
  getExecutionPreference: (options: { signal?: AbortSignal } = {}) =>
    apiRequest<ExecutionPreferenceResponse>("/account/execution-target", {
      signal: options.signal,
    }),

  setExecutionPreference: (preferLocalInference: boolean) =>
    apiRequest<ExecutionPreferenceResponse>("/account/execution-target", {
      method: "PUT",
      body: { prefer_local_inference: preferLocalInference },
    }),

  listProjectsPage: (options: ListPageOptions = {}) => {
    const query = cursorQuery(options);
    return apiRequest<PageResponse<ProjectResponse>>(
      query ? `/projects?${query}` : "/projects",
      {
        signal: options.signal,
      },
    );
  },

  listProjects: (options?: { maxPages?: number; signal?: AbortSignal }) =>
    collectCursorPages(
      (pageOptions) => projectsApi.listProjectsPage(pageOptions),
      options,
    ),

  createProject: (body: ProjectCreateRequest) =>
    apiRequest<ProjectResponse>("/projects", { method: "POST", body }),

  getProject: (projectId: string) =>
    apiRequest<ProjectResponse>(`/projects/${encodeURIComponent(projectId)}`),

  updateProject: (projectId: string, body: ProjectUpdateRequest) =>
    apiRequest<ProjectResponse>(`/projects/${encodeURIComponent(projectId)}`, {
      method: "PATCH",
      body,
    }),

  deleteProject: (projectId: string) =>
    apiRequest<void>(`/projects/${encodeURIComponent(projectId)}`, {
      method: "DELETE",
    }),
  /** Owner-only: the list carries collaborators' emails. */
  listProjectCollaborators: (projectId: string) =>
    apiRequest<ProjectCollaboratorResponse[]>(
      `/projects/${encodeURIComponent(projectId)}/share`,
    ),

  shareProject: (projectId: string, body: ShareUserRequest) =>
    apiRequest<void>(`/projects/${encodeURIComponent(projectId)}/share`, {
      method: "POST",
      body,
    }),

  /** By user id: a username may contain a `/`, which no encoding survives in a path segment. */
  unshareProject: (projectId: string, userId: string) =>
    apiRequest<void>(
      `/projects/${encodeURIComponent(projectId)}/share/${encodeURIComponent(userId)}`,
      { method: "DELETE" },
    ),

  enqueueTestJob: (body: EnqueueTestJobRequest = { handler: "noop" }) =>
    apiRequest<EnqueueTestJobResponse>("/jobs/test", {
      method: "POST",
      body,
    }),
};
