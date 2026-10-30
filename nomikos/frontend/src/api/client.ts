/**
 * Platform API client - short-lived JWT in memory, OpenAPI-aligned types.
 * Regenerate types: `npm run codegen:api` (after `python scripts/platform/export_openapi.py`).
 *
 * This module is the stable import path: every route lives in the resource
 * module that owns it (`./projects`, `./documents`, `./parts`,
 * `./publicDocuments`, `./jobs`, `./devices`), the transport in `./http`,
 * paths in `./paths` and wire types in `./types`. What stays here is the
 * `api` object and the re-exports importers actually read; anything else is
 * imported from its owning module directly.
 */
import { documentsApi } from "./documents";
import { jobsApi, waitForJob } from "./jobs";
import { partsApi } from "./parts";
import { authApi, projectsApi } from "./projects";
import { publicDocumentsApi } from "./publicDocuments";

export * from "./types";
export {
  API_BASE_URL,
  apiRequest,
  fetchBinaryApi,
  refreshAccessToken,
} from "./http";
export { publicPartMediaUrl } from "./paths";
export { waitForJob };

export const api = {
  ...authApi,
  ...projectsApi,
  ...documentsApi,
  ...partsApi,
  ...publicDocumentsApi,
  ...jobsApi,
};
