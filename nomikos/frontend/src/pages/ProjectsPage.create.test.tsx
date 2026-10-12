import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { ApiError } from "../api/errors";
import * as session from "../auth/session";
import { ProjectsPage } from "./ProjectsPage";

const success = vi.fn();
const error = vi.fn();

vi.mock("../components/ui/toast", () => ({
  toast: {
    success: (...args: unknown[]) => success(...args),
    error: (...args: unknown[]) => error(...args),
  },
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      me: vi.fn(),
      listProjects: vi.fn(),
      createProject: vi.fn(),
      listInferenceModels: vi.fn(),
      listProjectModelBindings: vi.fn(),
      createProjectModelBinding: vi.fn(),
      updateProjectModelBinding: vi.fn(),
    },
  };
});

const CATALOG = [
  { id: "seg-kraken", task: "segment", name: "kraken" },
  { id: "seg-ppocr", task: "segment", name: "ppocr" },
  { id: "htr-greek", task: "transcribe", name: "greek-calamari-v1" },
  { id: "htr-syriac", task: "transcribe", name: "syriac-ppocr-v1" },
];

function openCreateModal() {
  render(<ProjectsPage />);
  fireEvent.click(screen.getByRole("button", { name: "New project" }));
}

function nameIt(name: string) {
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: name } });
}

function submit() {
  fireEvent.click(screen.getByRole("button", { name: "Create" }));
}

function segmentSelect(): HTMLSelectElement {
  return screen.getByLabelText("Segmentation model") as HTMLSelectElement;
}

function transcribeSelect(): HTMLSelectElement {
  return screen.getByLabelText("Transcription model") as HTMLSelectElement;
}

describe("ProjectsPage create dialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(session, "hasAccessToken").mockReturnValue(true);
    vi.spyOn(session, "navigateToLogin").mockImplementation(() => {});
    vi.mocked(api.me).mockResolvedValue({
      id: "user-1",
      email: "dev@example.com",
      username: "dev",
      created_at: "2026-01-01T00:00:00Z",
    });
    vi.mocked(api.listProjects).mockResolvedValue([]);
    vi.mocked(api.createProject).mockResolvedValue({
      id: "project-new",
      name: "Alqosh",
      slug: "alqosh",
      guidelines: null,
      owner_id: "user-1",
      document_count: 0,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    });
    // A project this young has no bindings yet; saveProjectDefault lists first.
    vi.mocked(api.listProjectModelBindings).mockResolvedValue([]);
    vi.mocked(api.listInferenceModels).mockResolvedValue(
      CATALOG as unknown as Awaited<ReturnType<typeof api.listInferenceModels>>,
    );
  });

  it("writes no binding when both rows stay on No default", async () => {
    openCreateModal();
    await waitFor(() => expect(segmentSelect().options.length).toBe(3));
    expect(segmentSelect()).toHaveValue("");
    expect(transcribeSelect()).toHaveValue("");

    nameIt("Alqosh");
    submit();

    await waitFor(() => expect(api.createProject).toHaveBeenCalled());
    expect(api.createProjectModelBinding).not.toHaveBeenCalled();
    expect(api.listProjectModelBindings).not.toHaveBeenCalled();
    expect(success).toHaveBeenCalledWith("Project created");
  });

  it("writes both bindings for the new project id", async () => {
    vi.mocked(api.createProjectModelBinding).mockImplementation(
      async (projectId, body) =>
        ({
          id: `binding-${body.task}`,
          ...body,
        }) as unknown as Awaited<
          ReturnType<typeof api.createProjectModelBinding>
        >,
    );
    openCreateModal();
    await waitFor(() => expect(segmentSelect().options.length).toBe(3));

    nameIt("Alqosh");
    fireEvent.change(segmentSelect(), { target: { value: "seg-ppocr" } });
    fireEvent.change(transcribeSelect(), { target: { value: "htr-syriac" } });
    submit();

    await waitFor(() =>
      expect(api.createProjectModelBinding).toHaveBeenCalledTimes(2),
    );
    expect(api.createProjectModelBinding).toHaveBeenCalledWith("project-new", {
      task: "segment",
      model_id: "seg-ppocr",
    });
    expect(api.createProjectModelBinding).toHaveBeenCalledWith("project-new", {
      task: "transcribe",
      model_id: "htr-syriac",
    });
    expect(success).toHaveBeenCalledWith("Project created");
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New project" })).toBeNull(),
    );
  });

  it("keeps the project and warns once when a binding write fails", async () => {
    vi.mocked(api.createProjectModelBinding).mockRejectedValue(
      new ApiError("No access", 403),
    );
    openCreateModal();
    await waitFor(() => expect(segmentSelect().options.length).toBe(3));

    nameIt("Alqosh");
    fireEvent.change(segmentSelect(), { target: { value: "seg-ppocr" } });
    submit();

    await waitFor(() =>
      expect(error).toHaveBeenCalledWith(
        "Project created, but its default models could not be saved. You can set them on the project page.",
      ),
    );
    expect(api.createProject).toHaveBeenCalledTimes(1);
    expect(success).not.toHaveBeenCalled();
    // The project exists, so the dialog closes and the list is read again.
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New project" })).toBeNull(),
    );
    expect(vi.mocked(api.listProjects).mock.calls.length).toBeGreaterThan(1);
  });

  it("still creates by name alone when the catalog read fails", async () => {
    vi.mocked(api.listInferenceModels).mockRejectedValue(
      new ApiError("offline", 503),
    );
    openCreateModal();

    await waitFor(() => expect(segmentSelect()).toBeDisabled());
    expect(transcribeSelect()).toBeDisabled();

    nameIt("Alqosh");
    submit();

    await waitFor(() =>
      expect(api.createProject).toHaveBeenCalledWith({
        name: "Alqosh",
        slug: "alqosh",
      }),
    );
    expect(api.createProjectModelBinding).not.toHaveBeenCalled();
    expect(success).toHaveBeenCalledWith("Project created");
  });
});
