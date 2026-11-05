import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../../api/errors";
import { toast } from "./toast";

type BeaconPayload = {
  message: string;
  ref: string | null;
  status: number | null;
  source: string;
};

describe("toast.error", () => {
  let sendBeacon: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    sendBeacon = vi.fn(() => true);
    vi.stubGlobal("navigator", { sendBeacon });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  async function sentPayload(): Promise<BeaconPayload> {
    expect(sendBeacon).toHaveBeenCalledTimes(1);
    const blob = sendBeacon.mock.calls[0][1] as Blob;
    return JSON.parse(await blob.text()) as BeaconPayload;
  }

  it("reports the cause's status and ref while keeping the shown text", async () => {
    const cause = new ApiError("Internal server error", 500, "ref-123456789");

    toast.error("Server error: boom (ref ref-1234)", cause);

    const payload = await sentPayload();
    expect(payload.status).toBe(500);
    expect(payload.ref).toBe("ref-123456789");
    expect(payload.message).toBe("Server error: boom (ref ref-1234)");
    expect(payload.source).toBe("toast");
  });

  it("still reports a message-only toast, without status or ref", async () => {
    toast.error("Could not copy the link");

    const payload = await sentPayload();
    expect(payload.message).toBe("Could not copy the link");
    expect(payload.status).toBeNull();
    expect(payload.ref).toBeNull();
  });
});
