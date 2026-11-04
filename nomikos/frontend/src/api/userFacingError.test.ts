import { describe, expect, it } from "vitest";

import { ApiError } from "./errors";
import { errorRef, userFacingMessage } from "./userFacingError";

describe("userFacingMessage", () => {
  it("prefers allowlisted API messages", () => {
    expect(
      userFacingMessage(new ApiError("Access denied", 403), "fallback"),
    ).toBe("Access denied");
  });

  it("hides stack-like noise behind a status fallback", () => {
    expect(
      userFacingMessage(
        new ApiError(
          'Traceback (most recent call last):\nFile "app.py", line 1',
          500,
        ),
      ),
    ).toMatch(/something went wrong on the server/i);
  });

  it("exposes correlation refs from ApiError", () => {
    const err = new ApiError("Request failed", 500, "abc123");
    expect(errorRef(err)).toBe("abc123");
  });

  it("shows the server summary and short ref on a 500", () => {
    const err = new ApiError(
      "Internal server error",
      500,
      "0123456789abcdef0123456789abcdef",
      {
        code: "INTERNAL_SERVER_ERROR",
        summary:
          'UndefinedColumnError: column "model_name" of relation "transcriptions" does not exist',
      },
    );
    expect(userFacingMessage(err, "Failed to create document")).toBe(
      'Server error: UndefinedColumnError: column "model_name" of relation "transcriptions" does not exist (ref 01234567)',
    );
  });

  it("keeps the 500 copy and appends the ref when there is no summary", () => {
    const err = new ApiError("Internal server error", 500, "abcdef0123456789");
    expect(userFacingMessage(err, "Failed to create document")).toBe(
      "Internal server error (ref abcdef01)",
    );
  });

  it("explains a fetch TypeError as a connection problem", () => {
    expect(
      userFacingMessage(new TypeError("Failed to fetch"), "Failed to load"),
    ).toBe("Could not reach the server. Check your connection and try again.");
  });

  it("leaves 4xx messages unchanged, without a ref", () => {
    const err = new ApiError("Project slug already taken", 409, "abc12345xyz");
    expect(userFacingMessage(err, "Failed to create project")).toBe(
      "Project slug already taken",
    );
  });
});
