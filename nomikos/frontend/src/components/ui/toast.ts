import type { MessageInstance } from "antd/es/message/interface";
import { reportClientFailure } from "../../api/failureBeacon";

export type ToastVariant = "success" | "info" | "error";

/** Matches the dismiss delay the hand-rolled stack used, in antd's seconds. */
const DISMISS_SECONDS = 2.8;

/**
 * antd's `message` only picks up the app's theme when it comes from `App`, and
 * `App.useApp()` is a hook. Call sites here are plain functions, several of them
 * outside React entirely, so the themed instance is registered once at mount by
 * `ToastBridge` and read back through this module.
 */
let api: MessageInstance | null = null;

export function registerToastApi(instance: MessageInstance | null): void {
  api = instance;
}

export const toast = {
  success: (message: string): void => {
    void api?.success(message, DISMISS_SECONDS);
  },
  /**
   * Neither good news nor a failure: something changed under the person and
   * they should know. Deliberately not routed to the failure beacon.
   */
  info: (message: string): void => {
    void api?.info(message, DISMISS_SECONDS);
  },
  /**
   * `cause` is the caught error behind the message. Passing it lets the beacon
   * carry the real status and ref, so the admin table links the toast to the
   * server log line; the person still sees `message`.
   */
  error: (message: string, cause?: unknown): void => {
    // Centralized here, not at call sites, so every surfaced error is
    // reported.
    reportClientFailure(cause ?? new Error(message), "toast", message);
    void api?.error(message, DISMISS_SECONDS);
  },
};
