/**
 * The account-level **host preference**: "use my computer when it is
 * available".
 *
 * It is the only researcher input to **execution target** selection, and it is
 * chosen once for the account rather than per job - a researcher cannot know
 * which host is faster for a given page, so asking at every action is a
 * decision without a basis, and it is exactly what regrows the three-mode
 * complexity ADR 0002 deletes.
 *
 * The setting lives on the server, not in `localStorage`. The platform is what
 * decides an execution target at submission, so a preference the platform
 * cannot read would be a preference in name only.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, type ExecutionPreferenceResponse } from "../api/client";
import { userFacingMessage } from "../api/userFacingError";

export const HOST_PREFERENCE_LABEL = "Use my computer when it is available";

export const HOST_PREFERENCE_HINT =
  "Jobs run on this computer while the nomikos agent is running, and in the cloud otherwise. Each job says which one it used.";

export type HostPreference = {
  /** `null` until the account setting has been read. */
  preference: ExecutionPreferenceResponse | null;
  preferLocalInference: boolean;
  /**
   * Whether this account's own computer can take work right now.
   *
   * Read from the platform's **capacity** answer, not from this browser: the
   * platform is what an agent reports to and what decides an **execution
   * target** at submission (ADR 0002). A browser probe against `127.0.0.1`
   * could disagree with the decision the platform is about to make.
   */
  hasLocalCapacity: boolean;
  loading: boolean;
  saving: boolean;
  error: string | null;
  setPreferLocalInference: (preferLocal: boolean) => Promise<void>;
  /** Re-read the account setting and capacity with it. */
  refresh: () => Promise<void>;
};

function readSettled(
  signal: AbortSignal | undefined,
  mounted: boolean,
): boolean {
  return signal?.aborted === true || !mounted;
}

export function useHostPreference(): HostPreference {
  const [preference, setPreference] =
    useState<ExecutionPreferenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const read = useCallback(async (signal?: AbortSignal) => {
    try {
      const value = await api.getExecutionPreference({ signal });
      if (readSettled(signal, mountedRef.current)) return;
      setPreference(value);
      setError(null);
    } catch (failure) {
      if (readSettled(signal, mountedRef.current)) return;
      setError(userFacingMessage(failure, "Could not read your host setting."));
    } finally {
      if (!readSettled(signal, mountedRef.current)) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void read(controller.signal);
    return () => controller.abort();
  }, [read]);

  useEffect(() => {
    // Re-read when the researcher comes back to the tab: they may have just
    // started the agent in a terminal, and **capacity** is the only thing that
    // will say so. Everything between visits is covered by the explicit Retry
    // control rather than by a background poll.
    function onFocus() {
      void read();
    }
    function onVisibility() {
      if (document.visibilityState === "visible") void read();
    }
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [read]);

  /**
   * The response is the new state. Writing what the server returned rather than
   * what was sent is what makes the round trip visible: `available_targets` and
   * `preferred_execution_target` are recomputed from live **capacity**, and a
   * client that echoed its own input would show a preference the platform has
   * not agreed to.
   */
  const setPreferLocalInference = useCallback(async (preferLocal: boolean) => {
    setSaving(true);
    try {
      const saved = await api.setExecutionPreference(preferLocal);
      if (!mountedRef.current) return;
      setPreference(saved);
      setError(null);
    } catch (failure) {
      if (!mountedRef.current) return;
      setError(userFacingMessage(failure, "Could not save your host setting."));
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  }, []);

  return {
    preference,
    // Until the account is read, assume the cloud: it is the host that needs
    // nothing installed, so a wrong guess costs a re-render rather than a
    // failure.
    preferLocalInference: preference?.prefer_local_inference ?? false,
    hasLocalCapacity: preference?.available_targets.includes("local") ?? false,
    loading,
    saving,
    error,
    setPreferLocalInference,
    // Retry says "Checking…" while it runs; the focus and visibility re-reads
    // deliberately do not, because a quiet background refresh that flickers the
    // banner every time the tab regains focus is worse than one that does not.
    refresh: async () => {
      setLoading(true);
      await read();
    },
  };
}
