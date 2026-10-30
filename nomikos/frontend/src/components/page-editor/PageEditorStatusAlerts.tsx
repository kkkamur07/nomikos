import { useEffect, useRef } from "react";
import { toast } from "../ui/toast";
import type { StatusMessage } from "./statusMessage";

/**
 * One toast per message *raised*, not per distinct sentence.
 *
 * The effect is keyed on the message's token rather than its text, because
 * several of these sentences are constants and two saves in a row would
 * otherwise be one dependency that never changed. See `statusMessage`.
 */
function useSuccessToast(message: StatusMessage | null) {
  const messageRef = useRef(message);
  messageRef.current = message;
  const token = message?.at;
  useEffect(() => {
    if (token !== undefined && messageRef.current) {
      toast.success(messageRef.current.text);
    }
  }, [token]);
}

function useErrorToast(text: string | null) {
  useEffect(() => {
    if (text) toast.error(text);
  }, [text]);
}

type PageEditorStatusAlertsProps = {
  /**
   * "No inference host had capacity", as the platform explained it when it
   * refused the submission. Deliberately not routed through a toast: it names
   * something the researcher can fix, so it stays on screen until the next run.
   */
  submissionRefusal: string | null;
  saveMessage: StatusMessage | null;
  transcriptionSaveMessage: StatusMessage | null;
  ocrMessage: StatusMessage | null;
  segmentMessage: StatusMessage | null;
  mutationError: string | null;
  pairingError: string | null;
  layoutError: string | null;
  lineError: string | null;
};

function StatusItem({
  message,
  variant = "error",
}: {
  message: string;
  variant?: "error" | "warning";
}) {
  return (
    <div className={`pe-status-item pe-status-item--${variant}`}>
      {variant === "error" && <span aria-hidden="true">✕</span>}
      {variant === "warning" && <span aria-hidden="true">!</span>}
      <span>{message}</span>
    </div>
  );
}

type StickyAlert = {
  kind: string;
  text: string;
  variant: "error" | "warning";
};

function unavailable(api: string, error: string | null): string | null {
  return error ? `${api} API unavailable: ${error}` : null;
}

/** The alerts that stay on screen, in the order they are shown. */
function stickyAlerts(props: PageEditorStatusAlertsProps): StickyAlert[] {
  const alerts: [string, string | null, StickyAlert["variant"]][] = [
    ["refusal", props.submissionRefusal, "warning"],
    ["mutation", props.mutationError, "error"],
    ["pairing", props.pairingError, "warning"],
    ["layout", unavailable("Layout", props.layoutError), "warning"],
    ["line", unavailable("Segment", props.lineError), "warning"],
  ];
  return alerts.flatMap(([kind, text, variant]) =>
    text ? [{ kind, text, variant }] : [],
  );
}

/** Success/completion feedback uses auto-dismiss toasts; only errors stay sticky. */
export function PageEditorStatusAlerts(props: PageEditorStatusAlertsProps) {
  useSuccessToast(props.saveMessage);
  useSuccessToast(props.transcriptionSaveMessage);
  useSuccessToast(props.ocrMessage);
  useSuccessToast(props.segmentMessage);
  useErrorToast(props.mutationError);
  useErrorToast(props.pairingError);
  useErrorToast(unavailable("Layout", props.layoutError));
  useErrorToast(unavailable("Segment", props.lineError));

  const alerts = stickyAlerts(props);
  if (alerts.length === 0) return null;

  return (
    <div className="pe-status-alerts">
      {alerts.map((alert) => (
        <StatusItem
          key={alert.kind}
          message={alert.text}
          variant={alert.variant}
        />
      ))}
    </div>
  );
}

export function hasPageEditorStatusAlerts(
  props: PageEditorStatusAlertsProps,
): boolean {
  return stickyAlerts(props).length > 0;
}
