import { useState, type FormEvent } from "react";

import { api } from "../../api/client";
import { ApiError } from "../../api/errors";
import { FormModal } from "../ui/FormModal";

/** Shown once the note passes 1800 chars; below that it costs space. */
const COUNTER_FROM = 1800;
const MAX_LENGTH = 2000;

type FeedbackDialogProps = {
  open: boolean;
  pagePath: string;
  onClose: () => void;
  onSent: () => void;
};

export function FeedbackDialog({
  open,
  pagePath,
  onClose,
  onSent,
}: FeedbackDialogProps) {
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const trimmed = message.trim();

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!trimmed || sending) return;
    setSending(true);
    setError(null);
    api
      .sendFeedback({ message: trimmed, page_path: pagePath })
      .then(() => {
        setMessage("");
        onSent();
      })
      .catch((err: unknown) => {
        setError(
          err instanceof ApiError && err.status === 429
            ? "Too many notes sent lately; try again in a bit."
            : "Could not send feedback. Try again.",
        );
      })
      .finally(() => {
        setSending(false);
      });
  };

  return (
    <FormModal
      open={open}
      title="Send feedback"
      onClose={onClose}
      onSubmit={handleSubmit}
      submitLabel="Send"
      loading={sending}
    >
      <div className="field">
        <label htmlFor="feedback-message">Feedback</label>
        <textarea
          id="feedback-message"
          rows={4}
          maxLength={MAX_LENGTH}
          placeholder="What should we improve?"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />
        {message.length > COUNTER_FROM && (
          <p style={{ fontSize: "0.75rem", color: "var(--text-3)" }}>
            {message.length} / {MAX_LENGTH}
          </p>
        )}
      </div>
      {error && (
        <p className="notice-banner" role="alert">
          {error}
        </p>
      )}
      <button
        type="button"
        className="btn btn-ghost btn-block"
        onClick={onClose}
        disabled={sending}
      >
        Cancel
      </button>
    </FormModal>
  );
}
