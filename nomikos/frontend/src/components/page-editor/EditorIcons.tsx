import type { ReactNode } from "react";

type IconButtonProps = {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
};

export function IconButton({
  label,
  onClick,
  disabled,
  children,
}: IconButtonProps) {
  return (
    <button
      type="button"
      className="pe-icon-btn"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function RefreshIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M15.312 11.424a5.5 5.5 0 0 1-9.201 2.466l-.312-.311h2.433a.75.75 0 0 0 0-1.5H4.189a.75.75 0 0 0-.75.75v4.243a.75.75 0 0 0 1.5 0v-2.43l.31.31a7 7 0 0 0 11.712-3.138.75.75 0 0 0-1.449-.39Zm1.23-3.723a.75.75 0 0 0-1.449-.39 5.5 5.5 0 0 0-9.201 2.466l-.31-.31h2.433a.75.75 0 0 0 0-1.5H4.189a.75.75 0 0 0-.75.75v4.243a.75.75 0 0 0 1.5 0v-2.43l.312-.31a7 7 0 0 1 11.712 3.138.75.75 0 0 0 1.449.39Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function DownloadIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 1 0-1.09-1.03l-2.955 3.129V2.75Z" />
      <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
    </svg>
  );
}

export function CloseIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
    </svg>
  );
}

export function SettingsIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M8.34 1.804A1 1 0 0 1 9.32 1h1.36a1 1 0 0 1 .98.804l.295 1.473c.497.144.971.342 1.416.587l1.397-1.098a1 1 0 0 1 1.298.062l.962.962a1 1 0 0 1 .062 1.298l-1.098 1.397c.245.445.443.919.587 1.416l1.473.295a1 1 0 0 1 .804.98v1.361a1 1 0 0 1-.804.98l-1.473.295a6.95 6.95 0 0 1-.587 1.416l1.098 1.397a1 1 0 0 1-.062 1.298l-.962.962a1 1 0 0 1-1.298.062l-1.397-1.098a6.95 6.95 0 0 1-1.416.587l-.295 1.473a1 1 0 0 1-.98.804H9.32a1 1 0 0 1-.98-.804l-.295-1.473a6.95 6.95 0 0 1-1.416-.587l-1.397 1.098a1 1 0 0 1-1.298-.062l-.962-.962a1 1 0 0 1-.062-1.298l1.098-1.397a6.95 6.95 0 0 1-.587-1.416L1.804 10.68a1 1 0 0 1-.804-.98V8.34a1 1 0 0 1 .804-.98l1.473-.295c.144-.497.342-.971.587-1.416L2.766 4.252a1 1 0 0 1 .062-1.298l.962-.962a1 1 0 0 1 1.298-.062l1.397 1.098c.445-.245.919-.443 1.416-.587l.295-1.473ZM10 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function FeedbackIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M4 3h12a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9l-4 3.5v-3.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
    </svg>
  );
}

export function CheckIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/** Lines of text beside a page: the side-by-side transcript pane. */
export function TranscriptIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="1.8" y="2.3" width="12.4" height="11.4" rx="1" />
      <path d="M8 2.3v11.4M10 5.5h2.2M10 8h2.2M10 10.5h1.4" />
    </svg>
  );
}

/** Three dots: the overflow menu for actions that no longer fit the bar. */
export function MoreIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <circle cx="4.5" cy="10" r="1.5" />
      <circle cx="10" cy="10" r="1.5" />
      <circle cx="15.5" cy="10" r="1.5" />
    </svg>
  );
}

/** Stacked line boxes: segment the page. */
export function SegmentIcon() {
  return (
    <svg
      className="pe-tb-btn__icon"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="2.2" y="2.6" width="11.6" height="3" rx="0.8" />
      <rect x="2.2" y="6.9" width="11.6" height="3" rx="0.8" />
      <path d="M2.2 12.6h7.4" />
    </svg>
  );
}

/** A type T: transcribe. */
export function TranscribeIcon() {
  return (
    <svg
      className="pe-tb-btn__icon"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M2.6 4.2h10.8M8 4.2v8.2M5.6 12.4h4.8" />
    </svg>
  );
}

/** Three linked steps: the Workflow menu. */
export function WorkflowIcon() {
  return (
    <svg
      className="pe-tb-btn__icon"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="1.8" y="2.2" width="4.4" height="4.4" rx="0.8" />
      <rect x="9.8" y="9.4" width="4.4" height="4.4" rx="0.8" />
      <path d="M6.2 4.4h2.6a1.2 1.2 0 0 1 1.2 1.2v3.8" />
    </svg>
  );
}

/** A magnifier over a line: look for segmentation mistakes. */
export function CheckPageIcon() {
  return (
    <svg
      className="pe-tb-btn__icon"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <circle cx="7" cy="7" r="4.4" />
      <path d="M10.2 10.2l3.6 3.6M5 7h4" />
    </svg>
  );
}
