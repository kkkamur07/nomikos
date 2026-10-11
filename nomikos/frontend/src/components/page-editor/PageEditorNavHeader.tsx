import Link from "next/link";
import { usePathname } from "next/navigation";

import { FeedbackDialog } from "../layout/FeedbackDialog";
import { useFeedbackButton } from "../layout/useFeedbackButton";

type PageEditorBackLinkProps = {
  to: string;
  label?: string;
};

export function PageEditorBackLink({
  to,
  label = "Back",
}: PageEditorBackLinkProps) {
  return (
    <Link href={to} className="pe-toolbar__back" aria-label="Back to document">
      <span className="pe-toolbar__back-icon" aria-hidden="true">
        ←
      </span>
      <span className="pe-toolbar__back-label">{label}</span>
    </Link>
  );
}

type PageEditorNavHeaderProps = {
  backHref: string;
};

export function PageEditorNavHeader({ backHref }: PageEditorNavHeaderProps) {
  const pathname = usePathname();
  const { dialogOpen, thanks, openDialog, closeDialog, handleSent } =
    useFeedbackButton();

  return (
    <header className="pe-toolbar pe-toolbar--minimal" role="banner">
      <Link
        href="/projects"
        className="pe-toolbar__logo"
        aria-label="nomikos home"
      >
        <img src="/nomos.svg" alt="" />
        <span>nomikos</span>
      </Link>
      <PageEditorBackLink to={backHref} />
      {/* This header is otherwise empty, so a text button costs no space. */}
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        style={{ marginLeft: "auto" }}
        onClick={openDialog}
      >
        {thanks ? "Thanks" : "Feedback"}
      </button>
      <FeedbackDialog
        open={dialogOpen}
        pagePath={pathname ?? "/"}
        onClose={closeDialog}
        onSent={handleSent}
      />
    </header>
  );
}
