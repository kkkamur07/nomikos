import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuthSession } from "../../auth/AuthProvider";
import { FeedbackDialog } from "./FeedbackDialog";
import { useFeedbackButton } from "./useFeedbackButton";
export type BreadcrumbItem = {
  label: string;
  href?: string;
};

type TopNavProps = {
  breadcrumb?: BreadcrumbItem[];
  currentLabel?: string;
  username?: string | null;
  actions?: ReactNode;
  onLogout?: () => void;
};

export function TopNav({
  breadcrumb,
  currentLabel,
  username,
  actions,
  onLogout,
}: TopNavProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { logout, isAdmin } = useAuthSession();
  const { dialogOpen, thanks, openDialog, closeDialog, handleSent } =
    useFeedbackButton();
  const handleLogout = () => {
    if (onLogout) {
      onLogout();
      return;
    }
    void logout().finally(() => {
      router.push("/login");
    });
  };

  return (
    <nav className="topnav" aria-label="Main navigation">
      <Link href="/projects" className="topnav-logo" aria-label="nomikos home">
        <img src="/nomos.svg" alt="" />
        <span>nomikos</span>
      </Link>
      <div className="topnav-sep" aria-hidden="true" />
      {breadcrumb && breadcrumb.length > 0 ? (
        <nav className="topnav-breadcrumb" aria-label="Breadcrumb">
          {breadcrumb.map((item, index) => (
            <span
              key={`${item.label}-${index}`}
              style={{ display: "contents" }}
            >
              {index > 0 && (
                <span className="sep" aria-hidden="true">
                  /
                </span>
              )}
              {item.href ? (
                <Link href={item.href}>{item.label}</Link>
              ) : (
                <span className="current" aria-current="page">
                  {item.label}
                </span>
              )}
            </span>
          ))}
        </nav>
      ) : currentLabel ? (
        <div className="topnav-breadcrumb">
          <span className="current" aria-current="page">
            {currentLabel}
          </span>
        </div>
      ) : null}
      <div className="topnav-spacer" />
      <div className="topnav-actions">
        {actions}
        {username && <span className="topnav-user">{username}</span>}
        {username && (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={openDialog}
          >
            {thanks ? "Thanks" : "Feedback"}
          </button>
        )}
        {isAdmin === true && (
          <Link href="/admin" className="btn btn-ghost btn-sm">
            Admin
          </Link>
        )}
        <Link href="/settings" className="btn btn-ghost btn-sm">
          Settings
        </Link>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={handleLogout}
        >
          Log out
        </button>
      </div>
      {username && (
        <FeedbackDialog
          open={dialogOpen}
          pagePath={pathname ?? "/"}
          onClose={closeDialog}
          onSent={handleSent}
        />
      )}
    </nav>
  );
}
