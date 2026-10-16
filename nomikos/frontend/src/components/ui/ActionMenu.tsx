import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

type FocusTarget = "first" | "last";

export type ActionMenuProps = {
  /** Trigger text. The caret is appended here so every trigger wears the same one. */
  label: string;
  /** Names the popup for a screen reader. Defaults to {@link label}. */
  menuLabel?: string;
  /**
   * The trigger's accessible name, for the one trigger whose visible label is
   * a glyph. "⋯" is not a name.
   */
  triggerAriaLabel?: string;
  disabled?: boolean;
  /** Extra classes on the trigger, for the one menu that leads a row. */
  triggerClassName?: string;
  /** Widen the popup for menus whose items carry sub-lines. */
  wide?: boolean;
  /**
   * Fired on every open and close. A menu that holds a confirm step resets it
   * from here, so that closing the menu mid-confirm and opening it again does
   * not present the destructive button as the first thing under the cursor.
   */
  onOpenChange?: (open: boolean) => void;
  /**
   * The items. Given `close`, because an item decides for itself whether
   * activating it finishes the interaction (a download) or replaces the menu's
   * contents with a confirm step (a destructive job).
   */
  children: (close: () => void) => ReactNode;
};

/**
 * The focusable rows of an open menu, in the order the arrow keys walk them.
 *
 * Read from the DOM rather than from a registry the items opt into. The
 * contents of these menus change while they are open (a confirm step replaces
 * the items behind it, a count arrives and enables a row), and a registry
 * would have to be kept in step with every one of those transitions. Reading
 * the DOM cannot fall out of step with itself.
 */
function menuItems(container: HTMLElement): HTMLElement[] {
  const selector =
    '[role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"]';
  return Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
    (item) =>
      !item.hasAttribute("disabled") &&
      item.getAttribute("aria-disabled") !== "true",
  );
}

function menuItemsIn(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return menuItems(container);
}

function nextFocusIndex(
  items: HTMLElement[],
  current: number,
  delta: 1 | -1,
): number {
  if (current === -1) return delta > 0 ? 0 : items.length - 1;
  return (current + delta + items.length) % items.length;
}

function jumpMenuFocus(items: HTMLElement[], key: string): void {
  if (key === "Home") {
    items[0]?.focus();
  } else if (key === "End") {
    items[items.length - 1]?.focus();
  }
}

/**
 * A keyboard-operable dropdown menu.
 *
 * Enter, Space, ArrowDown and ArrowUp all open it from the trigger; the arrows
 * walk the items and wrap; Home and End jump; Escape closes it and puts focus
 * back on the trigger, as does Tab before it moves on. Every item is a real
 * `<button role="menuitem">`, so nothing here depends on a pointer.
 *
 * The trigger handles Enter and Space in `keydown` and calls `preventDefault`.
 * A native button would otherwise turn both into a click, and the click
 * handler would toggle a second time, closing the menu the key press had just
 * opened. That is the shape of the bug an earlier publish control shipped
 * with: reached by keyboard it did something other than what the pointer did.
 */
export function ActionMenu({
  label,
  menuLabel,
  triggerAriaLabel,
  disabled = false,
  triggerClassName = "btn btn-outline btn-sm",
  wide = false,
  onOpenChange,
  children,
}: ActionMenuProps) {
  const [open, setOpen] = useState(false);
  const [focusTarget, setFocusTarget] = useState<FocusTarget | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  // Held in a ref so a call site may pass an inline closure without every
  // render tearing down the listeners below.
  const onOpenChangeRef = useRef(onOpenChange);
  onOpenChangeRef.current = onOpenChange;

  const close = useCallback(() => {
    setOpen(false);
    setFocusTarget(null);
    onOpenChangeRef.current?.(false);
  }, []);

  const closeAndRestoreFocus = useCallback(() => {
    close();
    triggerRef.current?.focus();
  }, [close]);

  const openWith = useCallback((target: FocusTarget) => {
    setOpen(true);
    setFocusTarget(target);
    onOpenChangeRef.current?.(true);
  }, []);

  // Landing focus inside the menu is what makes it operable at all: a keyboard
  // user who opened it would otherwise still be standing on the trigger, with
  // the items only reachable by tabbing past everything the menu covers.
  useEffect(() => {
    if (!open || !focusTarget) return;
    const container = menuRef.current;
    if (!container) return;
    const items = menuItems(container);
    if (items.length === 0) {
      container.focus();
      return;
    }
    (focusTarget === "first" ? items[0] : items[items.length - 1]).focus();
    setFocusTarget(null);
  }, [open, focusTarget]);

  // A pointer press outside closes without stealing focus. Restoring focus to
  // the trigger here would yank the caret out of whatever the person had just
  // clicked on.
  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: MouseEvent) {
      if (!wrapRef.current?.contains(event.target as Node)) close();
    }
    globalThis.document.addEventListener("mousedown", handlePointerDown);
    return () =>
      globalThis.document.removeEventListener("mousedown", handlePointerDown);
  }, [open, close]);

  function handleTriggerKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      openWith("first");
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      openWith("last");
      return;
    }
    if (
      event.key === "Enter" ||
      event.key === " " ||
      event.key === "Spacebar"
    ) {
      // See the note above: without this the browser's synthesized click would
      // run the toggle a second time.
      event.preventDefault();
      event.stopPropagation();
      if (open) closeAndRestoreFocus();
      else openWith("first");
      return;
    }
    if (event.key === "Escape" && open) {
      event.preventDefault();
      closeAndRestoreFocus();
    }
  }

  function handleMenuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeAndRestoreFocus();
      return;
    }
    if (event.key === "Tab") {
      // Focus goes back to the trigger and the browser's own Tab handling then
      // moves on from there, which lands on whatever follows this menu instead
      // of somewhere inside a popup that is about to disappear.
      closeAndRestoreFocus();
      return;
    }
    const items = menuItemsIn(menuRef.current);
    if (items.length === 0) return;
    // -1 when focus is on the popup itself rather than on a row, which is
    // where it sits when every item happens to be disabled.
    const current = items.indexOf(
      globalThis.document.activeElement as HTMLElement,
    );
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const delta = event.key === "ArrowDown" ? 1 : -1;
      items[nextFocusIndex(items, current, delta)]?.focus();
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      jumpMenuFocus(items, event.key);
    }
  }

  return (
    <div className="action-menu" ref={wrapRef}>
      <button
        ref={triggerRef}
        type="button"
        className={`${triggerClassName}${open ? " action-menu__trigger--open" : ""}`}
        aria-label={triggerAriaLabel}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={disabled}
        onKeyDown={handleTriggerKeyDown}
        onClick={() => {
          if (open) close();
          else openWith("first");
        }}
      >
        {label}
        <span aria-hidden="true" className="action-menu__caret">
          ▾
        </span>
      </button>
      {open && (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label={menuLabel ?? label}
          tabIndex={-1}
          className={`action-menu__popup${wide ? " action-menu__popup--wide" : ""}`}
          onKeyDown={handleMenuKeyDown}
        >
          {children(closeAndRestoreFocus)}
        </div>
      )}
    </div>
  );
}

export type ActionMenuItemProps = {
  label: string;
  /**
   * The accessible name, when the visible label is not unique on its own.
   * Two menus here carry a "PAGE XML + images" row, one under "Whole chapter"
   * and one under "Reviewed only", and a section heading is not part of a
   * button's accessible name.
   */
  ariaLabel?: string;
  /** The count or size badge that sits at the end of the row. */
  meta?: string;
  /** A second line under the label, for a consequence worth spelling out. */
  detail?: string;
  disabled?: boolean;
  /** Marks an item that destroys work, in colour and in the accessible name. */
  destructive?: boolean;
  onSelect: () => void;
};

/** One row of an {@link ActionMenu}. */
export function ActionMenuItem({
  label,
  ariaLabel,
  meta,
  detail,
  disabled = false,
  destructive = false,
  onSelect,
}: ActionMenuItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      aria-label={ariaLabel}
      disabled={disabled}
      className={`action-menu__item${destructive ? " action-menu__item--destructive" : ""}`}
      onClick={onSelect}
    >
      <span className="action-menu__item-body">
        <span className="action-menu__item-label">{label}</span>
        {detail && <span className="action-menu__item-detail">{detail}</span>}
      </span>
      {meta && <span className="action-menu__meta">{meta}</span>}
    </button>
  );
}

/** The small caps heading that groups a run of items. */
export function ActionMenuSection({ children }: { children: ReactNode }) {
  return <p className="action-menu__section">{children}</p>;
}

/** A line of context under a section heading. Never focusable. */
export function ActionMenuCaption({ children }: { children: ReactNode }) {
  return <p className="action-menu__caption">{children}</p>;
}

/** A consequence the reader has to see before choosing. Never focusable. */
export function ActionMenuWarning({ children }: { children: ReactNode }) {
  return (
    <p className="action-menu__warning">
      <span aria-hidden="true">⚠</span> {children}
    </p>
  );
}

export function ActionMenuDivider() {
  return <div className="action-menu__divider" role="separator" />;
}

export type ActionMenuConfirmProps = {
  /** What is about to happen, in the numbers the person is about to expose or lose. */
  question: string;
  /** The consequence, spelled out. */
  detail?: string;
  confirmLabel: string;
  cancelLabel?: string;
  destructive?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

/**
 * The step between choosing a heavy action and it happening.
 *
 * It replaces the menu's items rather than opening a dialog on top of them, so
 * the keyboard user who arrowed onto the item is still inside the same menu,
 * with the same Escape, and the count they are agreeing to is still on screen.
 *
 * A destructive confirm lands focus on Cancel. The safe option is the one that
 * should be under a hurried Enter.
 */
export function ActionMenuConfirm({
  question,
  detail,
  confirmLabel,
  cancelLabel = "Cancel",
  destructive = false,
  busy = false,
  onConfirm,
  onCancel,
}: ActionMenuConfirmProps) {
  return (
    <div className="action-menu__confirm">
      <p className="action-menu__confirm-question">{question}</p>
      {detail && <p className="action-menu__confirm-detail">{detail}</p>}
      <div className="action-menu__confirm-actions">
        <button
          type="button"
          role="menuitem"
          autoFocus={destructive}
          disabled={busy}
          className="action-menu__item action-menu__item--compact"
          onClick={onCancel}
        >
          {cancelLabel}
        </button>
        <button
          type="button"
          role="menuitem"
          autoFocus={!destructive}
          disabled={busy}
          className={`action-menu__item action-menu__item--compact action-menu__item--confirm${
            destructive ? " action-menu__item--destructive" : ""
          }`}
          onClick={onConfirm}
        >
          {confirmLabel}
        </button>
      </div>
    </div>
  );
}
