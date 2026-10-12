import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ActionMenu, ActionMenuItem } from "./ActionMenu";

function openMenu(onSelect: () => void = () => {}) {
  render(
    <div>
      <button type="button">outside</button>
      <ActionMenu label="Actions" menuLabel="Actions menu">
        {(close) => (
          <>
            <label>
              Pick
              <select aria-label="Pick">
                <option value="a">a</option>
              </select>
            </label>
            <ActionMenuItem
              label="Run it"
              onSelect={() => {
                onSelect();
                close();
              }}
            />
            <ActionMenuItem label="Second" onSelect={() => {}} />
          </>
        )}
      </ActionMenu>
    </div>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Actions" }));
  return screen.getByRole("menu", { name: "Actions menu" });
}

function firstItem(): HTMLElement {
  return screen.getByRole("menuitem", { name: "Run it" });
}

describe("ActionMenu focusout close", () => {
  it("keeps the popup open on a null-target blur and the next click still runs", () => {
    const onSelect = vi.fn();
    const menu = openMenu(onSelect);
    const item = firstItem();
    (item as HTMLButtonElement).focus();

    // A Safari click that moves no focus, a press on a non-focusable spot,
    // or a window blur: no named target, so nothing closes.
    fireEvent.blur(item, { relatedTarget: null });
    expect(menu).toBeInTheDocument();

    fireEvent.click(item);
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("closes when focus moves to a real target outside", async () => {
    openMenu();
    const item = firstItem();
    (item as HTMLButtonElement).focus();

    fireEvent.blur(item, {
      relatedTarget: screen.getByRole("button", { name: "outside" }),
    });
    await waitFor(() =>
      expect(screen.queryByRole("menu", { name: "Actions menu" })).toBeNull(),
    );
  });

  it("closes when focus leaves the whole wrapper through the trigger", async () => {
    openMenu();
    // Shift+Tab walked popup to trigger (inside the wrap, still open), then
    // one more Shift+Tab leaves the widget entirely.
    const trigger = screen.getByRole("button", { name: "Actions" });
    (trigger as HTMLButtonElement).focus();
    fireEvent.blur(trigger, {
      relatedTarget: screen.getByRole("button", { name: "outside" }),
    });
    await waitFor(() =>
      expect(screen.queryByRole("menu", { name: "Actions menu" })).toBeNull(),
    );
  });

  it("stays open when focus moves inside", () => {
    const menu = openMenu();
    const item = firstItem();
    (item as HTMLButtonElement).focus();

    fireEvent.blur(item, {
      relatedTarget: screen.getByRole("combobox", { name: "Pick" }),
    });
    expect(menu).toBeInTheDocument();
  });

  it("closes on a pointer press outside", async () => {
    openMenu();
    fireEvent.mouseDown(screen.getByRole("button", { name: "outside" }));
    await waitFor(() =>
      expect(screen.queryByRole("menu", { name: "Actions menu" })).toBeNull(),
    );
  });
});
