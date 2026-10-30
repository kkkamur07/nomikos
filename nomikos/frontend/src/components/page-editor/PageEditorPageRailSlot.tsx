import { useEffect, useState } from "react";
import type { DocumentPartResponse } from "../../api/client";
import {
  PAGE_RAIL_TOGGLE_ID,
  PageEditorPageRail,
  PageEditorPageRailTab,
} from "./PageEditorPageRail";
import { loadPageRailOpen, savePageRailOpen } from "./pageEditorSettings";

type PageEditorPageRailSlotProps = {
  parts: DocumentPartResponse[];
  activePartId: string | undefined;
  onSelectPart: (partId: string) => void;
};

/**
 * The page rail, or the edge tab that stands in for it when collapsed. The
 * open state is remembered across visits. A one-page document has no rail.
 */
export function PageEditorPageRailSlot({
  parts,
  activePartId,
  onSelectPart,
}: PageEditorPageRailSlotProps) {
  const [open, setOpen] = useState(loadPageRailOpen);
  const [reclaimFocus, setReclaimFocus] = useState(false);

  function handleOpenChange(nextOpen: boolean) {
    setOpen(nextOpen);
    savePageRailOpen(nextOpen);
    // Closing the rail unmounts whatever inside it had focus, and the rail's
    // own collapse button is the likeliest thing that did. The effect below
    // hands focus to the edge tab that takes the rail's place once the rail
    // is actually gone.
    if (!nextOpen) setReclaimFocus(true);
  }

  useEffect(() => {
    if (!reclaimFocus) return;
    setReclaimFocus(false);
    globalThis.document.getElementById(PAGE_RAIL_TOGGLE_ID)?.focus();
  }, [reclaimFocus]);

  if (parts.length <= 1) return null;
  if (!open) {
    return <PageEditorPageRailTab onExpand={() => handleOpenChange(true)} />;
  }
  return (
    <PageEditorPageRail
      parts={parts}
      activePartId={activePartId}
      onSelectPart={onSelectPart}
      onCollapse={() => handleOpenChange(false)}
    />
  );
}
