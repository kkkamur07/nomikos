import { useEffect, useState, type CSSProperties } from "react";

import { regionName, regionsInOrder, type Region } from "./regions";

/**
 * The region controls over the canvas (ADR 0011).
 *
 * A selected region gets its label and a delete; a selected Segment on a
 * Page that has regions gets the region it belongs to. One small panel in a
 * fixed corner rather than a menu per shape, so it reads the same in the
 * side-by-side layout, where the Segment strip is hidden, as in the strip
 * layout.
 */

type PageEditorRegionPanelProps = {
  regions: Region[];
  selectedRegionId: string | null;
  /** The selected Segment, when no region is selected. */
  selectedSegment: { id: string; block_id?: string | null } | null;
  selectedSegmentNumber: number | null;
  onRename: (regionId: string, label: string) => void;
  onDelete: (regionId: string) => void;
  onAssign: (segmentId: string, regionId: string | null) => void;
  onClose: () => void;
  error: string | null;
};

const PANEL_STYLE: CSSProperties = {
  position: "absolute",
  top: 12,
  left: 12,
  zIndex: 6,
  display: "flex",
  flexDirection: "column",
  gap: 8,
  width: 240,
  padding: "10px 12px",
  background: "rgba(255, 255, 255, 0.96)",
  border: "1px solid var(--border, #ebe8e3)",
  borderRadius: "var(--radius, 6px)",
  boxShadow: "0 4px 16px rgba(0, 0, 0, 0.12)",
  fontSize: "0.8125rem",
  color: "var(--text, #1a1a1a)",
};

const ROW_STYLE: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 8,
};

const INPUT_STYLE: CSSProperties = {
  width: "100%",
  padding: "5px 8px",
  border: "1px solid var(--border, #ebe8e3)",
  borderRadius: "var(--radius-sm, 4px)",
  font: "inherit",
};

function RegionLabelInput({
  region,
  onRename,
}: {
  region: Region;
  onRename: (regionId: string, label: string) => void;
}) {
  const [draft, setDraft] = useState(region.label ?? "");
  useEffect(() => setDraft(region.label ?? ""), [region.id, region.label]);
  const commit = () => onRename(region.id, draft);
  return (
    <input
      type="text"
      aria-label="Region label"
      placeholder="Folio label, e.g. 4v"
      maxLength={64}
      value={draft}
      autoFocus
      style={INPUT_STYLE}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commit();
        }
        // Esc belongs to the editor (deselect); keep typing keys local.
        if (event.key !== "Escape") event.stopPropagation();
      }}
    />
  );
}

export function PageEditorRegionPanel({
  regions,
  selectedRegionId,
  selectedSegment,
  selectedSegmentNumber,
  onRename,
  onDelete,
  onAssign,
  onClose,
  error,
}: PageEditorRegionPanelProps) {
  const region = regions.find((candidate) => candidate.id === selectedRegionId);
  const ordered = regionsInOrder(regions);

  if (region) {
    return (
      <section
        style={PANEL_STYLE}
        aria-label="Selected region"
        className="pe-region-panel"
      >
        <div style={ROW_STYLE}>
          <strong>{regionName(region, regions)}</strong>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={onClose}
            aria-label="Close region panel"
          >
            ×
          </button>
        </div>
        <RegionLabelInput region={region} onRename={onRename} />
        <button
          type="button"
          className="btn btn--danger-ghost btn--sm"
          onClick={() => onDelete(region.id)}
        >
          Delete region
        </button>
        {error ? <p role="alert">{error}</p> : null}
      </section>
    );
  }

  if (selectedSegment && regions.length > 0) {
    const current = ordered.some(
      (candidate) => candidate.id === selectedSegment.block_id,
    )
      ? (selectedSegment.block_id ?? "")
      : "";
    return (
      <section
        style={PANEL_STYLE}
        aria-label="Segment region"
        className="pe-region-panel"
      >
        <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span>
            Assign segment
            {selectedSegmentNumber !== null
              ? ` ${selectedSegmentNumber}`
              : ""}{" "}
            to region
          </span>
          <select
            aria-label="Assign to region"
            value={current}
            style={INPUT_STYLE}
            onChange={(event) =>
              onAssign(selectedSegment.id, event.target.value || null)
            }
          >
            {ordered.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {regionName(candidate, regions)}
              </option>
            ))}
            <option value="">No region</option>
          </select>
        </label>
        {error ? <p role="alert">{error}</p> : null}
      </section>
    );
  }

  return null;
}
