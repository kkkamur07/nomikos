type PageEditorSegmentNavProps = {
  segmentNumber: number | null;
  totalSegments: number;
  onPrevious: () => void;
  onNext: () => void;
  disabled?: boolean;
};

function navLabel(segmentNumber: number | null, totalSegments: number) {
  if (totalSegments === 0) return "-";
  return `${segmentNumber ?? "-"} / ${totalSegments}`;
}

export function PageEditorSegmentNav({
  segmentNumber,
  totalSegments,
  onPrevious,
  onNext,
  disabled = false,
}: PageEditorSegmentNavProps) {
  // With no segment selected, or none at all, neither button moves.
  const stuck = disabled || segmentNumber === null || totalSegments === 0;
  const atStart = stuck || segmentNumber <= 1;
  const atEnd = stuck || segmentNumber >= totalSegments;

  return (
    <nav className="pe-seg-nav" aria-label="Segment navigation">
      <button
        type="button"
        className="pe-seg-nav__btn"
        aria-label="Previous segment"
        disabled={atStart}
        onClick={onPrevious}
      >
        ‹
      </button>
      <span className="pe-seg-nav__label" aria-live="polite">
        {navLabel(segmentNumber, totalSegments)}
      </span>
      <button
        type="button"
        className="pe-seg-nav__btn"
        aria-label="Next segment"
        disabled={atEnd}
        onClick={onNext}
      >
        ›
      </button>
    </nav>
  );
}
