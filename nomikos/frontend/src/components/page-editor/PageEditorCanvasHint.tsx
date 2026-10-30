type PageEditorCanvasHintProps = {
  hint: string;
  processing: boolean;
};

/** The status line under the canvas, and the paired / unpaired legend. */
export function PageEditorCanvasHint({
  hint,
  processing,
}: PageEditorCanvasHintProps) {
  return (
    <>
      <p
        className={`pe-canvas-hint${processing ? " pe-canvas-hint--processing" : ""}`}
        id="canvas-hint"
        role="status"
      >
        {hint}
      </p>
      <div className="pe-seg-legend" aria-label="Segment pairing">
        <div className="pe-seg-legend__item">
          <span className="pe-seg-legend__swatch pe-seg-legend__swatch--paired" />
          paired
        </div>
        <div className="pe-seg-legend__item">
          <span className="pe-seg-legend__swatch pe-seg-legend__swatch--unpaired" />
          unpaired
        </div>
      </div>
    </>
  );
}
