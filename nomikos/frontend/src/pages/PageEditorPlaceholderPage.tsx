import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { api, type LayoutPoint, type LinePoint } from "../api/client";
import { invalidateAfter } from "../api/resources";
import { useHostPreference } from "../inference";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { PageEditorCanvas } from "../components/page-editor/PageEditorCanvas";
import { PageEditorSplitPane } from "../components/page-editor/PageEditorSplitPane";
import { PageEditorTranscriptEditor } from "../components/page-editor/PageEditorTranscriptEditor";
import {
  PageEditorPageRail,
  PageEditorPageRailTab,
  PAGE_RAIL_TOGGLE_ID,
} from "../components/page-editor/PageEditorPageRail";
import { PageEditorTranscriptionStrip } from "../components/page-editor/PageEditorTranscriptionStrip";
import { PageEditorShell } from "../components/page-editor/PageEditorShell";
import {
  loadPageEditorSettings,
  savePageEditorSettings,
} from "../components/page-editor/pageEditorSettings";
import {
  PageEditorStatusAlerts,
  hasPageEditorStatusAlerts,
} from "../components/page-editor/PageEditorStatusAlerts";
import { PageEditorInferenceBanner } from "../components/page-editor/PageEditorInferenceBanner";
import { PageEditorToolbar } from "../components/page-editor/PageEditorToolbar";
import { PageEditorTranscriptionPdfWrap } from "../components/page-editor/PageEditorTranscriptionPdfWrap";
import {
  rectanglePoints,
  removePolygonVertex,
} from "../components/page-editor/canvasGeometry";
import {
  pageNeighbours,
  useDocumentPaging,
  useLayoutMutations,
  usePageEditorData,
  usePageEditorJobQueue,
  usePageEditorRunState,
  usePairingState,
  useSegmentHealth,
} from "../components/page-editor/hooks";
import {
  loadPageRailOpen,
  savePageRailOpen,
} from "../components/page-editor/pageEditorSettings";
import {
  segmentHasGroundTruth,
  segmentIdsWithGroundTruth,
  withLocalGroundTruth,
} from "../components/page-editor/hooks/utils";

export function PageEditorPlaceholderPage() {
  const {
    projectId,
    documentId,
    partId: routePartId,
  } = useParams<{
    projectId: string;
    documentId: string;
    partId: string;
  }>() ?? {};

  const [drawMode, setDrawMode] = useState<"none" | "rectangle" | "polygon">(
    "none",
  );
  const [draftStart, setDraftStart] = useState<LayoutPoint | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<LinePoint[]>([]);
  const [actionsOpen, setActionsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [segmentHealthOpen, setSegmentHealthOpen] = useState(false);
  const [canvasSettings, setCanvasSettings] = useState(loadPageEditorSettings);
  const [transcriptionPdfOpen, setTranscriptionPdfOpen] = useState(false);
  const [transcriptionPdfRefreshKey, setTranscriptionPdfRefreshKey] =
    useState(0);
  const [stripDismissed, setStripDismissed] = useState(false);
  const [pageRailOpen, setPageRailOpen] = useState(loadPageRailOpen);
  const [reclaimRailFocus, setReclaimRailFocus] = useState(false);
  const [selectedVertexIndex, setSelectedVertexIndex] = useState<number | null>(
    null,
  );
  const [vertexCommitSignal, setVertexCommitSignal] = useState(0);
  // The platform refused a submission because no **inference host** had
  // capacity. Held on the page, not in a toast: it is an explanation to act on.
  const [submissionRefusal, setSubmissionRefusal] = useState<string | null>(
    null,
  );
  const [hoveredSegmentId, setHoveredSegmentId] = useState<string | null>(null);
  const [focusRequest, setFocusRequest] = useState<{
    segmentId: string;
    nonce: number;
  } | null>(null);
  // The document response carries no script or direction field, so the transcript reads left to right.
  const textDirection = "ltr" as const;

  // The page the editor is on. The route seeds it and mirrors it back, but the
  // editor turns pages by changing this rather than by re-entering the route,
  // so the toolbar, the page rail and the focused control survive a page turn.
  const { activePartId: partId, goToPart } = useDocumentPaging({
    projectId,
    documentId,
    routePartId,
  });

  const editorData = usePageEditorData(projectId, documentId, partId, () => {
    setDrawMode("none");
    setDraftPolygon([]);
    setDraftStart(null);
    setStripDismissed(false);
    setSelectedVertexIndex(null);
    setHoveredSegmentId(null);
    setFocusRequest(null);
  });
  const {
    document,
    setDocument,
    part,
    layout,
    setLayout,
    lines,
    setLines,
    loading,
    partLoading,
    error,
    layoutError,
    lineError,
    setLineError,
    transcriptionLayers,
    setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    textLines,
    setTextLines,
    pairingProgress,
    setPairingProgress,
    pairingError,
    setPairingError,
    transcribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    segmentModels,
    selectedSegmentModelId,
    setSelectedSegmentModelId,
    parts,
    partIndex,
  } = editorData;

  const { previousPartId, nextPartId } = pageNeighbours(parts, partId);
  const goToPreviousPage = useCallback(() => {
    if (previousPartId) goToPart(previousPartId);
  }, [previousPartId, goToPart]);
  const goToNextPage = useCallback(() => {
    if (nextPartId) goToPart(nextPartId);
  }, [nextPartId, goToPart]);

  function handlePageRailOpenChange(open: boolean) {
    setPageRailOpen(open);
    savePageRailOpen(open);
    // Closing the rail unmounts whatever inside it had focus, and the rail's
    // own collapse button is the likeliest thing that did. The effect below
    // hands focus to the edge tab that takes the rail's place once the rail
    // is actually gone.
    if (!open) setReclaimRailFocus(true);
  }

  useEffect(() => {
    if (!reclaimRailFocus) return;
    setReclaimRailFocus(false);
    globalThis.document.getElementById(PAGE_RAIL_TOGGLE_ID)?.focus();
  }, [reclaimRailFocus]);

  const segmentHealth = useSegmentHealth({
    projectId,
    documentId,
    partId,
    open: segmentHealthOpen,
    setLines,
  });

  const jobQueue = usePageEditorJobQueue();
  const hostPreference = useHostPreference();

  /** Turn the account setting off, from the install prompt's escape hatch. */
  function preferCloudInferencePermanently() {
    void hostPreference.setPreferLocalInference(false);
  }

  const pairing = usePairingState({
    projectId,
    documentId,
    partId,
    lines,
    setLines,
    transcriptionLayers,
    setTranscriptionLayers,
    selectedTranscriptionLayerId,
    setSelectedTranscriptionLayerId,
    groundTruthTranscriptionId,
    setTextLines,
    setPairingProgress,
    setPairingError,
    selectedTranscribeModelId,
    setSubmissionRefusal,
    trackJobAndWait: jobQueue.trackAndWait,
  });

  const layoutMutations = useLayoutMutations({
    projectId,
    documentId,
    partId,
    layout,
    setLayout,
    lines,
    setLines,
    setLineError,
    setTextLines,
    setPairingProgress,
    setPairingError,
    selectedSegmentId: pairing.selectedSegmentId,
    setSelectedSegmentId: pairing.setSelectedSegmentId,
    setApprovedTextDraft: pairing.setApprovedTextDraft,
    selectedSegmentModelId,
    onDrawComplete: () => setDrawMode("none"),
    setSubmissionRefusal,
    trackJobAndWait: jobQueue.trackAndWait,
    subscribeToJobCompletion: jobQueue.subscribeToJobCompletion,
  });

  const {
    selectedLineId,
    setSelectedLineId,
    setSelectedLineSnapshot,
    saveMessage,
    setSaveMessage,
    mutationError,
    segmenting,
    segmentMessage,
    moveSelectedBaseline,
    saveSelectedLine,
    resetSelectedLine,
    replaceWithManualLine,
    updateSegmentPoints,
    deleteSelectedSegment,
    undoEdit,
    redoEdit,
    runAutoSegment,
  } = layoutMutations;

  const {
    selectedSegmentId,
    setSelectedSegmentId,
    approvedTextDraft,
    setApprovedTextDraft,
    transcriptionSaveMessage,
    ocrRunning,
    ocrScope,
    ocrMessage,
    selectedSegment,
    selectedSegmentNumber,
    selectedTranscriptionLayer,
    pairTextLine,
    saveApprovedText,
    selectTranscriptionLayer,
    saveGroundTruthText,
    runSegmentOcr,
    runPageOcr,
    promoteSelectedSegmentToGroundTruth,
    selectSegment,
    navigateSegment,
  } = pairing;

  // Latest selection, readable after an await: a save may resolve after the user
  // has switched segments, and the strip must not be dismissed on the new one.
  const selectedSegmentIdRef = useRef(selectedSegmentId);
  selectedSegmentIdRef.current = selectedSegmentId;

  const pairedIds = useMemo(() => segmentIdsWithGroundTruth(lines), [lines]);
  const stripVisible =
    Boolean(selectedSegment) && !stripDismissed && !canvasSettings.sideBySide;

  function handleSelectSegment(lineId: string) {
    setSelectedLineId(null);
    setSaveMessage(null);
    setStripDismissed(false);
    setSelectedVertexIndex(null);
    selectSegment(lineId);
  }

  async function handleCommitText(lineId: string, text: string): Promise<void> {
    if (!projectId || !documentId || !partId) {
      setPairingError("Page context is missing. Reload and try again.");
      throw new Error("Page context is missing.");
    }
    if (!groundTruthTranscriptionId) {
      setPairingError("Ground truth transcription layer is not available.");
      throw new Error("Ground truth transcription layer is not available.");
    }
    try {
      const updated = await api.updateGroundTruthLineText(
        projectId,
        documentId,
        groundTruthTranscriptionId,
        lineId,
        { text },
      );
      setLines((current) =>
        withLocalGroundTruth(
          current,
          groundTruthTranscriptionId,
          lineId,
          updated.text,
        ),
      );
      const pairingData = await api.getPagePairing(
        projectId,
        documentId,
        partId,
      );
      setTextLines(pairingData.text_lines);
      setPairingProgress(pairingData.pairing_progress);
      invalidateAfter.partContentChanged(projectId, documentId);
      setPairingError(null);
    } catch (err) {
      setPairingError(
        err instanceof Error
          ? err.message
          : "Failed to save Ground truth text.",
      );
      throw err;
    }
  }

  function handleRemoveSelectedVertex() {
    if (!selectedSegmentId || selectedVertexIndex === null) return;
    const segment = lines.find((line) => line.id === selectedSegmentId);
    if (!segment) return;
    const nextPoints = removePolygonVertex(segment.points, selectedVertexIndex);
    if (!nextPoints) return;
    setSelectedVertexIndex(null);
    void updateSegmentPoints(selectedSegmentId, nextPoints);
  }

  const runState = usePageEditorRunState({ segmenting, ocrRunning, ocrScope });

  // A page turn keeps the editor on screen, so the hint is where it says the
  // next page's Segments are still on their way.
  const canvasHint = partLoading
    ? partIndex
      ? `Loading page ${partIndex}…`
      : "Loading page…"
    : runState.processingLabel
      ? `${runState.processingLabel}…`
      : drawMode === "polygon"
        ? draftPolygon.length === 0
          ? "Polygon: click to place the first corner · hold Space to pan"
          : `Polygon: ${draftPolygon.length} point${draftPolygon.length === 1 ? "" : "s"} · click to add · double-click or Enter to finish`
        : drawMode === "rectangle"
          ? "Rectangle: drag to draw a segment · hold Space to pan"
          : selectedSegment
            ? selectedVertexIndex !== null
              ? `Segment ${selectedSegmentNumber} · vertex ${selectedVertexIndex + 1} selected · Delete removes point · Esc deselects`
              : `Segment ${selectedSegmentNumber} ${
                  segmentHasGroundTruth(selectedSegment) ? "paired" : "unpaired"
                } · click edge to add · click handle to select · Esc deselects`
            : "Drag to pan · scroll to zoom · Page Up/Down for pages · select a segment to edit it";

  function pickDrawMode(nextMode: "rectangle" | "polygon") {
    setDrawMode((mode) => (mode === nextMode ? "none" : nextMode));
    setDraftPolygon([]);
    setDraftStart(null);
    setActionsOpen(false);
  }

  const handlePanSelect = useCallback(() => {
    setDrawMode("none");
    setDraftPolygon([]);
    setDraftStart(null);
    setActionsOpen(false);
    setVertexCommitSignal((value) => value + 1);
    setSelectedVertexIndex(null);
    setSelectedSegmentId(null);
    setSelectedLineId(null);
    setApprovedTextDraft("");
    setStripDismissed(false);
  }, [setSelectedSegmentId, setSelectedLineId, setApprovedTextDraft]);

  function completeDraftPolygon() {
    if (draftPolygon.length >= 3) {
      void replaceWithManualLine("polygon", draftPolygon);
    }
    setDraftPolygon([]);
  }

  useKeyboardShortcuts({
    onDrawBox: () => pickDrawMode("rectangle"),
    onDrawPolygon: () => pickDrawMode("polygon"),
    onDelete:
      selectedVertexIndex !== null && selectedSegmentId
        ? () => handleRemoveSelectedVertex()
        : selectedSegmentId || selectedLineId
          ? () => {
              if (selectedSegmentId) void deleteSelectedSegment();
              if (selectedLineId) void resetSelectedLine();
            }
          : undefined,
    onEscape: handlePanSelect,
    onUndo: () => void undoEdit(),
    onRedo: () => void redoEdit(),
    onEnter:
      drawMode === "polygon" && draftPolygon.length >= 3
        ? completeDraftPolygon
        : undefined,
    onPreviousPage: previousPartId ? goToPreviousPage : undefined,
    onNextPage: nextPartId ? goToNextPage : undefined,
  });

  function handleCanvasSettingsChange(next: typeof canvasSettings) {
    setCanvasSettings(next);
    savePageEditorSettings(next);
  }

  function openTranscriptionPdf() {
    setTranscriptionPdfRefreshKey(Date.now());
    setTranscriptionPdfOpen(true);
    setActionsOpen(false);
  }

  function refreshTranscriptionPdf() {
    setTranscriptionPdfRefreshKey(Date.now());
  }

  const statusAlertProps = {
    submissionRefusal,
    saveMessage,
    transcriptionSaveMessage,
    ocrMessage,
    segmentMessage,
    mutationError,
    pairingError,
    layoutError,
    lineError,
  };

  const documentHref =
    projectId && documentId
      ? `/projects/${projectId}/documents/${documentId}`
      : "/projects";

  const imageWidth = part?.width ?? 640;
  const imageHeight = part?.height ?? 900;
  const preferredModelLayerId =
    selectedTranscriptionLayer?.kind === "model"
      ? selectedTranscriptionLayer.id
      : null;

  const canvasPaneContent = (
    <>
      <PageEditorCanvas
        imageUrl={part?.image_url ?? ""}
        imageAlt={`Page ${partIndex}`}
        imageWidth={imageWidth}
        imageHeight={imageHeight}
        layout={layout}
        lines={lines}
        selectedSegmentId={selectedSegmentId}
        pairedSegmentIds={pairedIds}
        hoveredSegmentId={hoveredSegmentId}
        onHoverSegment={setHoveredSegmentId}
        focusRequest={focusRequest}
        settings={canvasSettings}
        drawingRectangle={drawMode === "rectangle"}
        drawingPolygon={drawMode === "polygon"}
        draftStart={draftStart}
        draftPolygon={draftPolygon}
        onDraftStart={setDraftStart}
        onRectangleDrawn={async (end) => {
          if (!draftStart) return;
          const rectangle = rectanglePoints(draftStart, end);
          await replaceWithManualLine("rectangle", rectangle);
          setDraftStart(null);
        }}
        onPolygonPoint={(point) =>
          setDraftPolygon((current) => [...current, point])
        }
        onPolygonComplete={completeDraftPolygon}
        onSelectLine={(lineId) => {
          const selectedLine = layout.lines.find((line) => line.id === lineId);
          setSelectedLineId(lineId);
          setSelectedSegmentId(null);
          setSelectedVertexIndex(null);
          setSelectedLineSnapshot({
            baseline: selectedLine?.baseline,
            mask: selectedLine?.mask,
          });
        }}
        onSelectSegment={handleSelectSegment}
        segmentVertexEditEnabled={
          drawMode === "none" && Boolean(selectedSegmentId)
        }
        onSelectTool={handlePanSelect}
        onPickDrawMode={pickDrawMode}
        canDelete={Boolean(selectedSegmentId || selectedLineId)}
        onDeleteSelected={() => {
          if (selectedSegmentId) void deleteSelectedSegment();
          if (selectedLineId) void resetSelectedLine();
        }}
        selectedVertexIndex={selectedVertexIndex}
        onSelectedVertexChange={setSelectedVertexIndex}
        commitSignal={vertexCommitSignal}
        onSegmentPointsChange={updateSegmentPoints}
      />
      <p
        className={`pe-canvas-hint${runState.processingKind ? " pe-canvas-hint--processing" : ""}`}
        id="canvas-hint"
        role="status"
      >
        {canvasHint}
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

  return (
    <PageEditorShell
      loading={loading}
      backHref={documentHref}
      unavailableDescription={
        error || !document || !part
          ? (error ?? "This document part was not found.")
          : null
      }
      showStatusAlerts={hasPageEditorStatusAlerts(statusAlertProps)}
      statusAlerts={<PageEditorStatusAlerts {...statusAlertProps} />}
      inferenceBanner={
        <PageEditorInferenceBanner
          hasLocalCapacity={hostPreference.hasLocalCapacity}
          loading={hostPreference.loading}
          preferLocalInference={hostPreference.preferLocalInference}
          onRetry={() => void hostPreference.refresh()}
          onUseCloudInstead={preferCloudInferencePermanently}
        />
      }
      toolbar={
        document && part ? (
          <PageEditorToolbar
            projectId={projectId}
            documentId={documentId}
            partId={part.id}
            document={document}
            partIndex={partIndex ?? 1}
            pageCount={parts.length}
            hasPreviousPart={Boolean(previousPartId)}
            hasNextPart={Boolean(nextPartId)}
            onPreviousPart={goToPreviousPage}
            onNextPart={goToNextPage}
            lines={lines}
            pairingProgress={pairingProgress}
            selectedSegmentId={selectedSegmentId}
            textLines={textLines}
            onPairTextLine={pairTextLine}
            onDocumentWorkflowChange={(workflow) => {
              setDocument((current) =>
                current ? { ...current, workflow } : current,
              );
              // Publishing is the one write in this editor that changes what a
              // reader can reach at all, so the local copy above isn't
              // enough: without this, the document list, detail page and
              // public page would keep showing the old status.
              if (projectId && documentId) {
                invalidateAfter.documentUpdated(projectId, documentId);
              }
            }}
            actionsOpen={actionsOpen}
            onActionsOpenChange={setActionsOpen}
            segmenting={segmenting}
            ocrRunning={ocrRunning}
            ocrScope={ocrScope}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            segmentModels={segmentModels}
            selectedSegmentModelId={selectedSegmentModelId}
            onSelectedSegmentModelIdChange={setSelectedSegmentModelId}
            onRunAutoSegment={runAutoSegment}
            onRunSegmentOcr={runSegmentOcr}
            onRunPageOcr={runPageOcr}
            transcriptionPdfOpen={transcriptionPdfOpen}
            onOpenTranscriptionPdf={openTranscriptionPdf}
            onCloseTranscriptionPdf={() => setTranscriptionPdfOpen(false)}
            segmentHealthOpen={segmentHealthOpen}
            onSegmentHealthOpenChange={setSegmentHealthOpen}
            segmentHealth={segmentHealth}
            settingsOpen={settingsOpen}
            onSettingsOpenChange={setSettingsOpen}
            canvasSettings={canvasSettings}
            onCanvasSettingsChange={handleCanvasSettingsChange}
            preferLocalInference={hostPreference.preferLocalInference}
            onPreferLocalInferenceChange={(preferLocal) =>
              void hostPreference.setPreferLocalInference(preferLocal)
            }
            preferenceSaving={hostPreference.saving}
            hasLocalCapacity={hostPreference.hasLocalCapacity}
            hostPreferenceLoading={hostPreference.loading}
          />
        ) : null
      }
    >
      {document && part && (
        <div className="pe-workspace">
          <div className="pe-body">
            {parts.length > 1 &&
              (pageRailOpen ? (
                <PageEditorPageRail
                  parts={parts}
                  activePartId={partId}
                  onSelectPart={goToPart}
                  onCollapse={() => handlePageRailOpenChange(false)}
                />
              ) : (
                <PageEditorPageRailTab
                  onExpand={() => handlePageRailOpenChange(true)}
                />
              ))}
            <div className="pe-canvas-pane">
              {canvasSettings.sideBySide ? (
                <PageEditorSplitPane
                  ratio={canvasSettings.splitRatio}
                  onRatioChange={(ratio) =>
                    handleCanvasSettingsChange({
                      ...canvasSettings,
                      splitRatio: ratio,
                    })
                  }
                  left={canvasPaneContent}
                  right={
                    <PageEditorTranscriptEditor
                      lines={lines}
                      selectedSegmentId={selectedSegmentId}
                      hoveredSegmentId={hoveredSegmentId}
                      textDirection={textDirection}
                      preferredLayerId={preferredModelLayerId}
                      onSelectSegment={handleSelectSegment}
                      onHoverSegment={setHoveredSegmentId}
                      onFocusSegment={(id) =>
                        setFocusRequest({ segmentId: id, nonce: Date.now() })
                      }
                      onCommitText={handleCommitText}
                    />
                  }
                />
              ) : (
                canvasPaneContent
              )}
            </div>
            {transcriptionPdfOpen && projectId && documentId && partId && (
              <PageEditorTranscriptionPdfWrap
                projectId={projectId}
                documentId={documentId}
                partId={partId}
                downloadFilename={`${document.name.replace(/\s+/g, "_")}_page_${partIndex}_transcription.pdf`}
                refreshKey={transcriptionPdfRefreshKey}
                onClose={() => setTranscriptionPdfOpen(false)}
                onRefresh={refreshTranscriptionPdf}
              />
            )}
          </div>

          {selectedLineId && (
            <div className="pe-baseline-bar">
              <span>Selected baseline</span>
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => moveSelectedBaseline(5)}
              >
                Move baseline down
              </button>
              <button
                type="button"
                className="btn btn--active btn--sm"
                onClick={() => void saveSelectedLine()}
              >
                Save layout
              </button>
              <button
                type="button"
                className="btn btn--danger-ghost btn--sm"
                onClick={() => void resetSelectedLine()}
              >
                Reset layout
              </button>
            </div>
          )}

          <PageEditorTranscriptionStrip
            visible={stripVisible}
            transcriptionLayers={transcriptionLayers}
            selectedTranscriptionLayerId={selectedTranscriptionLayerId}
            onSelectTranscriptionLayer={selectTranscriptionLayer}
            selectedSegmentNumber={selectedSegmentNumber}
            selectedSegment={selectedSegment}
            selectedTranscriptionLayer={selectedTranscriptionLayer}
            approvedTextDraft={approvedTextDraft}
            onApprovedTextDraftChange={setApprovedTextDraft}
            onSaveGroundTruthText={async () => {
              const savedSegmentId = selectedSegmentId;
              if (
                (await saveGroundTruthText()) &&
                selectedSegmentIdRef.current === savedSegmentId
              ) {
                setStripDismissed(true);
              }
            }}
            onPromoteSelectedSegmentToGroundTruth={
              promoteSelectedSegmentToGroundTruth
            }
            onRunSegmentOcr={runSegmentOcr}
            onNavigateSegment={navigateSegment}
            onDismiss={() => setStripDismissed(true)}
            lines={lines}
            selectedSegmentId={selectedSegmentId}
            onSaveApprovedText={async () => {
              const savedSegmentId = selectedSegmentId;
              if (
                (await saveApprovedText()) &&
                selectedSegmentIdRef.current === savedSegmentId
              ) {
                setStripDismissed(true);
              }
            }}
            transcribeModels={transcribeModels}
            selectedTranscribeModelId={selectedTranscribeModelId}
            onSelectedTranscribeModelIdChange={setSelectedTranscribeModelId}
            ocrRunning={ocrRunning}
            ocrScope={ocrScope}
            backgroundJobsActive={runState.backgroundJobsActive}
          />
        </div>
      )}
    </PageEditorShell>
  );
}
