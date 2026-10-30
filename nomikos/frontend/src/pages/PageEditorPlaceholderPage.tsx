import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import type { LayoutPoint, LinePoint } from "../api/client";
import { invalidateAfter } from "../api/resources";
import { useHostPreference } from "../inference";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { PageEditorCanvas } from "../components/page-editor/PageEditorCanvas";
import type { DrawTool } from "../components/page-editor/PageEditorCanvasIsland";
import { PageEditorRegionPanel } from "../components/page-editor/PageEditorRegionPanel";
import { PageEditorSplitPane } from "../components/page-editor/PageEditorSplitPane";
import {
  confirmLeaveWithUnsavedRows,
  PageEditorTranscriptEditor,
} from "../components/page-editor/PageEditorTranscriptEditor";
import { PageEditorMergeConflictDialog } from "../components/page-editor/PageEditorMergeConflictDialog";
import {
  inPageOrder,
  prunedGroup,
} from "../components/page-editor/segmentSelection";
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
  useRegions,
  useSegmentHealth,
  useSegmentMerge,
} from "../components/page-editor/hooks";
import {
  loadPageRailOpen,
  savePageRailOpen,
} from "../components/page-editor/pageEditorSettings";
import {
  segmentHasGroundTruth,
  segmentIdsWithGroundTruth,
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

  const [drawMode, setDrawMode] = useState<"none" | DrawTool>("none");
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

  // The page the editor is on. The route seeds it and mirrors it back, but the
  // editor turns pages by changing this rather than by re-entering the route,
  // so the toolbar, the page rail and the focused control survive a page turn.
  const { activePartId: partId, goToPart: goToPartUnguarded } =
    useDocumentPaging({
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

  // A page turn drops the rows on screen: ask first while any holds unsaved text.
  const goToPart = useCallback(
    (id: string) => {
      if (!confirmLeaveWithUnsavedRows(lines.map((line) => line.id))) return;
      goToPartUnguarded(id);
    },
    [lines, goToPartUnguarded],
  );

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
    lines,
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

  const segmentMerge = useSegmentMerge({
    projectId,
    documentId,
    partId,
    setLines,
    setTextLines,
    setPairingProgress,
    // The group is gone once it has been merged; the survivor stays selected so
    // the strip shows the text that was kept rather than nothing at all.
    onMerged: (survivingLineId) => {
      pairing.setSelectedSegmentIds([]);
      pairing.setSelectedSegmentId(survivingLineId);
      setSelectedVertexIndex(null);
    },
  });

  const mergeSelectedSegments = useCallback(() => {
    const group = inPageOrder(
      pairing.selectedSegmentIds,
      lines.map((line) => line.id),
    );
    if (group.length < 2) return;
    void segmentMerge.mergeSegments(group);
  }, [pairing.selectedSegmentIds, lines, segmentMerge]);

  // A merge, a delete or a re-segment replaces the line list. Ids the group
  // still names would be a 404 on the next merge, so they go with them.
  useEffect(() => {
    const kept = prunedGroup(
      pairing.selectedSegmentIds,
      lines.map((line) => line.id),
    );
    if (kept !== pairing.selectedSegmentIds) {
      pairing.setSelectedSegmentIds(kept);
    }
  }, [lines, pairing]);

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

  const regions = useRegions({
    projectId,
    documentId,
    partId,
    layout,
    setLayout,
    lines,
    setLines,
    onRecordEdit: layoutMutations.recordRegionEdit,
  });
  const { selectedRegionId, setSelectedRegionId } = regions;

  const {
    selectedSegmentId,
    setSelectedSegmentId,
    setSelectedSegmentIds,
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
    commitLineText,
    reloadLineText,
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

  function handleSelectSegment(lineId: string, additive = false) {
    setSelectedRegionId(null);
    setSelectedLineId(null);
    setSaveMessage(null);
    setStripDismissed(false);
    setSelectedVertexIndex(null);
    selectSegment(lineId, additive);
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
          : drawMode === "region"
            ? "Region: drag around a folio or column · the segments inside join it"
            : selectedSegment
              ? selectedVertexIndex !== null
                ? `Segment ${selectedSegmentNumber} · vertex ${selectedVertexIndex + 1} selected · Delete removes point · Esc deselects`
                : `Segment ${selectedSegmentNumber} ${
                    segmentHasGroundTruth(selectedSegment)
                      ? "paired"
                      : "unpaired"
                  } · click edge to add · click handle to select · Esc deselects`
              : "Drag to pan · scroll to zoom · Page Up/Down for pages · select a segment to edit it";

  function pickDrawMode(nextMode: DrawTool) {
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
    setSelectedSegmentIds([]);
    setSelectedLineId(null);
    setSelectedRegionId(null);
    setApprovedTextDraft("");
    setStripDismissed(false);
  }, [
    setSelectedSegmentId,
    setSelectedSegmentIds,
    setSelectedLineId,
    setSelectedRegionId,
    setApprovedTextDraft,
  ]);

  // G arms the Region tool and T toggles the side-by-side transcript pane.
  // Kept here rather than in the shared shortcut hook, with the same rule
  // that hook applies: never while typing.
  const pickDrawModeRef = useRef(pickDrawMode);
  pickDrawModeRef.current = pickDrawMode;
  const canvasSettingsRef = useRef(canvasSettings);
  canvasSettingsRef.current = canvasSettings;
  // Set when the keyboard opens the pane, read once by the pane as it mounts
  // so the selected row takes focus.
  const transcriptFocusOnOpenRef = useRef(false);
  const toggleSideBySideRef = useRef<(focusRow?: boolean) => void>(() => {});
  toggleSideBySideRef.current = (focusRow = false) => {
    const current = canvasSettingsRef.current;
    transcriptFocusOnOpenRef.current = focusRow && !current.sideBySide;
    const next = { ...current, sideBySide: !current.sideBySide };
    setCanvasSettings(next);
    savePageEditorSettings(next);
  };
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (key === "g") {
        event.preventDefault();
        pickDrawModeRef.current("region");
        return;
      }
      if (key === "t") {
        event.preventDefault();
        toggleSideBySideRef.current(true);
      }
    }
    // Enter on the selected canvas segment goes to its transcript row,
    // opening the pane first. Captured: the segment stops Enter itself.
    function onEnterCapture(event: KeyboardEvent) {
      if (event.key !== "Enter" || event.ctrlKey || event.metaKey) return;
      if (event.altKey || event.shiftKey) return;
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (!target.matches('.pe-segment-shape[aria-current="true"]')) return;
      if (!canvasSettingsRef.current.sideBySide) {
        toggleSideBySideRef.current(true);
        return;
      }
      const lineId = selectedSegmentIdRef.current;
      window.document
        .querySelector<HTMLTextAreaElement>(
          `.pe-transcript-row[data-line-id="${lineId}"] textarea`,
        )
        ?.focus();
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keydown", onEnterCapture, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keydown", onEnterCapture, true);
    };
  }, []);

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
    onMerge:
      pairing.selectedSegmentIds.length >= 2
        ? mergeSelectedSegments
        : undefined,
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
        selectedSegmentIds={pairing.selectedSegmentIds}
        pairedSegmentIds={pairedIds}
        mergeBusy={segmentMerge.merging}
        mergeError={segmentMerge.mergeError}
        onMergeSelected={mergeSelectedSegments}
        onClearMergeSelection={() => {
          pairing.setSelectedSegmentIds([]);
          segmentMerge.setMergeError(null);
        }}
        hoveredSegmentId={hoveredSegmentId}
        onHoverSegment={setHoveredSegmentId}
        focusRequest={focusRequest}
        settings={canvasSettings}
        drawingRectangle={drawMode === "rectangle"}
        drawingRegion={drawMode === "region"}
        drawingPolygon={drawMode === "polygon"}
        draftStart={draftStart}
        draftPolygon={draftPolygon}
        onDraftStart={setDraftStart}
        onRectangleDrawn={async (end) => {
          if (!draftStart) return;
          const rectangle = rectanglePoints(draftStart, end);
          if (drawMode === "region") {
            setDraftStart(null);
            setDrawMode("none");
            setSelectedSegmentId(null);
            await regions.createRegion(rectangle);
            return;
          }
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
        selectedRegionId={selectedRegionId}
        onSelectRegion={(regionId) => {
          setSelectedSegmentId(null);
          setSelectedLineId(null);
          setSelectedVertexIndex(null);
          setSelectedRegionId(regionId);
        }}
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
      <PageEditorRegionPanel
        regions={layout.blocks}
        selectedRegionId={selectedRegionId}
        selectedSegment={selectedSegment ?? null}
        selectedSegmentNumber={selectedSegmentNumber}
        onRename={(regionId, label) =>
          void regions.renameRegion(regionId, label)
        }
        onDelete={(regionId) => void regions.deleteRegion(regionId)}
        onAssign={(segmentId, regionId) =>
          void regions.assignSegmentToRegion(segmentId, regionId)
        }
        onClose={() => setSelectedRegionId(null)}
        error={regions.regionError}
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
            transcriptOpen={canvasSettings.sideBySide}
            onToggleTranscript={() => toggleSideBySideRef.current()}
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
              {/* Always the split pane, collapsed while the transcript is
                  closed: the canvas stays mounted, so T keeps zoom and pan. */}
              <PageEditorSplitPane
                collapsed={!canvasSettings.sideBySide}
                ratio={canvasSettings.splitRatio}
                onRatioChange={(ratio) =>
                  handleCanvasSettingsChange({
                    ...canvasSettings,
                    splitRatio: ratio,
                  })
                }
                left={canvasPaneContent}
                right={
                  canvasSettings.sideBySide && (
                    <PageEditorTranscriptEditor
                      lines={lines}
                      regions={layout.blocks}
                      selectedSegmentId={selectedSegmentId}
                      hoveredSegmentId={hoveredSegmentId}
                      preferredLayerId={preferredModelLayerId}
                      onSelectSegment={handleSelectSegment}
                      onHoverSegment={setHoveredSegmentId}
                      onFocusSegment={(id) =>
                        setFocusRequest({ segmentId: id, nonce: Date.now() })
                      }
                      onCommitText={commitLineText}
                      onReloadLine={reloadLineText}
                      onReturnToCanvas={() =>
                        window.document
                          .querySelector<SVGElement>(
                            '.pe-segment-shape[aria-current="true"]',
                          )
                          ?.focus()
                      }
                      takeFocusOnOpen={() => {
                        const focusRow = transcriptFocusOnOpenRef.current;
                        transcriptFocusOnOpenRef.current = false;
                        return focusRow;
                      }}
                      scaleFontToMask={canvasSettings.transcriptFontFromMask}
                      onClose={() =>
                        handleCanvasSettingsChange({
                          ...canvasSettings,
                          sideBySide: false,
                        })
                      }
                    />
                  )
                }
              />
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

          <PageEditorMergeConflictDialog
            open={segmentMerge.conflict !== null}
            lineIds={segmentMerge.conflict?.lineIds ?? []}
            lines={lines}
            busy={segmentMerge.merging}
            onCancel={segmentMerge.dismissConflict}
            onKeep={segmentMerge.resolveConflict}
          />
        </div>
      )}
    </PageEditorShell>
  );
}
