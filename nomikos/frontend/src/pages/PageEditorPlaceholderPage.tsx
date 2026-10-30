import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import type {
  DocumentPartResponse,
  LayoutPoint,
  DocumentWithPartsResponse,
  DocumentWorkflow,
} from "../api/client";
import { invalidateAfter } from "../api/resources";
import { useHostPreference } from "../inference";
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
import { PageEditorPageRailSlot } from "../components/page-editor/PageEditorPageRailSlot";
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
import { PageEditorTranscriptionPdfSlot } from "../components/page-editor/PageEditorTranscriptionPdfWrap";
import { PageEditorBaselineBar } from "../components/page-editor/PageEditorBaselineBar";
import { PageEditorCanvasHint } from "../components/page-editor/PageEditorCanvasHint";
import { canvasHintFor } from "../components/page-editor/canvasHint";
import { rectanglePoints } from "../components/page-editor/canvasGeometry";
import {
  useDocumentPaging,
  useLayoutMutations,
  usePageEditorData,
  usePageEditorJobQueue,
  usePageEditorRunState,
  usePageTurns,
  usePairingState,
  useRegions,
  useSegmentHealth,
  useSegmentMerge,
} from "../components/page-editor/hooks";
import { useEditorDrawState } from "../components/page-editor/hooks/useEditorDrawState";
import { useEditorSelection } from "../components/page-editor/hooks/useEditorSelection";
import { useEditorShortcuts } from "../components/page-editor/hooks/useEditorShortcuts";
import { segmentIdsWithGroundTruth } from "../components/page-editor/hooks/utils";

function documentHrefFor(
  projectId: string | undefined,
  documentId: string | undefined,
): string {
  return projectId && documentId
    ? `/projects/${projectId}/documents/${documentId}`
    : "/projects";
}

/** The document and page together, once both have loaded. */
function loadedPage(
  document: DocumentWithPartsResponse | null,
  part: DocumentPartResponse | null,
) {
  return document && part ? { document, part } : null;
}

function unavailableDescriptionFor(
  error: string | null,
  document: DocumentWithPartsResponse | null,
  part: DocumentPartResponse | null,
): string | null {
  if (error) return error;
  return loadedPage(document, part)
    ? null
    : "This document part was not found.";
}

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

  const draw = useEditorDrawState();
  const [actionsOpen, setActionsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [segmentHealthOpen, setSegmentHealthOpen] = useState(false);
  const [canvasSettings, setCanvasSettings] = useState(loadPageEditorSettings);
  const [transcriptionPdfOpen, setTranscriptionPdfOpen] = useState(false);
  const [transcriptionPdfRefreshKey, setTranscriptionPdfRefreshKey] =
    useState(0);
  const [stripDismissed, setStripDismissed] = useState(false);
  const [selectedVertexIndex, setSelectedVertexIndex] = useState<number | null>(
    null,
  );
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
    draw.resetDrawing();
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

  const pageTurns = usePageTurns(parts, partId, goToPart);

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
    onDrawComplete: () => draw.setDrawMode("none"),
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
    onMerged: (survivor) => {
      pairing.selectMergedSurvivor(survivor);
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

  const { selectedLineId, segmenting, replaceWithManualLine } = layoutMutations;
  const {
    selectedSegmentId,
    ocrRunning,
    ocrScope,
    selectedSegment,
    selectedSegmentNumber,
  } = pairing;

  const { resetDrawing } = draw;
  const resetTools = useCallback(() => {
    resetDrawing();
    setActionsOpen(false);
  }, [resetDrawing]);

  const selection = useEditorSelection({
    lines,
    layoutLines: layout.lines,
    pairing,
    mutations: layoutMutations,
    selectedVertexIndex,
    setSelectedVertexIndex,
    setStripDismissed,
    setSelectedRegionId,
    resetTools,
  });

  const pairedIds = useMemo(() => segmentIdsWithGroundTruth(lines), [lines]);
  const stripVisible =
    Boolean(selectedSegment) && !stripDismissed && !canvasSettings.sideBySide;
  const runState = usePageEditorRunState({ segmenting, ocrRunning, ocrScope });

  function pickDrawMode(nextMode: DrawTool) {
    draw.toggleDrawMode(nextMode);
    setActionsOpen(false);
  }

  // G arms the Region tool and T toggles the side-by-side transcript pane.
  // Kept here rather than in the shared shortcut hook, with the same rule
  // that hook applies: never while typing.
  const pickDrawModeRef = useRef(pickDrawMode);
  pickDrawModeRef.current = pickDrawMode;
  const canvasSettingsRef = useRef(canvasSettings);
  canvasSettingsRef.current = canvasSettings;
  const selectedSegmentIdRef = useRef(selectedSegmentId);
  selectedSegmentIdRef.current = selectedSegmentId;
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

  const completeDraftPolygon = () =>
    draw.completeDraftPolygon(replaceWithManualLine);

  useEditorShortcuts({
    selectedSegmentId,
    selectedLineId,
    selectedVertexIndex,
    onPickDrawMode: pickDrawMode,
    onRemoveSelectedVertex: selection.removeSelectedVertex,
    onDeleteSelection: selection.deleteSelection,
    onEscape: selection.clearSelection,
    onMerge:
      pairing.selectedSegmentIds.length >= 2
        ? mergeSelectedSegments
        : undefined,
    onUndo: () => void layoutMutations.undoEdit(),
    onRedo: () => void layoutMutations.redoEdit(),
    canCompletePolygon: draw.canCompletePolygon,
    onCompletePolygon: completeDraftPolygon,
    onPreviousPage: pageTurns.turnBack,
    onNextPage: pageTurns.turnForward,
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
    saveMessage: layoutMutations.saveMessage,
    transcriptionSaveMessage: pairing.transcriptionSaveMessage,
    ocrMessage: pairing.ocrMessage,
    segmentMessage: layoutMutations.segmentMessage,
    mutationError: layoutMutations.mutationError,
    pairingError,
    layoutError,
    lineError,
  };

  function handleDocumentWorkflowChange(workflow: DocumentWorkflow) {
    setDocument((current) => (current ? { ...current, workflow } : current));
    // Publishing is the one write in this editor that changes what a reader
    // can reach at all, so the local copy above isn't enough: without this,
    // the document list, detail page and public page would keep showing the
    // old status.
    if (projectId && documentId) {
      invalidateAfter.documentUpdated(projectId, documentId);
    }
  }

  const loaded = loadedPage(document, part);
  const canvasHint = canvasHintFor({
    partLoading,
    partIndex,
    processingLabel: runState.processingLabel,
    drawMode: draw.drawMode,
    draftPointCount: draw.draftPolygon.length,
    selectedSegment,
    selectedSegmentNumber,
    selectedVertexIndex,
  });

  const preferredModelLayerId =
    pairing.selectedTranscriptionLayer?.kind === "model"
      ? pairing.selectedTranscriptionLayer.id
      : null;

  async function handleRectangleDrawn(end: LayoutPoint) {
    if (draw.drawMode === "region") {
      if (!draw.draftStart) return;
      const rectangle = rectanglePoints(draw.draftStart, end);
      draw.resetDrawing();
      pairing.setSelectedSegmentId(null);
      await regions.createRegion(rectangle);
      return;
    }
    await draw.finishRectangle(end, replaceWithManualLine);
  }

  const canvasPaneContent = (page: DocumentPartResponse) => (
    <>
      <PageEditorCanvas
        imageUrl={page.image_url}
        imageAlt={`Page ${partIndex}`}
        imageWidth={page.width ?? 640}
        imageHeight={page.height ?? 900}
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
        drawingRectangle={draw.drawMode === "rectangle"}
        drawingRegion={draw.drawMode === "region"}
        drawingPolygon={draw.drawMode === "polygon"}
        draftStart={draw.draftStart}
        draftPolygon={draw.draftPolygon}
        onDraftStart={draw.setDraftStart}
        onRectangleDrawn={handleRectangleDrawn}
        onPolygonPoint={draw.addPolygonPoint}
        onPolygonComplete={completeDraftPolygon}
        onSelectLine={selection.selectLine}
        onSelectSegment={selection.selectSegment}
        selectedRegionId={selectedRegionId}
        onSelectRegion={selection.selectRegion}
        segmentVertexEditEnabled={
          draw.drawMode === "none" && Boolean(selectedSegmentId)
        }
        onSelectTool={selection.clearSelection}
        onPickDrawMode={pickDrawMode}
        canDelete={selection.hasSelection}
        onDeleteSelected={selection.deleteSelection}
        selectedVertexIndex={selectedVertexIndex}
        onSelectedVertexChange={setSelectedVertexIndex}
        commitSignal={selection.vertexCommitSignal}
        onSegmentPointsChange={layoutMutations.updateSegmentPoints}
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
      <PageEditorCanvasHint
        hint={canvasHint}
        processing={Boolean(runState.processingKind)}
      />
    </>
  );

  return (
    <PageEditorShell
      loading={loading}
      backHref={documentHrefFor(projectId, documentId)}
      unavailableDescription={unavailableDescriptionFor(error, document, part)}
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
        loaded ? (
          <PageEditorToolbar
            projectId={projectId}
            documentId={documentId}
            partId={loaded.part.id}
            document={loaded.document}
            partIndex={partIndex ?? 1}
            pageCount={parts.length}
            hasPreviousPart={pageTurns.hasPreviousPage}
            hasNextPart={pageTurns.hasNextPage}
            onPreviousPart={pageTurns.goToPreviousPage}
            onNextPart={pageTurns.goToNextPage}
            lines={lines}
            pairingProgress={pairingProgress}
            selectedSegmentId={selectedSegmentId}
            textLines={textLines}
            onPairTextLine={pairing.pairTextLine}
            onDocumentWorkflowChange={handleDocumentWorkflowChange}
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
            onRunAutoSegment={layoutMutations.runAutoSegment}
            onRunSegmentOcr={pairing.runSegmentOcr}
            onRunPageOcr={pairing.runPageOcr}
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
      {loaded && (
        <div className="pe-workspace">
          <div className="pe-body">
            <PageEditorPageRailSlot
              parts={parts}
              activePartId={partId}
              onSelectPart={goToPart}
            />
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
                left={canvasPaneContent(loaded.part)}
                right={
                  canvasSettings.sideBySide && (
                    <PageEditorTranscriptEditor
                      lines={lines}
                      regions={layout.blocks}
                      selectedSegmentId={selectedSegmentId}
                      hoveredSegmentId={hoveredSegmentId}
                      preferredLayerId={preferredModelLayerId}
                      onSelectSegment={selection.selectSegment}
                      onHoverSegment={setHoveredSegmentId}
                      onFocusSegment={(id) =>
                        setFocusRequest({ segmentId: id, nonce: Date.now() })
                      }
                      onCommitText={pairing.commitLineText}
                      onReloadLine={pairing.reloadLineText}
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
            <PageEditorTranscriptionPdfSlot
              open={transcriptionPdfOpen}
              projectId={projectId}
              documentId={documentId}
              partId={partId}
              downloadFilename={`${loaded.document.name.replace(/\s+/g, "_")}_page_${partIndex}_transcription.pdf`}
              refreshKey={transcriptionPdfRefreshKey}
              onClose={() => setTranscriptionPdfOpen(false)}
              onRefresh={refreshTranscriptionPdf}
            />
          </div>

          {selectedLineId && (
            <PageEditorBaselineBar
              onMoveDown={() => layoutMutations.moveSelectedBaseline(5)}
              onSave={() => void layoutMutations.saveSelectedLine()}
              onReset={() => void layoutMutations.resetSelectedLine()}
            />
          )}

          <PageEditorTranscriptionStrip
            visible={stripVisible}
            transcriptionLayers={transcriptionLayers}
            selectedTranscriptionLayerId={selectedTranscriptionLayerId}
            onSelectTranscriptionLayer={pairing.selectTranscriptionLayer}
            selectedSegmentNumber={selectedSegmentNumber}
            selectedSegment={selectedSegment}
            selectedTranscriptionLayer={pairing.selectedTranscriptionLayer}
            approvedTextDraft={pairing.approvedTextDraft}
            onApprovedTextDraftChange={pairing.setApprovedTextDraft}
            onSaveGroundTruthText={() =>
              selection.dismissAfterSave(pairing.saveGroundTruthText)
            }
            onPromoteSelectedSegmentToGroundTruth={
              pairing.promoteSelectedSegmentToGroundTruth
            }
            onRunSegmentOcr={pairing.runSegmentOcr}
            onNavigateSegment={pairing.navigateSegment}
            onDismiss={() => setStripDismissed(true)}
            lines={lines}
            selectedSegmentId={selectedSegmentId}
            onSaveApprovedText={() =>
              selection.dismissAfterSave(pairing.saveApprovedText)
            }
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
