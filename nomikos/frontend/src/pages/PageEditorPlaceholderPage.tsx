import { useCallback, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import type {
  DocumentPartResponse,
  DocumentWithPartsResponse,
  DocumentWorkflow,
} from "../api/client";
import { invalidateAfter } from "../api/resources";
import { useHostPreference } from "../inference";
import { PageEditorCanvas } from "../components/page-editor/PageEditorCanvas";
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
import {
  useDocumentPaging,
  useLayoutMutations,
  usePageEditorData,
  usePageEditorJobQueue,
  usePageEditorRunState,
  usePageTurns,
  usePairingState,
  useSegmentHealth,
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

  // The page the editor is on. The route seeds it and mirrors it back, but the
  // editor turns pages by changing this rather than by re-entering the route,
  // so the toolbar, the page rail and the focused control survive a page turn.
  const { activePartId: partId, goToPart } = useDocumentPaging({
    projectId,
    documentId,
    routePartId,
  });

  const editorData = usePageEditorData(projectId, documentId, partId, () => {
    draw.resetDrawing();
    setStripDismissed(false);
    setSelectedVertexIndex(null);
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

  const pageTurns = usePageTurns(parts, partId, goToPart);

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
    onDrawComplete: () => draw.setDrawMode("none"),
    setSubmissionRefusal,
    trackJobAndWait: jobQueue.trackAndWait,
    subscribeToJobCompletion: jobQueue.subscribeToJobCompletion,
  });

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
    resetTools,
  });

  const pairedIds = useMemo(() => segmentIdsWithGroundTruth(lines), [lines]);
  const stripVisible = Boolean(selectedSegment) && !stripDismissed;
  const runState = usePageEditorRunState({ segmenting, ocrRunning, ocrScope });

  function pickDrawMode(nextMode: "rectangle" | "polygon") {
    draw.toggleDrawMode(nextMode);
    setActionsOpen(false);
  }

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
            processingKind={runState.processingKind}
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
              <PageEditorCanvas
                imageUrl={loaded.part.image_url}
                imageAlt={`Page ${partIndex}`}
                imageWidth={loaded.part.width ?? 640}
                imageHeight={loaded.part.height ?? 900}
                layout={layout}
                lines={lines}
                selectedSegmentId={selectedSegmentId}
                pairedSegmentIds={pairedIds}
                settings={canvasSettings}
                drawingRectangle={draw.drawMode === "rectangle"}
                drawingPolygon={draw.drawMode === "polygon"}
                draftStart={draw.draftStart}
                draftPolygon={draw.draftPolygon}
                onDraftStart={draw.setDraftStart}
                onRectangleDrawn={(end) =>
                  draw.finishRectangle(end, replaceWithManualLine)
                }
                onPolygonPoint={draw.addPolygonPoint}
                onPolygonComplete={completeDraftPolygon}
                onSelectLine={selection.selectLine}
                onSelectSegment={selection.selectSegment}
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
              <PageEditorCanvasHint
                hint={canvasHint}
                processing={Boolean(runState.processingKind)}
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
        </div>
      )}
    </PageEditorShell>
  );
}
