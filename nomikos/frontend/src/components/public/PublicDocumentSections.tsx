import type { ReactNode } from "react";
import Link from "next/link";
import { WorkflowBadge } from "../WorkflowBadge";
import { ContentRegionLoading } from "../layout/ContentRegionLoading";
import { PublicCanvasPdfView } from "./PublicCanvasPdfView";
import { PublicDocumentExports } from "./PublicDocumentExports";
import { PublicPageCanvas } from "./PublicPageCanvas";
import { PublicPartTabs } from "./PublicPartTabs";
import { PublicTranscriptPanel } from "./PublicTranscriptPanel";
import type { PublicDocumentModel } from "../../pages/usePublicDocument";

/**
 * The public reader's own nav, not `TopNav`: this page is anonymous-friendly
 * (it offers "Sign in" when logged out) and its logo goes home, so the
 * authenticated shell's nav does not apply. The brand block is shared copy,
 * not a shared component, for that reason.
 */
export function PublicReaderNav(model: PublicDocumentModel): ReactNode {
  const { isLoggedIn, projectId, documentId } = model;
  return (
    <nav className="topnav" aria-label="Main navigation">
      <Link href="/" className="topnav-logo" aria-label="nomikos home">
        <img src="/nomos.svg" alt="" />
        <span>nomikos</span>
      </Link>
      <div className="topnav-sep" aria-hidden="true" />
      <div className="topnav-breadcrumb">
        <span className="current" aria-current="page">
          Public view
        </span>
      </div>
      <div className="topnav-spacer" />
      <div className="topnav-actions">
        {isLoggedIn && projectId && documentId && (
          <Link
            href={`/projects/${projectId}/documents/${documentId}`}
            className="btn btn-outline btn-sm"
          >
            Editor
          </Link>
        )}
        {!isLoggedIn && (
          <Link href="/login" className="btn btn-ghost btn-sm">
            Sign in
          </Link>
        )}
      </div>
    </nav>
  );
}

export function PublicReaderHeader(model: PublicDocumentModel): ReactNode {
  const { document, parts } = model;
  return (
    <header className="pub-header pub-header--compact">
      <div className="pub-header__main">
        <div className="pub-header__title-row">
          <h1>{document?.name ?? "Document"}</h1>
          {document && <WorkflowBadge workflow={document.workflow} />}
        </div>
        {document && (
          <p className="pub-header__meta">
            {parts.length} page{parts.length === 1 ? "" : "s"}
          </p>
        )}
      </div>
    </header>
  );
}

function PageViewToggle(model: PublicDocumentModel): ReactNode {
  const { canvasView, setCanvasView } = model;
  return (
    <div className="pub-segment" role="tablist" aria-label="Page view">
      <button
        type="button"
        role="tab"
        className={`pub-segment__btn${canvasView === "image" ? " pub-segment__btn--active" : ""}`}
        aria-selected={canvasView === "image"}
        onClick={() => setCanvasView("image")}
      >
        Image
      </button>
      <button
        type="button"
        role="tab"
        className={`pub-segment__btn${canvasView === "pdf" ? " pub-segment__btn--active" : ""}`}
        aria-selected={canvasView === "pdf"}
        onClick={() => setCanvasView("pdf")}
      >
        PDF
      </button>
    </div>
  );
}

function WorkspaceExports(model: PublicDocumentModel): ReactNode {
  const {
    projectId,
    documentId,
    document,
    activePart,
    activePartIndex,
    token,
  } = model;
  if (!projectId || !documentId || !activePart) return null;
  return (
    <PublicDocumentExports
      projectId={projectId}
      documentId={documentId}
      documentName={document?.name ?? ""}
      partId={activePart.id}
      partIndex={activePartIndex}
      token={token}
    />
  );
}

function PublicWorkspaceToolbar(model: PublicDocumentModel): ReactNode {
  const { partTabs, activePart, selectPart } = model;
  return (
    <div className="pub-workspace__toolbar">
      <PublicPartTabs
        parts={partTabs}
        activeId={activePart?.id ?? null}
        onChange={selectPart}
        variant="workspace"
      />

      <div className="pub-workspace__tools">
        <PageViewToggle {...model} />
        <WorkspaceExports {...model} />
      </div>
    </div>
  );
}

function PublicCanvasPane(model: PublicDocumentModel): ReactNode {
  const {
    canvasView,
    projectId,
    documentId,
    activePart,
    activePartIndex,
    token,
    imageUrl,
    imageWidth,
    imageHeight,
    regions,
    selectedRegionId,
    selectRegion,
  } = model;
  if (canvasView === "pdf" && projectId && documentId && activePart) {
    return (
      <PublicCanvasPdfView
        projectId={projectId}
        documentId={documentId}
        partId={activePart.id}
        token={token}
        downloadFilename={`page-${activePartIndex}.pdf`}
      />
    );
  }
  if (imageUrl && imageWidth > 0) {
    return (
      <PublicPageCanvas
        imageUrl={imageUrl}
        layoutWidth={imageWidth}
        layoutHeight={imageHeight}
        regions={regions}
        selectedRegionId={selectedRegionId}
        onSelectRegion={selectRegion}
      />
    );
  }
  return (
    <div className="pub-canvas__empty">
      <p>No page image available</p>
    </div>
  );
}

function PublicTranscriptPane(model: PublicDocumentModel): ReactNode {
  const { activePart, layout, selectedLineIndex, setSelectedLineIndex } = model;
  if (!activePart) return null;
  return (
    <PublicTranscriptPanel
      partId={activePart.id}
      layout={layout}
      selectedLineIndex={selectedLineIndex}
      onSelectLine={setSelectedLineIndex}
    />
  );
}

export function PublicWorkspace(model: PublicDocumentModel): ReactNode {
  return (
    <>
      <PublicWorkspaceToolbar {...model} />

      <div className="pub-split">
        {/*
          No `role="img"` here. It made every descendant presentational, which
          dropped the PDF pane's open/download links and the zoom controls out
          of the accessibility tree entirely - and this container is not an
          image in two of its three branches. The labelling it provided was
          redundant anyway: PublicPageCanvas already names itself
          ("Manuscript page viewer", and `alt` on the image), and
          PublicCanvasPdfView names its own iframe.
        */}
        <div className="pub-canvas">
          <PublicCanvasPane {...model} />
        </div>

        <PublicTranscriptPane {...model} />
      </div>

      {model.parts.length === 0 && (
        <p className="list-hint">
          This published document has no page images yet.
        </p>
      )}
    </>
  );
}

export function PublicDocumentContent(model: PublicDocumentModel): ReactNode {
  const { notFound, loading, errorMessage } = model;
  if (notFound) {
    return (
      <div className="notice-banner" role="alert">
        <strong>Document not available</strong>
        This document is not published or does not exist.
      </div>
    );
  }
  if (loading) {
    return <ContentRegionLoading label="Loading document" />;
  }
  if (errorMessage) {
    return (
      <div className="notice-banner" role="alert">
        <strong>Could not load document</strong>
        {errorMessage}
      </div>
    );
  }
  return <PublicWorkspace {...model} />;
}
