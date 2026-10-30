import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from "react";
import type { Region } from "../../types";
import { PublicZoomSurface } from "./PublicZoomSurface";

type PublicPageCanvasProps = {
  imageUrl: string;
  layoutWidth: number;
  layoutHeight: number;
  regions: Region[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
};

type Size = { width: number; height: number };

function useRenderedImage(
  imageUrl: string,
  layoutWidth: number,
  layoutHeight: number,
): {
  imageRef: React.RefObject<HTMLImageElement | null>;
  displaySize: Size | null;
  coordSize: Size;
  imageLoaded: boolean;
  imageFailed: boolean;
  syncDisplaySize: () => void;
  handleImageLoad: () => void;
  handleImageError: () => void;
} {
  const imageRef = useRef<HTMLImageElement>(null);
  const [displaySize, setDisplaySize] = useState<Size | null>(null);
  const [coordSize, setCoordSize] = useState<Size>({
    width: layoutWidth,
    height: layoutHeight,
  });
  // Gates the overlay on the actual decoded image, not just a pre-decode
  // resize event, so polygons never get scaled against a transient box.
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  /**
   * Reset during render, not in an effect.
   *
   * Switching pages hands this component a new `imageUrl` and the new page's
   * `regions` in one commit, while `imageLoaded`, `displaySize` and
   * `coordSize` still describe the page being left. An effect runs after the
   * commit, so the browser can paint the new page's polygons scaled against
   * the old page's box first: the same wrong-box defect this component exists
   * to prevent, moved from first load to navigation. Adjusting state during
   * render re-runs this component before anything is committed, so that frame
   * never exists. (useLayoutEffect would also close it, but this page is
   * server-rendered and it would warn there.)
   */
  const renderedUrlRef = useRef(imageUrl);
  if (renderedUrlRef.current !== imageUrl) {
    renderedUrlRef.current = imageUrl;
    setImageLoaded(false);
    setImageFailed(false);
    setDisplaySize(null);
    // coordSize has to go back to the props too. Left alone it would keep the
    // previous page's natural size, which is the wrong viewBox for the new one.
    setCoordSize({ width: layoutWidth, height: layoutHeight });
  }

  const syncDisplaySize = useCallback(() => {
    const image = imageRef.current;
    if (!image) return;
    const width = image.clientWidth;
    const height = image.clientHeight;
    if (width > 0 && height > 0) {
      setDisplaySize({ width, height });
    }
  }, []);

  useEffect(() => {
    const image = imageRef.current;
    if (!image || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(() => {
      syncDisplaySize();
    });
    observer.observe(image);
    return () => observer.disconnect();
  }, [syncDisplaySize, imageUrl]);

  const handleImageLoad = () => {
    const image = imageRef.current;
    if (!image) return;
    const { naturalWidth, naturalHeight } = image;
    if (naturalWidth > 0 && naturalHeight > 0) {
      setCoordSize({ width: naturalWidth, height: naturalHeight });
    } else if (layoutWidth > 0 && layoutHeight > 0) {
      setCoordSize({ width: layoutWidth, height: layoutHeight });
    }
    syncDisplaySize();
    setImageLoaded(true);
  };

  const handleImageError = () => {
    setImageFailed(true);
  };

  return {
    imageRef,
    displaySize,
    coordSize,
    imageLoaded,
    imageFailed,
    syncDisplaySize,
    handleImageLoad,
    handleImageError,
  };
}

function overlayReady(
  imageLoaded: boolean,
  displaySize: Size | null,
  coordSize: Size,
): displaySize is Size {
  return (
    imageLoaded &&
    displaySize !== null &&
    coordSize.width > 0 &&
    coordSize.height > 0
  );
}

function frameClassName(imageLoaded: boolean, imageFailed: boolean): string {
  return `public-page-canvas__frame${
    !imageLoaded && !imageFailed ? " public-page-canvas__frame--loading" : ""
  }`;
}

export function PublicPageCanvas({
  imageUrl,
  layoutWidth,
  layoutHeight,
  regions,
  selectedRegionId,
  onSelectRegion,
}: PublicPageCanvasProps) {
  const {
    imageRef,
    displaySize,
    coordSize,
    imageLoaded,
    imageFailed,
    handleImageLoad,
    handleImageError,
  } = useRenderedImage(imageUrl, layoutWidth, layoutHeight);

  const selectWithKeyboard = (
    event: KeyboardEvent<SVGPolygonElement>,
    regionId: number,
    selected: boolean,
  ) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onSelectRegion(selected ? null : regionId);
  };

  // Reserves the image's true box before it arrives, so the page does not
  // jump when the browser finishes decoding it. Falls back to no
  // aspect-ratio when the layout dimensions are missing or zero.
  const imageStyle: CSSProperties | undefined =
    layoutWidth > 0 && layoutHeight > 0
      ? { aspectRatio: `${layoutWidth} / ${layoutHeight}` }
      : undefined;

  return (
    <PublicZoomSurface ariaLabel="Manuscript page viewer">
      <div className={frameClassName(imageLoaded, imageFailed)}>
        {imageFailed ? (
          <p className="public-page-canvas__error">
            This page image could not be loaded.
          </p>
        ) : (
          <>
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Manuscript page"
              draggable={false}
              onLoad={handleImageLoad}
              onError={handleImageError}
              className="public-page-canvas__image"
              style={imageStyle}
            />
            {overlayReady(imageLoaded, displaySize, coordSize) && (
              <svg
                className="public-page-canvas__overlay"
                viewBox={`0 0 ${coordSize.width} ${coordSize.height}`}
                preserveAspectRatio="none"
                style={{
                  width: displaySize.width,
                  height: displaySize.height,
                }}
                aria-hidden={regions.length === 0}
                role="group"
                aria-label="Selectable transcription lines"
              >
                {regions.map((region) => {
                  const isSelected = region.id === selectedRegionId;
                  const points = region.boundary
                    .map(([x, y]) => `${x},${y}`)
                    .join(" ");
                  return (
                    <polygon
                      key={region.id}
                      role="button"
                      tabIndex={0}
                      points={points}
                      aria-label={`Line ${region.id}`}
                      aria-pressed={isSelected}
                      fill={
                        isSelected
                          ? "rgba(13, 31, 60, 0.18)"
                          : "rgba(82, 196, 26, 0.15)"
                      }
                      stroke={isSelected ? "var(--navy, #0d1f3c)" : "#52c41a"}
                      strokeWidth={isSelected ? 2.5 : 2}
                      style={{ pointerEvents: "all", cursor: "pointer" }}
                      onClick={() =>
                        onSelectRegion(isSelected ? null : region.id)
                      }
                      onKeyDown={(event) =>
                        selectWithKeyboard(event, region.id, isSelected)
                      }
                    />
                  );
                })}
              </svg>
            )}
          </>
        )}
      </div>
    </PublicZoomSurface>
  );
}
