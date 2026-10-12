import { useCallback, useEffect, useRef, useState } from "react";

const MIN_HEIGHT = 128;
const DEFAULT_HEIGHT = 220;
const MAX_VH = 0.45;

/**
 * ``visible`` is whether the strip is on screen. The strip component calls
 * this hook even while it renders nothing, so the hook alone decides the
 * published inset: the height while visible, 0 otherwise.
 */
export function useStripResize(initialHeight = DEFAULT_HEIGHT, visible = true) {
  const [height, setHeight] = useState(initialHeight);
  const draggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHeightRef = useRef(initialHeight);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      draggingRef.current = true;
      startYRef.current = event.clientY;
      startHeightRef.current = height;
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.classList.add("is-dragging");
    },
    [height],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const delta = startYRef.current - event.clientY;
      const maxHeight = window.innerHeight * MAX_VH;
      setHeight(
        Math.min(
          maxHeight,
          Math.max(MIN_HEIGHT, startHeightRef.current + delta),
        ),
      );
    },
    [],
  );

  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      event.currentTarget.releasePointerCapture(event.pointerId);
      event.currentTarget.classList.remove("is-dragging");
    },
    [],
  );

  useEffect(() => {
    const onResize = () => {
      const maxHeight = window.innerHeight * MAX_VH;
      setHeight((h) => Math.min(h, maxHeight));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // The floating background-jobs panel clears the strip via this inset (see
  // .pe-job-panel in page-editor.css), following the resizer live. A hidden
  // strip publishes 0; unmounting removes it so other routes fall back to 0.
  useEffect(() => {
    document.documentElement.style.setProperty(
      "--nk-bottom-inset",
      `${visible ? height : 0}px`,
    );
  }, [height, visible]);
  useEffect(
    () => () => {
      document.documentElement.style.removeProperty("--nk-bottom-inset");
    },
    [],
  );

  return { height, setHeight, onPointerDown, onPointerMove, onPointerUp };
}
