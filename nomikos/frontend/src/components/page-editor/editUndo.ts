import type { LinePoint, LineResponse } from "../../api/client";

/** One in-session canvas step for Edit undo / Edit redo (not Annotation history). */
export type CanvasEdit =
  | {
      kind: "points";
      segmentId: string;
      before: LinePoint[];
      after: LinePoint[];
    }
  | {
      kind: "create";
      line: LineResponse;
    }
  | {
      kind: "delete";
      line: LineResponse;
    };

export function applyCanvasEdit(
  lines: LineResponse[],
  edit: CanvasEdit,
): LineResponse[] {
  switch (edit.kind) {
    case "points":
      return withPoints(lines, edit.segmentId, edit.after);
    case "create":
      return mergeLine(lines, edit.line);
    case "delete":
      return lines.filter((line) => line.id !== edit.line.id);
  }
}

export function applyCanvasEditInverse(
  lines: LineResponse[],
  edit: CanvasEdit,
): LineResponse[] {
  switch (edit.kind) {
    case "points":
      return withPoints(lines, edit.segmentId, edit.before);
    case "create":
      return lines.filter((line) => line.id !== edit.line.id);
    case "delete":
      return mergeLine(lines, edit.line);
  }
}

/** A points edit marks the Segment manual, whichever way it is replayed. */
function withPoints(
  lines: LineResponse[],
  segmentId: string,
  points: LinePoint[],
): LineResponse[] {
  return lines.map((line) =>
    line.id === segmentId
      ? { ...line, points, source: "manual" as const }
      : line,
  );
}

function mergeLine(lines: LineResponse[], saved: LineResponse): LineResponse[] {
  const without = lines.filter((line) => line.id !== saved.id);
  return [...without, saved].sort((a, b) => a.order - b.order);
}

export function pushEditOntoStack(
  stack: CanvasEdit[],
  edit: CanvasEdit,
  maxSize = 50,
): CanvasEdit[] {
  const next = [...stack, edit];
  if (next.length <= maxSize) return next;
  return next.slice(next.length - maxSize);
}
