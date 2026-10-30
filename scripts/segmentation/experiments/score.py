#!/usr/bin/env python
"""Score a directory of line-segmentation predictions against the Coptic
ground truth.

    score.py --pred-dir out/v0 --gt-dir <recon> --manifest scripts/coptic/manifest.json [--out results.csv]

The prediction format is one JSON per page:

    {"page": "<slug>", "variant": "<name>", "image_size": [w, h],
     "seconds": 1.23,
     "lines": [{"baseline": [[x, y], ...], "polygon": [[x, y], ...]}]}

with coordinates in SOURCE image pixels. Nothing here assumes the predictions
came from kraken, or from any particular model.

Metrics
-------
Matching follows the cBAD spirit. A prediction matches a ground-truth line when

  * their baselines overlap horizontally over at least half of the
    ground-truth baseline's x extent, and
  * the median vertical distance between the two baselines over that overlap is
    at most 0.25 x the local line pitch (the median baseline spacing of the
    ground-truth column the line sits in),

and matching is one to one (greedy, closest pair first).

Unpaired ground-truth lines are DON'T CARE: they are folio numbers, running
titles and the two halves of a transcription line the scribe wrote across two
rows. They take part in the matching, but a prediction that lands on one is
removed from the scoring entirely rather than counted as a false positive.

False positives decompose into

  * spurious: no relation at all to any ground-truth line,
  * duplicate: related to a ground-truth line that another prediction already
    took (the second half of a fragmented line).

Fragments and merges are counted on a looser relation graph: an edge joins a
prediction and a ground-truth line when the baselines are within 0.5 x pitch
and overlap over at least half of the SHORTER of the two extents, which catches
both a short fragment sitting on a long line and a long merged box spanning two
lines (including across the gutter).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gtlib  # noqa: E402

SAMPLES = 9


# --------------------------------------------------------------------------
# baseline geometry
# --------------------------------------------------------------------------
def extent(points: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    return min(xs), max(xs)


def y_at(polyline: list[list[float]], x: float) -> float:
    pts = sorted(polyline, key=lambda p: p[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2.0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def relation(a: list[list[float]], b: list[list[float]]) -> tuple[float, float] | None:
    """(median |dy|, overlap length) over the shared x range, or None."""
    ax0, ax1 = extent(a)
    bx0, bx1 = extent(b)
    lo, hi = max(ax0, bx0), min(ax1, bx1)
    if hi <= lo:
        return None
    step = (hi - lo) / (SAMPLES - 1)
    diffs = [abs(y_at(a, lo + i * step) - y_at(b, lo + i * step)) for i in range(SAMPLES)]
    return statistics.median(diffs), hi - lo


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass
class PageScore:
    slug: str
    manuscript: str
    width: int
    height: int
    scale: float
    pitch_src: float
    pitch_model: float
    lh_src: float
    lh_model: float
    seconds: float
    n_gt: int
    n_pred: int
    n_dontcare_pred: int
    tp: int
    fp: int
    fn: int
    spurious: int
    duplicate: int
    fragments: int
    merges: int
    columns: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_page(gt: gtlib.GTPage, pred: dict[str, Any]) -> PageScore:
    pred_lines = [ln["baseline"] for ln in pred["lines"] if len(ln.get("baseline", [])) >= 2]

    gt_lines = gt.lines
    pitches = [gt.pitch_for(ln.column) for ln in gt_lines]

    # --- relations -------------------------------------------------------
    strict: list[tuple[float, int, int]] = []  # (dy, pred, gt)
    loose: list[tuple[int, int]] = []
    for pi, pb in enumerate(pred_lines):
        px0, px1 = extent(pb)
        for gi, gl in enumerate(gt_lines):
            rel = relation(pb, gl.baseline)
            if rel is None:
                continue
            dy, ov = rel
            gx0, gx1 = extent(gl.baseline)
            pitch = pitches[gi]
            if ov >= 0.5 * max(gx1 - gx0, 1e-6) and dy <= 0.25 * pitch:
                strict.append((dy, pi, gi))
            shorter = min(gx1 - gx0, px1 - px0)
            if shorter > 0 and ov >= 0.5 * shorter and dy <= 0.5 * pitch:
                loose.append((pi, gi))

    # --- one-to-one strict matching -------------------------------------
    strict.sort()
    used_pred: dict[int, int] = {}
    used_gt: dict[int, int] = {}
    for _dy, pi, gi in strict:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred[pi] = gi
        used_gt[gi] = pi

    loose_by_pred: dict[int, set[int]] = {}
    loose_by_gt: dict[int, set[int]] = {}
    for pi, gi in loose:
        loose_by_pred.setdefault(pi, set()).add(gi)
        loose_by_gt.setdefault(gi, set()).add(pi)

    dontcare_gt = {gi for gi, gl in enumerate(gt_lines) if not gl.paired}

    # A prediction is DON'T CARE when it matched an unpaired ground-truth line,
    # or (unmatched) when the only lines it relates to are unpaired ones.
    dontcare_pred: set[int] = set()
    for pi in range(len(pred_lines)):
        matched = used_pred.get(pi)
        if matched is not None:
            if matched in dontcare_gt:
                dontcare_pred.add(pi)
            continue
        related = loose_by_pred.get(pi, set())
        if related and related <= dontcare_gt:
            dontcare_pred.add(pi)

    tp = sum(1 for pi, gi in used_pred.items() if gi not in dontcare_gt and pi not in dontcare_pred)
    fp_ids = [
        pi for pi in range(len(pred_lines)) if pi not in dontcare_pred and used_pred.get(pi) is None
    ]
    spurious = sum(1 for pi in fp_ids if not loose_by_pred.get(pi))
    duplicate = len(fp_ids) - spurious
    paired_gt = [gi for gi in range(len(gt_lines)) if gi not in dontcare_gt]
    fn = sum(1 for gi in paired_gt if gi not in used_gt)

    fragments = sum(1 for gi in paired_gt if len(loose_by_gt.get(gi, set()) - dontcare_pred) >= 2)
    merges = sum(
        1
        for pi in range(len(pred_lines))
        if pi not in dontcare_pred and len(loose_by_pred.get(pi, set()) - dontcare_gt) >= 2
    )

    # --- per column counts ----------------------------------------------
    counts = [0] * gt.ncols
    for pi, pb in enumerate(pred_lines):
        if pi in dontcare_pred:
            continue
        xs = [p[0] for p in pb]
        counts[gt.column_of_x(sum(xs) / len(xs))] += 1
    columns = []
    for c in range(gt.ncols):
        expected = gt.rtf_counts[c] if c < len(gt.rtf_counts) else 0
        columns.append((gt.column_names[c], counts[c], expected))

    pitch_src = statistics.median([p for p in gt.pitch_by_column if p > 0] or [0.0])
    lh_src = statistics.median([gtlib.line_height(ln.polygon) for ln in gt_lines] or [0.0])
    return PageScore(
        slug=gt.slug,
        manuscript=gt.manuscript,
        width=gt.width,
        height=gt.height,
        scale=gt.scale,
        pitch_src=pitch_src,
        pitch_model=pitch_src * gt.scale,
        lh_src=lh_src,
        lh_model=lh_src * gt.scale,
        seconds=float(pred.get("seconds") or 0.0),
        n_gt=len(paired_gt),
        n_pred=len(pred_lines),
        n_dontcare_pred=len(dontcare_pred),
        tp=tp,
        fp=len(fp_ids),
        fn=fn,
        spurious=spurious,
        duplicate=duplicate,
        fragments=fragments,
        merges=merges,
        columns=columns,
    )


def aggregate(scores: list[PageScore], label: str) -> dict[str, Any]:
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cols = [c for s in scores for c in s.columns]
    exact = sum(1 for _n, got, want in cols if got == want)
    diffs = [got - want for _n, got, want in cols]
    return {
        "group": label,
        "pages": len(scores),
        "gt_lines": tp + fn,
        "pred_lines": sum(s.n_pred for s in scores),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "spurious": sum(s.spurious for s in scores),
        "duplicate": sum(s.duplicate for s in scores),
        "fragments": sum(s.fragments for s in scores),
        "merges": sum(s.merges for s in scores),
        "columns": len(cols),
        "exact_columns": exact,
        "exact_share": round(exact / len(cols), 4) if cols else 0.0,
        "median_col_diff": statistics.median(diffs) if diffs else 0.0,
        "mean_abs_col_diff": round(statistics.fmean(abs(d) for d in diffs), 3) if diffs else 0.0,
        "median_pitch_model": round(statistics.median([s.pitch_model for s in scores]), 1)
        if scores
        else 0.0,
        "seconds_per_page": round(statistics.fmean([s.seconds for s in scores]), 2)
        if scores
        else 0.0,
        "total_seconds": round(sum(s.seconds for s in scores), 1),
    }


PER_PAGE_FIELDS = [
    "variant",
    "slug",
    "manuscript",
    "width",
    "height",
    "scale_to_1800",
    "pitch_src",
    "pitch_model",
    "line_height_src",
    "line_height_model",
    "seconds",
    "gt_lines",
    "pred_lines",
    "dontcare_pred",
    "tp",
    "fp",
    "fn",
    "spurious",
    "duplicate",
    "fragments",
    "merges",
    "precision",
    "recall",
    "f1",
    "columns",
    "exact_columns",
    "col_diffs",
]


def page_row(variant: str, s: PageScore) -> dict[str, Any]:
    return {
        "variant": variant,
        "slug": s.slug,
        "manuscript": s.manuscript,
        "width": s.width,
        "height": s.height,
        "scale_to_1800": round(s.scale, 4),
        "pitch_src": round(s.pitch_src, 1),
        "pitch_model": round(s.pitch_model, 1),
        "line_height_src": round(s.lh_src, 1),
        "line_height_model": round(s.lh_model, 1),
        "seconds": round(s.seconds, 2),
        "gt_lines": s.n_gt,
        "pred_lines": s.n_pred,
        "dontcare_pred": s.n_dontcare_pred,
        "tp": s.tp,
        "fp": s.fp,
        "fn": s.fn,
        "spurious": s.spurious,
        "duplicate": s.duplicate,
        "fragments": s.fragments,
        "merges": s.merges,
        "precision": round(s.precision, 4),
        "recall": round(s.recall, 4),
        "f1": round(s.f1, 4),
        "columns": len(s.columns),
        "exact_columns": sum(1 for _n, got, want in s.columns if got == want),
        "col_diffs": " ".join(f"{n}:{got - want:+d}" for n, got, want in s.columns),
    }


def run(
    *,
    pred_dir: Path,
    gt_dir: Path,
    manifest: Path,
    repo_root: Path,
    out: Path | None,
    quiet: bool = False,
    only: set[str] | None = None,
) -> list[dict[str, Any]]:
    pages = gtlib.load_all(gt_dir=gt_dir, manifest_path=manifest, repo_root=repo_root)
    if only:
        pages = [p for p in pages if p.slug in only]
    scores: list[PageScore] = []
    variant = pred_dir.name
    for gt in pages:
        path = pred_dir / f"{gt.slug}.json"
        if not path.exists():
            continue
        pred = json.loads(path.read_text())
        variant = pred.get("variant") or variant
        scores.append(score_page(gt, pred))
    if not scores:
        raise SystemExit(f"no predictions found in {pred_dir}")

    rows = [page_row(variant, s) for s in scores]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_PAGE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    summaries = [
        aggregate([s for s in scores if s.manuscript == ms], ms)
        for ms in ("vatican", "napoli")
        if any(s.manuscript == ms for s in scores)
    ]
    summaries.append(aggregate(scores, "overall"))
    for summary in summaries:
        summary["variant"] = variant
    if not quiet:
        head = [
            "group",
            "pages",
            "precision",
            "recall",
            "f1",
            "fragments",
            "merges",
            "spurious",
            "fn",
            "exact_share",
            "mean_abs_col_diff",
            "median_pitch_model",
            "seconds_per_page",
        ]
        print(f"variant: {variant}")
        print(" | ".join(f"{h}" for h in head))
        for summary in summaries:
            print(" | ".join(str(summary[h]) for h in head))
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=gtlib.data_root(),
        help="checkout the manifest's image_path/rtf_path are relative to "
        "(default $SEG_DATA_ROOT, else this checkout)",
    )
    args = parser.parse_args(argv)
    run(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        manifest=args.manifest,
        repo_root=args.repo_root,
        out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
