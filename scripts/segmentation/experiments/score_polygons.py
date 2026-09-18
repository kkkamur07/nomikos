#!/usr/bin/env python
"""Score line POLYGONS (segments) of a prediction directory against the Coptic
ground truth.

    score_polygons.py --pred-dir out/orli-fixed-poly --gt-dir <recon> \
        --manifest scripts/coptic/manifest.json [--out results.csv]

``score.py`` answers "did the model find the line?". This answers "is the
segment the model drew around the line the right shape?".

Pairing is the baseline matching of ``score.py``, imported from it rather than
re-derived: a prediction is paired with a ground-truth line when their
baselines overlap over at least half the ground-truth x extent and the median
vertical distance over that overlap is at most 0.25 x the local column pitch,
matched one to one, closest pair first. Ground-truth lines that the pairing
tools left unpaired (folio numbers, running titles, split rows) are don't care
here exactly as they are there, and are left out of the polygon statistics.

BIAS WARNING
------------
The ground-truth polygons are CORRECTED KRAKEN POLYGONS: a human moved, merged,
split and deleted lines that kraken's blla.mlmodel produced, but the polygon
shape of a line the human kept is still kraken's own shape, drawn by kraken's
watershed polygonizer. Every IoU number below is therefore biased in kraken's
favour, and a systematically different but equally defensible segment
convention (a tighter box, a different ascender allowance) will score lower
without being worse. Read the IoU columns as "how kraken-like is this segment",
and read the neighbour-overlap, height-to-pitch and degeneracy columns, which
do not reference the ground-truth shape at all, as the shape-quality signal.

Metrics per page, per manuscript and overall
--------------------------------------------
* iou_mean / iou_median: polygon IoU over matched pairs.
* iou_ge_50 / iou_ge_70: share of matched pairs at or above that IoU.
* neigh_defects: pairs of PREDICTED polygons whose intersection is at least
  20 percent of the smaller polygon's area. On these manuscripts 5 to 7 percent
  overlap between vertical neighbours is normal ink bleed, 20 percent is always
  a defect.
* height_over_pitch: mean predicted polygon height divided by the ground-truth
  pitch of the column the polygon sits in. Around 1.0 means the segment is as
  tall as the line spacing; well above 1.0 means it eats its neighbours.
* degenerate: predicted polygons with fewer than three points, zero area, or a
  self-intersecting ring.
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
import score  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

NEIGHBOUR_OVERLAP = 0.20


# --------------------------------------------------------------------------
# polygons
# --------------------------------------------------------------------------
@dataclass
class Shape:
    """A prediction's polygon plus why it is unusable, if it is."""

    geom: Polygon | None
    reason: str = ""

    @property
    def degenerate(self) -> bool:
        return bool(self.reason)


def to_shape(points: list[list[float]] | None) -> Shape:
    if not points or len(points) < 3:
        return Shape(None, "few_points")
    ring = Polygon(points)
    if not ring.is_valid:
        fixed = ring.buffer(0)
        geom = fixed if (not fixed.is_empty and fixed.area > 0) else None
        return Shape(geom if isinstance(geom, Polygon) else None, "self_intersecting")
    if ring.area <= 0:
        return Shape(None, "zero_area")
    return Shape(ring)


def iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def neighbour_defects(shapes: list[Shape]) -> int:
    """Pairs of predicted polygons overlapping by >= 20 percent of the smaller."""
    geoms = [s.geom for s in shapes if s.geom is not None]
    if len(geoms) < 2:
        return 0
    tree = STRtree(geoms)
    defects = 0
    for i, g in enumerate(geoms):
        for raw in tree.query(g):
            j = int(raw)
            if j <= i:  # each unordered pair once
                continue
            other = geoms[j]
            inter = g.intersection(other).area
            smaller = min(g.area, other.area)
            if smaller > 0 and inter / smaller >= NEIGHBOUR_OVERLAP:
                defects += 1
    return defects


# --------------------------------------------------------------------------
# pairing (score.py's strict matching, one to one)
# --------------------------------------------------------------------------
def strict_pairs(gt: gtlib.GTPage, pred_baselines: list[list[list[float]]]) -> dict[int, int]:
    pitches = [gt.pitch_for(ln.column) for ln in gt.lines]
    cand: list[tuple[float, int, int]] = []
    for pi, pb in enumerate(pred_baselines):
        for gi, gl in enumerate(gt.lines):
            rel = score.relation(pb, gl.baseline)
            if rel is None:
                continue
            dy, ov = rel
            gx0, gx1 = score.extent(gl.baseline)
            if ov >= 0.5 * max(gx1 - gx0, 1e-6) and dy <= 0.25 * pitches[gi]:
                cand.append((dy, pi, gi))
    cand.sort()
    used_pred: dict[int, int] = {}
    used_gt: dict[int, int] = {}
    for _dy, pi, gi in cand:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred[pi] = gi
        used_gt[gi] = pi
    return used_pred


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass
class PagePoly:
    slug: str
    manuscript: str
    n_pred: int
    n_with_poly: int
    matched: int
    matched_scored: int
    ious: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)
    neigh: int = 0
    degenerate: int = 0
    degenerate_kinds: dict[str, int] = field(default_factory=dict)


def score_page(gt: gtlib.GTPage, pred: dict[str, Any]) -> PagePoly:
    lines = [ln for ln in pred["lines"] if len(ln.get("baseline", [])) >= 2]
    baselines = [ln["baseline"] for ln in lines]
    shapes = [to_shape(ln.get("polygon")) for ln in lines]

    kinds: dict[str, int] = {}
    for s in shapes:
        if s.degenerate:
            kinds[s.reason] = kinds.get(s.reason, 0) + 1

    pairs = strict_pairs(gt, baselines)
    dontcare_gt = {gi for gi, gl in enumerate(gt.lines) if not gl.paired}

    ious: list[float] = []
    for pi, gi in pairs.items():
        if gi in dontcare_gt:
            continue
        pshape = shapes[pi]
        if pshape.geom is None:
            ious.append(0.0)
            continue
        gpoly = to_shape(gt.lines[gi].polygon).geom
        if gpoly is None:
            continue
        ious.append(iou(pshape.geom, gpoly))

    heights: list[float] = []
    for s in shapes:
        if s.geom is None:
            continue
        x0, y0, x1, y1 = s.geom.bounds
        pitch = gt.pitch_for(gt.column_of_x((x0 + x1) / 2.0))
        if pitch > 0:
            heights.append((y1 - y0) / pitch)

    return PagePoly(
        slug=gt.slug,
        manuscript=gt.manuscript,
        n_pred=len(lines),
        n_with_poly=sum(1 for s in shapes if s.geom is not None),
        matched=len(pairs),
        matched_scored=len(ious),
        ious=ious,
        heights=heights,
        neigh=neighbour_defects(shapes),
        degenerate=sum(1 for s in shapes if s.degenerate),
        degenerate_kinds=kinds,
    )


def summarise(pages: list[PagePoly], label: str) -> dict[str, Any]:
    ious = [v for p in pages for v in p.ious]
    heights = [v for p in pages for v in p.heights]
    kinds: dict[str, int] = {}
    for p in pages:
        for k, v in p.degenerate_kinds.items():
            kinds[k] = kinds.get(k, 0) + v
    return {
        "group": label,
        "pages": len(pages),
        "pred_lines": sum(p.n_pred for p in pages),
        "with_polygon": sum(p.n_with_poly for p in pages),
        "matched": sum(p.matched_scored for p in pages),
        "iou_mean": round(statistics.fmean(ious), 4) if ious else 0.0,
        "iou_median": round(statistics.median(ious), 4) if ious else 0.0,
        "iou_ge_50": round(sum(1 for v in ious if v >= 0.5) / len(ious), 4) if ious else 0.0,
        "iou_ge_70": round(sum(1 for v in ious if v >= 0.7) / len(ious), 4) if ious else 0.0,
        "neigh_defects": sum(p.neigh for p in pages),
        "height_over_pitch": round(statistics.fmean(heights), 3) if heights else 0.0,
        "degenerate": sum(p.degenerate for p in pages),
        "degenerate_kinds": " ".join(f"{k}:{v}" for k, v in sorted(kinds.items())) or "-",
    }


def page_row(variant: str, p: PagePoly) -> dict[str, Any]:
    return {
        "variant": variant,
        "slug": p.slug,
        "manuscript": p.manuscript,
        "pred_lines": p.n_pred,
        "with_polygon": p.n_with_poly,
        "matched": p.matched_scored,
        "iou_mean": round(statistics.fmean(p.ious), 4) if p.ious else 0.0,
        "iou_median": round(statistics.median(p.ious), 4) if p.ious else 0.0,
        "iou_ge_50": round(sum(1 for v in p.ious if v >= 0.5) / len(p.ious), 4) if p.ious else 0.0,
        "iou_ge_70": round(sum(1 for v in p.ious if v >= 0.7) / len(p.ious), 4) if p.ious else 0.0,
        "neigh_defects": p.neigh,
        "height_over_pitch": round(statistics.fmean(p.heights), 3) if p.heights else 0.0,
        "degenerate": p.degenerate,
    }


PER_PAGE_FIELDS = list(page_row("", PagePoly("", "", 0, 0, 0, 0)).keys())

HEAD = [
    "group",
    "pages",
    "pred_lines",
    "with_polygon",
    "matched",
    "iou_mean",
    "iou_median",
    "iou_ge_50",
    "iou_ge_70",
    "neigh_defects",
    "height_over_pitch",
    "degenerate",
    "degenerate_kinds",
]

BIAS_NOTE = (
    "NOTE: the ground-truth polygons are CORRECTED KRAKEN POLYGONS (a human "
    "fixed which lines exist, not the shapes kraken drew for them), so every "
    "IoU column is biased towards kraken. neigh_defects, height_over_pitch and "
    "degenerate do not use the ground-truth shape and are unbiased."
)


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
    gts = gtlib.load_all(gt_dir=gt_dir, manifest_path=manifest, repo_root=repo_root)
    if only:
        gts = [g for g in gts if g.slug in only]
    variant = pred_dir.name
    pages: list[PagePoly] = []
    for gt in gts:
        path = pred_dir / f"{gt.slug}.json"
        if not path.exists():
            continue
        pred = json.loads(path.read_text())
        variant = pred.get("variant") or variant
        pages.append(score_page(gt, pred))
    if not pages:
        raise SystemExit(f"no predictions found in {pred_dir}")

    rows = [page_row(variant, p) for p in pages]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_PAGE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    summaries = [
        summarise([p for p in pages if p.manuscript == ms], ms)
        for ms in ("vatican", "napoli")
        if any(p.manuscript == ms for p in pages)
    ]
    summaries.append(summarise(pages, "overall"))
    for s in summaries:
        s["variant"] = variant

    if not quiet:
        print(BIAS_NOTE)
        print(f"variant: {variant}  (pred-dir {pred_dir})")
        print("per page: " + " | ".join(PER_PAGE_FIELDS[1:]))
        for row in rows:
            print("          " + " | ".join(str(row[h]) for h in PER_PAGE_FIELDS[1:]))
        print("summary:  " + " | ".join(HEAD))
        for s in summaries:
            print("          " + " | ".join(str(s[h]) for h in HEAD))
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--only", default=None, help="comma separated slugs")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR"),
        help="checkout the manifest's image_path/rtf_path are relative to",
    )
    args = parser.parse_args(argv)
    run(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        manifest=args.manifest,
        repo_root=args.repo_root,
        out=args.out,
        only=set(args.only.split(",")) if args.only else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
