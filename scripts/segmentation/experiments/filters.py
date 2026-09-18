#!/usr/bin/env python
"""Post-hoc filter rules over a directory of predictions, with their damage.

    filters.py --pred-dir out/v0 --rules F1 F2 F3 --tag f7 [--sweep]

Each rule writes a filtered prediction directory ``out/<variant>-<rule>`` in
the same format, and the report says, for every rule: how many boxes it removed
or merged, how many of those were real text lines, and what the exact-count
share of columns is afterwards.

The geometry rules reuse the pure-Python detectors in
``nomikos/backend/annotation/application/segment_health.py`` wherever they
exist, so what is measured here is the rule that would actually ship.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKTREE = HERE.parents[2]
sys.path.insert(0, str(WORKTREE / "nomikos"))

import gtlib  # noqa: E402
import score as scorelib  # noqa: E402

from backend.annotation.application.segment_health import (  # noqa: E402
    Segment,
    column_bands,
    find_fragments,
    overlap_area,
    page_stats,
    polygon_area,
)

MAIN = Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR")
OUT = HERE / "out"
GT_DIR = Path(
    "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon"
)


def _segments(lines: list[dict[str, Any]]) -> list[Segment]:
    return [
        Segment(id=str(i), points=ln["polygon"], baseline=ln["baseline"])
        for i, ln in enumerate(lines)
    ]


def _bands(lines: list[dict[str, Any]], width: float) -> list[tuple[float, float]]:
    bands = column_bands(_segments(lines), width)
    return bands or [(0.0, width)]


def _band_of(line: dict[str, Any], bands: list[tuple[float, float]]) -> int:
    xs = [p[0] for p in line["polygon"]]
    centre = (min(xs) + max(xs)) / 2
    return min(range(len(bands)), key=lambda i: abs(centre - (bands[i][0] + bands[i][1]) / 2))


def _wh(line: dict[str, Any]) -> tuple[float, float]:
    xs = [p[0] for p in line["polygon"]]
    ys = [p[1] for p in line["polygon"]]
    return max(xs) - min(xs), max(ys) - min(ys)


# --------------------------------------------------------------------------
# rules. each returns (kept lines, n_removed, n_merged)
# --------------------------------------------------------------------------
def rule_f1(lines, gt, **kw):
    """Drop the smaller of two boxes that overlap at or above 50% of the
    smaller box's area."""
    ratio = kw.get("f1_ratio", 0.5)
    areas = [polygon_area(ln["polygon"]) or 1e-9 for ln in lines]
    drop: set[int] = set()
    for i in range(len(lines)):
        if i in drop:
            continue
        bi = _box(lines[i])
        for j in range(i + 1, len(lines)):
            if j in drop:
                continue
            bj = _box(lines[j])
            if bi[1] <= bj[0] or bj[1] <= bi[0] or bi[3] <= bj[2] or bj[3] <= bi[2]:
                continue
            shared = overlap_area(lines[i]["polygon"], lines[j]["polygon"])
            if shared / min(areas[i], areas[j]) >= ratio:
                drop.add(i if areas[i] < areas[j] else j)
    return [ln for i, ln in enumerate(lines) if i not in drop], len(drop), 0


def _box(line):
    xs = [p[0] for p in line["polygon"]]
    ys = [p[1] for p in line["polygon"]]
    return min(xs), max(xs), min(ys), max(ys)


def rule_f2(lines, gt, **kw):
    """Merge same-baseline fragments with the platform's own find_fragments."""
    stats = page_stats(_segments(lines), float(gt.width), float(gt.height))
    merges = find_fragments(_segments(lines), stats)
    if not merges:
        return lines, 0, 0
    out = {i: dict(ln) for i, ln in enumerate(lines)}
    gone: set[int] = set()
    for merge in merges:
        primary, fragment = int(merge.primary_id), int(merge.fragment_id)
        if primary in gone or fragment in gone:
            continue
        out[primary] = {
            "polygon": [list(p) for p in merge.points],
            "baseline": [list(p) for p in merge.baseline],
        }
        gone.add(fragment)
    return [out[i] for i in sorted(out) if i not in gone], 0, len(gone)


def rule_f3(lines, gt, **kw):
    """Remove narrow boxes that sit outside every column band, exempting the
    top 12% of the page (headings and running titles live there)."""
    narrow_fraction = kw.get("f3_narrow", 0.06)
    bands = column_bands(_segments(lines), float(gt.width))
    if not bands:
        return lines, 0, 0
    exempt_y = 0.12 * gt.height
    kept, removed = [], 0
    for ln in lines:
        w, _h = _wh(ln)
        box = _box(ln)
        if box[2] <= exempt_y or w >= narrow_fraction * gt.width:
            kept.append(ln)
            continue
        span = (box[0], box[1])
        inside = any(
            max(0.0, min(span[1], b[1]) - max(span[0], b[0])) >= 0.5 * (span[1] - span[0])
            for b in bands
        )
        if inside:
            kept.append(ln)
        else:
            removed += 1
    return kept, removed, 0


def rule_f4(lines, gt, **kw):
    """Relative outliers: height or width far from the column's own median."""
    low_h = kw.get("f4_low_h", 0.45)
    high_h = kw.get("f4_high_h", 2.2)
    low_w = kw.get("f4_low_w", 0.25)
    bands = _bands(lines, float(gt.width))
    groups: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        groups.setdefault(_band_of(ln, bands), []).append(i)
    drop: set[int] = set()
    for members in groups.values():
        ws = [_wh(lines[i])[0] for i in members]
        hs = [_wh(lines[i])[1] for i in members]
        mw = statistics.median(ws) or 1.0
        mh = statistics.median(hs) or 1.0
        for i in members:
            w, h = _wh(lines[i])
            if h < low_h * mh or h > high_h * mh or w < low_w * mw:
                drop.add(i)
    return [ln for i, ln in enumerate(lines) if i not in drop], len(drop), 0


def _ink_densities(lines, gt):
    import numpy as np
    from PIL import Image, ImageDraw

    with Image.open(gt.image_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32)
    # A per-page ink level: the page is mostly parchment, so the 60th
    # percentile is background and anything much darker is ink.
    background = float(np.percentile(gray, 60))
    out = []
    for ln in lines:
        box = _box(ln)
        x0, x1 = int(max(0, box[0])), int(min(gt.width, box[1] + 1))
        y0, y1 = int(max(0, box[2])), int(min(gt.height, box[3] + 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            out.append(0.0)
            continue
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).polygon([(p[0] - x0, p[1] - y0) for p in ln["polygon"]], fill=255)
        marr = np.asarray(mask, dtype=bool)
        patch = gray[y0:y1, x0:x1]
        if marr.sum() < 4:
            out.append(0.0)
            continue
        dark = np.maximum(0.0, background - patch[marr])
        out.append(float(dark.mean()))
    return out


def rule_f5(lines, gt, **kw):
    """Drop boxes whose ink density is far below the column's median."""
    ratio = kw.get("f5_ratio", 0.45)
    bands = _bands(lines, float(gt.width))
    density = _ink_densities(lines, gt)
    groups: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        groups.setdefault(_band_of(ln, bands), []).append(i)
    drop: set[int] = set()
    for members in groups.values():
        med = statistics.median([density[i] for i in members]) or 1.0
        for i in members:
            if density[i] < ratio * med:
                drop.add(i)
    return [ln for i, ln in enumerate(lines) if i not in drop], len(drop), 0


def _badness(lines, gt):
    """Per-box badness, used by F6 to rank what to drop first."""
    bands = _bands(lines, float(gt.width))
    density = _ink_densities(lines, gt)
    areas = [polygon_area(ln["polygon"]) or 1e-9 for ln in lines]
    boxes = [_box(ln) for ln in lines]
    worst_overlap = [0.0] * len(lines)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            bi, bj = boxes[i], boxes[j]
            if bi[1] <= bj[0] or bj[1] <= bi[0] or bi[3] <= bj[2] or bj[3] <= bi[2]:
                continue
            shared = overlap_area(lines[i]["polygon"], lines[j]["polygon"])
            ratio = shared / min(areas[i], areas[j])
            worst_overlap[i] = max(worst_overlap[i], ratio)
            worst_overlap[j] = max(worst_overlap[j], ratio)
    groups: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        groups.setdefault(_band_of(ln, bands), []).append(i)
    scores = [0.0] * len(lines)
    for members in groups.values():
        mw = statistics.median([_wh(lines[i])[0] for i in members]) or 1.0
        mh = statistics.median([_wh(lines[i])[1] for i in members]) or 1.0
        md = statistics.median([density[i] for i in members]) or 1.0
        for i in members:
            w, h = _wh(lines[i])
            scores[i] = (
                2.0 * worst_overlap[i]
                + 1.5 * max(0.0, 1.0 - w / mw)
                + 1.0 * abs(h - mh) / mh
                + 1.5 * max(0.0, 1.0 - density[i] / md)
            )
    return scores, bands, groups


def rule_f6(lines, gt, **kw):
    """Count-constrained ranking: the transcription says how many lines a
    column has, so drop the k worst-ranked extras in a column that has k too
    many."""
    scores, bands, groups = _badness(lines, gt)
    expected = gt.rtf_counts
    # Bands are left to right, and so are the RTF ranges.
    drop: set[int] = set()
    for band_index in sorted(groups):
        members = groups[band_index]
        want = expected[band_index] if band_index < len(expected) else len(members)
        extra = len(members) - want
        if extra <= 0:
            continue
        ranked = sorted(members, key=lambda i: -scores[i])
        drop.update(ranked[:extra])
    return [ln for i, ln in enumerate(lines) if i not in drop], len(drop), 0


RULES = {
    "F1": rule_f1,
    "F2": rule_f2,
    "F3": rule_f3,
    "F4": rule_f4,
    "F5": rule_f5,
    "F6": rule_f6,
}


# --------------------------------------------------------------------------
# damage accounting
# --------------------------------------------------------------------------
def tp_set(gt: gtlib.GTPage, lines: list[dict[str, Any]]) -> set[int]:
    """Indices of predictions that are the one-to-one match of a paired
    ground-truth line, i.e. the real text lines among the predictions."""
    pred = {"lines": lines, "seconds": 0.0}
    strict = []
    for pi, ln in enumerate(lines):
        pb = ln["baseline"]
        for gi, gl in enumerate(gt.lines):
            rel = scorelib.relation(pb, gl.baseline)
            if rel is None:
                continue
            dy, ov = rel
            gx0, gx1 = scorelib.extent(gl.baseline)
            if ov >= 0.5 * max(gx1 - gx0, 1e-6) and dy <= 0.25 * gt.pitch_for(gl.column):
                strict.append((dy, pi, gi))
    strict.sort()
    used_p, used_g = {}, {}
    for _dy, pi, gi in strict:
        if pi in used_p or gi in used_g:
            continue
        used_p[pi] = gi
        used_g[gi] = pi
    del pred
    return {pi for pi, gi in used_p.items() if gt.lines[gi].paired}


def apply_chain(lines, gt, chain, kwargs):
    removed = merged = 0
    for name in chain:
        lines, r, m = RULES[name](lines, gt, **kwargs)
        removed += r
        merged += m
    return lines, removed, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--chains", nargs="+", required=True,
                        help="comma-joined rule chains, e.g. F1 F2 F3 F1,F2,F3")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--write-dirs", action="store_true")
    parser.add_argument("--param", nargs="*", default=[], help="k=v overrides")
    args = parser.parse_args(argv)

    kwargs = {}
    for item in args.param:
        key, _, value = item.partition("=")
        kwargs[key] = float(value)

    pages = gtlib.load_all(
        gt_dir=args.gt_dir,
        manifest_path=MAIN / "scripts/coptic/manifest.json",
        repo_root=MAIN,
    )
    variant = args.pred_dir.name
    rows = []
    for spec in args.chains:
        chain = spec.split(",")
        total = {
            "variant": variant,
            "chain": spec,
            "removed": 0,
            "merged": 0,
            "removed_were_text": 0,
            "merged_were_text": 0,
            "pred_before": 0,
            "pred_after": 0,
        }
        out_dir = OUT / f"{variant}-{spec.replace(',', '')}"
        scores_before, scores_after = [], []
        for gt in pages:
            path = args.pred_dir / f"{gt.slug}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            lines = payload["lines"]
            before_tp = tp_set(gt, lines)
            kept, removed, merged = apply_chain(list(lines), gt, chain, kwargs)
            # Which survived: identity by baseline, which the merge rewrites.
            kept_keys = {json.dumps(ln["baseline"]) for ln in kept}
            lost = [
                i
                for i, ln in enumerate(lines)
                if json.dumps(ln["baseline"]) not in kept_keys
            ]
            total["removed"] += removed
            total["merged"] += merged
            total["removed_were_text"] += sum(1 for i in lost if i in before_tp)
            total["pred_before"] += len(lines)
            total["pred_after"] += len(kept)
            scores_before.append(scorelib.score_page(gt, payload))
            after = dict(payload)
            after["lines"] = kept
            scores_after.append(scorelib.score_page(gt, after))
            if args.write_dirs:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{gt.slug}.json").write_text(
                    json.dumps({**payload, "variant": f"{variant}-{spec}", "lines": kept})
                )
        before = scorelib.aggregate(scores_before, "before")
        after = scorelib.aggregate(scores_after, "after")
        row = {
            **total,
            "f1_before": before["f1"],
            "f1_after": after["f1"],
            "precision_after": after["precision"],
            "recall_after": after["recall"],
            "fn_before": before["fn"],
            "fn_after": after["fn"],
            "fragments_after": after["fragments"],
            "merges_after": after["merges"],
            "exact_share_before": before["exact_share"],
            "exact_share_after": after["exact_share"],
            "mean_abs_col_diff_after": after["mean_abs_col_diff"],
        }
        rows.append(row)
        print(
            f"{spec:16} removed {row['removed']:5} merged {row['merged']:4} "
            f"false-deletes {row['removed_were_text']:4} "
            f"F1 {row['f1_before']:.3f}->{row['f1_after']:.3f} "
            f"exact {row['exact_share_before']:.3f}->{row['exact_share_after']:.3f} "
            f"missed {row['fn_before']}->{row['fn_after']}",
            flush=True,
        )
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
