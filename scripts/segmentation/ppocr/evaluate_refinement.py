#!/usr/bin/env python3
"""Score the ppocr-det adapter with refinement off, default and drop policies.

Runs the production entry point on the 12 Coptic benchmark pages three ways
and scores each response against the same target lines with the benchmark's
own line matching: a prediction matches a target when at least 50 percent of
31 mid-axis samples fall inside the target polygon with at least 50 percent
horizontal overlap, assigned one to one for maximum cardinality then affinity
(server/ppocrv6_quality_eval_20260919.py: `axis`, `assign`, `finish`). The
benchmark's own tool recovers 1008 of 1018 care targets on these pages; this
port recovers 1002, so it is a consistent off-vs-defaults comparator rather
than identical tooling.

Prints per variant: detections, precision, recall, F1, care targets covered
by exactly one detection, line pairs sharing at least 20 percent area,
suspects flagged and real lines wrongly flagged. Also writes reading-order
overlays for vat-1r and c13 with suspects in red. The pass gate lives in the
release report, not in the exit code: this script always exits 0.

Example::

    PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 $REPO/.venv/bin/python \\
        scripts/segmentation/ppocr/evaluate_refinement.py \\
        --fixtures $WORK/_ppocr-parity/fixtures \\
        --images $WORK/_orli-env/server/images \\
        --gt $WORK/_gt-recon/human-guide/v4/recon \\
        --onnx $WORK/_ppocr-parity/pp-ocrv6-medium-det.onnx \\
        --output-dir $WORK/_ppocr-parity/refinement
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, Polygon

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.segmentation.ppocr._fixtures import (
    COPTIC_PAGES as PAGES,
)
from scripts.segmentation.ppocr._fixtures import (
    decode_overlay_canvas,
    load_image_bytes,
    load_targets,
    put_order_label,
)

OVERLAY_PAGES = ("vat-1r", "c13")
N_SAMPLES = 31
PAIR_THRESHOLD = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, required=True, help="Directory of <page>.json fixtures"
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory of page images")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth recon directory")
    parser.add_argument("--onnx", type=Path, required=True, help="Adapter ONNX artifact")
    parser.add_argument("--artifact-sha256", default=None, help="Expected SHA-256 of the ONNX")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where overlays go")
    parser.add_argument(
        "--merge-gap-ratio",
        action="append",
        default=[],
        help=(
            "Score the defaults and drop variants at these merge gap ratios "
            "(repeatable or comma separated). In sweep mode overlays are "
            "skipped and the JSON gains a per-ratio sweep summary."
        ),
    )
    parser.add_argument(
        "--merge-max-overlap-ratio",
        action="append",
        default=[],
        help=(
            "Score the defaults and drop variants at gap 0.25 for these "
            "merge max-overlap ratios (repeatable or comma separated). "
            "Same sweep output style as --merge-gap-ratio."
        ),
    )
    parser.add_argument(
        "--box-type",
        choices=("quad", "poly", "both"),
        default="quad",
        help=(
            "Which served geometry to score: quad masks, poly masks, or both "
            "(both runs off/defaults/drop in each mode and prints the "
            "quad-vs-poly comparison with polygon statistics)."
        ),
    )
    return parser.parse_args()


def _sweep_ratios(raw: list[str], flag: str) -> list[float]:
    ratios: list[float] = []
    for chunk in raw:
        ratios.extend(float(part) for part in chunk.split(",") if part.strip())
    if not ratios:
        raise ValueError(f"{flag} needs at least one number")
    return ratios


def _axis(polygon: Polygon) -> list[list[float]]:
    """31 mid-height axis samples of a quad, per the benchmark's `axis`."""
    x0, y0, x1, y1 = polygon.bounds
    if x1 <= x0:
        return []
    points = []
    for fraction in (np.arange(N_SAMPLES) + 0.5) / N_SAMPLES:
        x = float(x0 + fraction * (x1 - x0))
        section = polygon.intersection(LineString([(x, y0 - 1), (x, y1 + 1)]))
        if section.is_empty:
            return []
        points.append([x, (section.bounds[1] + section.bounds[3]) / 2])
    return points


def _eligibility(targets: list[dict], polygons: list[Polygon]) -> tuple[np.ndarray, np.ndarray]:
    """Per (prediction, target) eligibility and affinity, per the benchmark's `score`."""
    eligible = np.zeros((len(polygons), len(targets)), dtype=bool)
    affinity = np.zeros(eligible.shape, dtype=float)
    for pi, polygon in enumerate(polygons):
        samples = _axis(polygon)
        if not samples:
            continue
        extent = (polygon.bounds[0], polygon.bounds[2])
        for gi, target in enumerate(targets):
            gt = target["polygon"]
            x0, _, x1, _ = gt.bounds
            overlap = max(0.0, min(extent[1], x1) - max(extent[0], x0)) / max(x1 - x0, 1e-12)
            if overlap < 0.5:
                continue
            inside = sum(gt.covers(Point(point)) for point in samples) / N_SAMPLES
            eligible[pi, gi] = inside >= 0.5
            affinity[pi, gi] = (inside + min(overlap, 1.0)) / 2
    return eligible, affinity


def _assign_pairs(
    eligible: np.ndarray, affinity: np.ndarray, care: list[int]
) -> tuple[list[tuple[int, int]], set[int]]:
    """Maximum-cardinality one-to-one assignment over eligible pairs."""
    weight = eligible[:, care] * (min(eligible[:, care].shape) + 1 + affinity[:, care])
    paired: list[tuple[int, int]] = []
    used: set[int] = set()
    if weight.size:
        rows, cols = linear_sum_assignment(weight, maximize=True)
        for row, col in zip(rows, cols, strict=True):
            if eligible[int(row), care[int(col)]]:
                paired.append((int(row), care[int(col)]))
                used.add(int(row))
    return paired, used


def _score_page(targets: list[dict], polygons: list[Polygon]) -> dict:
    """One-to-one line matching, per the benchmark's `score` and `finish`."""
    care = [i for i, target in enumerate(targets) if target["care"]]
    ignore = [i for i, target in enumerate(targets) if not target["care"]]
    eligible, affinity = _eligibility(targets, polygons)
    paired, used = _assign_pairs(eligible, affinity, care)
    return _summarise_match(targets, polygons, eligible, care, ignore, paired, used)


def _pairs_above(polygons: list[Polygon], threshold: float) -> int:
    """Line pairs sharing at least `threshold` of the smaller polygon."""
    count = 0
    for left, right in itertools.combinations(polygons, 2):
        smaller = min(left.area, right.area)
        if smaller > 0 and left.intersection(right).area / smaller >= threshold:
            count += 1
    return count


def _match_count(eligible: np.ndarray, pred_count: int, g: int) -> int:
    """Predictions eligible for target g."""
    return sum(1 for p in range(pred_count) if eligible[p, g])


def _summarise_match(
    targets: list[dict],
    polygons: list[Polygon],
    eligible: np.ndarray,
    care: list[int],
    ignore: list[int],
    paired: list[tuple[int, int]],
    used: set[int],
) -> dict:
    """Precision/recall counts and matched ids, per the benchmark's `finish`."""
    ignored = [
        p
        for p in range(len(polygons))
        if p not in used
        and any(eligible[p, g] for g in ignore)
        and not any(eligible[p, g] for g in care)
    ]
    fps = sorted(set(range(len(polygons))) - used - set(ignored))
    fns = sorted(set(care) - {g for _, g in paired})
    coverage = {g: _match_count(eligible, len(polygons), g) for _, g in paired}
    alone = sum(1 for count in coverage.values() if count == 1)
    paired_ids = sorted(targets[g]["id"] for _, g in paired)
    alone_ids = sorted(targets[g]["id"] for g, count in coverage.items() if count == 1)
    return {
        "paired_ids": paired_ids,
        "alone_ids": alone_ids,
        "n_gt": len(care),
        "n_pred": len(polygons),
        "tp": len(paired),
        "fp": len(fps),
        "fn": len(fns),
        "ignored": len(ignored),
        "alone": alone,
        "matched_pred": sorted(used),
    }


def _adjacent_overlap_mean(polygons: list[Polygon]) -> float:
    """Mean shared-over-smaller area between consecutive served lines."""
    ratios = []
    for upper, lower in zip(polygons, polygons[1:], strict=False):
        smaller = min(upper.area, lower.area)
        if smaller > 0:
            ratios.append(upper.intersection(lower).area / smaller)
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


def _poly_quad_area_ratios(
    poly_polygons: list[Polygon], quad_polygons: list[Polygon]
) -> list[float]:
    """Per-line polygon over quad area, paired by reading order position."""
    ratios = []
    for poly, quad in zip(poly_polygons, quad_polygons, strict=False):
        if quad.area > 0:
            ratios.append(poly.area / quad.area)
    return ratios


def _draw_overlay(
    page: str, image_bytes: bytes, lines: list, output_dir: Path, mode: str | None = None
) -> list[int]:
    """Reading-order overlay: body quads green with numbers, suspects red.

    Drawing mirrors verify_adapter.py (green quads, order numbers beside the
    left edge, yellow baselines); suspects use red for both quad and number.
    Returns the suspect line numbers.
    """
    canvas, scale, width, height = decode_overlay_canvas(image_bytes, page)
    thickness = max(1, int(round(2 * scale)))

    suspect_numbers = []
    for position, line in enumerate(lines):
        number = position + 1
        quad = np.asarray(line["points"], dtype=np.float64)
        suspect = bool(line["source_metadata"].get("suspect", False))
        colour = (0, 0, 255) if suspect else (0, 200, 0)
        if suspect:
            suspect_numbers.append(number)
        cv2.polylines(
            canvas, [quad.astype(np.int32).reshape(-1, 1, 2)], True, colour, thickness, cv2.LINE_AA
        )
        baseline = np.asarray(line["baseline"]["points"], dtype=np.float64)
        if not suspect:
            cv2.polylines(
                canvas,
                [baseline.astype(np.int32).reshape(-1, 1, 2)],
                False,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        put_order_label(
            canvas,
            number,
            float(quad[:, 0].min()),
            float(quad[:, 1].mean()),
            scale,
            width,
            height,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode is None:
        overlay_name = f"{page}.refined.overlay.jpg"
    else:
        overlay_name = f"{page}.refined.{mode}.overlay.jpg"
    ok = cv2.imwrite(str(output_dir / overlay_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"could not write overlay for page {page}")
    return suspect_numbers


def _build_variants(
    gap_sweep: list[float], overlap_sweep: list[float], box_type: str
) -> tuple[dict[str, dict | None], list[tuple[str, str, str]]]:
    """Variant param sets plus the sweep ledger; empty sweeps mean off/defaults/drop."""
    if gap_sweep or overlap_sweep:
        variants: dict[str, dict | None] = {
            "off": {
                "merge_fragments": False,
                "resolve_overlaps": False,
                "noise_policy": "off",
            },
        }
        sweep: list[tuple[str, str, str]] = []
        for ratio in gap_sweep:
            variants[f"gap-{ratio}"] = {"merge_gap_ratio": ratio}
            variants[f"drop-{ratio}"] = {"noise_policy": "drop", "merge_gap_ratio": ratio}
            sweep.append((str(ratio), f"gap-{ratio}", f"drop-{ratio}"))
        for ratio in overlap_sweep:
            variants[f"ov-{ratio}"] = {
                "merge_gap_ratio": 0.25,
                "merge_max_overlap_ratio": ratio,
            }
            variants[f"dropov-{ratio}"] = {
                "noise_policy": "drop",
                "merge_gap_ratio": 0.25,
                "merge_max_overlap_ratio": ratio,
            }
            sweep.append((str(ratio), f"ov-{ratio}", f"dropov-{ratio}"))
        return variants, sweep
    if box_type == "both":
        return (
            {
                "off-quad": {
                    "merge_fragments": False,
                    "resolve_overlaps": False,
                    "noise_policy": "off",
                    "box_type": "quad",
                },
                "off-poly": {
                    "merge_fragments": False,
                    "resolve_overlaps": False,
                    "noise_policy": "off",
                    "box_type": "poly",
                },
                "defaults-quad": {"box_type": "quad"},
                "defaults-poly": {"box_type": "poly"},
                "drop-quad": {"noise_policy": "drop", "box_type": "quad"},
                "drop-poly": {"noise_policy": "drop", "box_type": "poly"},
            },
            [],
        )
    return (
        {
            "off": {
                "merge_fragments": False,
                "resolve_overlaps": False,
                "noise_policy": "off",
                "box_type": box_type,
            },
            "defaults": {"box_type": box_type},
            "drop": {"noise_policy": "drop", "box_type": box_type},
        },
        [],
    )


def _score_variant_page(
    params: dict | None,
    image_bytes: bytes,
    targets: list[dict],
    args: argparse.Namespace,
) -> tuple[dict, list, float]:
    """Run one variant on one page and score it against its targets.

    Returns the score entry, the served line points and the run time in seconds.
    """
    from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment

    started = time.perf_counter()
    response = run_ppocr_det_segment(
        image_bytes,
        model_path=args.onnx,
        artifact_sha256=args.artifact_sha256,
        params=params,
    )
    elapsed = time.perf_counter() - started
    polygons = [Polygon(line.points) for line in response.lines]
    scored = _score_page(targets, polygons)
    suspects = [
        position
        for position, line in enumerate(response.lines)
        if line.source_metadata.get("suspect", False)
    ]
    wrong = sum(1 for p in suspects if p in scored["matched_pred"])
    merged = sum(1 for line in response.lines if line.source_metadata.get("merged_from", 1) > 1)
    entry = {
        **scored,
        "pairs": _pairs_above(polygons, PAIR_THRESHOLD),
        "suspects": len(suspects),
        "wrong": wrong,
        "merged": merged,
    }
    return entry, [line.points for line in response.lines], elapsed


def _accumulate_totals(total: dict[str, float], entry: dict) -> None:
    """Fold one page-variant score into its running totals."""
    for key in ("tp", "fp", "fn", "n_pred", "alone"):
        total[key] += entry[key]
    total["pairs"] += entry["pairs"]
    total["suspects"] += entry["suspects"]
    total["wrong"] += entry["wrong"]
    total["merged"] += entry["merged"]


def _write_page_overlay(
    page: str, image_bytes: bytes, args: argparse.Namespace, pages: dict, mode: str
) -> None:
    """Re-run the defaults variant and write its reading-order overlay, per mode."""
    from nomikos_inference.architectures.ppocr_det import run_ppocr_det_segment

    both = args.box_type == "both"
    for overlay_mode in ("quad", "poly") if both else (mode,):
        defaults = run_ppocr_det_segment(
            image_bytes,
            model_path=args.onnx,
            artifact_sha256=args.artifact_sha256,
            params={"box_type": overlay_mode},
        )
        serial = [
            {
                "points": line.points,
                "baseline": line.baseline,
                "source_metadata": line.source_metadata,
            }
            for line in defaults.lines
        ]
        numbers = _draw_overlay(
            page, image_bytes, serial, args.output_dir, mode=overlay_mode if both else None
        )
        if both:
            pages[page][f"suspect_numbers_{overlay_mode}"] = numbers
            print(f"{page} {overlay_mode} suspect line numbers: {numbers}", flush=True)
        else:
            pages[page]["suspect_numbers"] = numbers
            print(f"{page} suspect line numbers: {numbers}", flush=True)


def _print_variant_totals(
    variants: dict[str, dict | None], totals: dict[str, dict[str, float]]
) -> None:
    """Precision/recall/F1 per variant, in variant order."""
    for name in variants:
        total = totals[name]
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = (
            2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"])
            if 2 * total["tp"] + total["fp"] + total["fn"]
            else 0.0
        )
        total["precision"] = precision
        total["recall"] = recall
        total["f1"] = f1
        print(
            f"{name} detections={total['n_pred']} P={precision:.4f} R={recall:.4f} "
            f"F1={f1:.4f} alone={total['alone']} pairs={total['pairs']} "
            f"suspects={total['suspects']} wrong={total['wrong']} merged={total['merged']}",
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    gap_sweep = (
        _sweep_ratios(args.merge_gap_ratio, "--merge-gap-ratio") if args.merge_gap_ratio else []
    )
    overlap_sweep = (
        _sweep_ratios(args.merge_max_overlap_ratio, "--merge-max-overlap-ratio")
        if args.merge_max_overlap_ratio
        else []
    )
    variants, sweep = _build_variants(gap_sweep, overlap_sweep, args.box_type)
    mode = args.box_type if args.box_type != "both" else "quad"
    for name, params in list(variants.items()):
        if params is None:
            variants[name] = {"box_type": mode}
        else:
            params.setdefault("box_type", mode)
    totals: dict[str, dict[str, float]] = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_pred": 0,
            "alone": 0,
            "pairs": 0,
            "suspects": 0,
            "wrong": 0,
            "merged": 0,
        }
        for name in variants
    }
    pages: dict[str, dict] = {}
    served: dict[str, dict[str, list]] = {}
    timings: dict[str, list[float]] = {name: [] for name in variants}
    print(
        "variant detections precision recall f1 alone pairs suspects wrong",
        flush=True,
    )
    for page in PAGES:
        image_bytes = load_image_bytes(page, args.fixtures, args.images)
        targets = load_targets(args.gt, page)
        pages[page] = {}
        served[page] = {}
        for name, params in variants.items():
            entry, served[page][name], elapsed = _score_variant_page(
                params, image_bytes, targets, args
            )
            timings[name].append(elapsed)
            pages[page][name] = entry
            _accumulate_totals(totals[name], entry)
        if not sweep and page in OVERLAY_PAGES:
            _write_page_overlay(page, image_bytes, args, pages, mode)
    _print_variant_totals(variants, totals)
    payload: dict = {"pages": pages, "totals": totals}
    if args.box_type == "both" and not sweep:
        payload["poly_stats"] = _print_quad_poly_comparison(pages, served, timings)
    if sweep:
        payload["sweep"] = _summarise_sweep(pages, totals, sweep)
    (args.output_dir / "refinement-evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


def _print_quad_poly_comparison(
    pages: dict, served: dict[str, dict[str, list]], timings: dict[str, list[float]]
) -> dict:
    """Per-page and overall quad-vs-poly geometry statistics.

    Grouping decisions run on quads in both modes, so the line-level counts
    are expected identical; the polygons should be smaller (tighter around
    the ink) and overlap their vertical neighbours less.
    """
    print(
        "page n_quad n_poly mean_pts mean_poly_over_quad adj_overlap_quad adj_overlap_poly",
        flush=True,
    )
    all_points: list[int] = []
    all_ratios: list[float] = []
    quad_overlaps: list[float] = []
    poly_overlaps: list[float] = []
    summary: dict[str, dict] = {}
    for page in pages:
        quad_polys = [Polygon(points) for points in served[page]["defaults-quad"]]
        poly_polys = [Polygon(points) for points in served[page]["defaults-poly"]]
        points = [len(ring) for ring in served[page]["defaults-poly"]]
        ratios = _poly_quad_area_ratios(poly_polys, quad_polys)
        quad_overlap = _adjacent_overlap_mean(quad_polys)
        poly_overlap = _adjacent_overlap_mean(poly_polys)
        all_points.extend(points)
        all_ratios.extend(ratios)
        quad_overlaps.append(quad_overlap)
        poly_overlaps.append(poly_overlap)
        summary[page] = {
            "n_quad": len(quad_polys),
            "n_poly": len(poly_polys),
            "mean_points": float(sum(points) / len(points)) if points else 0.0,
            "mean_poly_over_quad": float(sum(ratios) / len(ratios)) if ratios else 0.0,
            "adj_overlap_quad": quad_overlap,
            "adj_overlap_poly": poly_overlap,
        }
        row = summary[page]
        print(
            f"{page} {row['n_quad']} {row['n_poly']} {row['mean_points']:.2f} "
            f"{row['mean_poly_over_quad']:.4f} {row['adj_overlap_quad']:.4f} "
            f"{row['adj_overlap_poly']:.4f}",
            flush=True,
        )
    overall = {
        "mean_points": float(sum(all_points) / len(all_points)) if all_points else 0.0,
        "mean_poly_over_quad": float(sum(all_ratios) / len(all_ratios)) if all_ratios else 0.0,
        "adj_overlap_quad": float(sum(quad_overlaps) / len(quad_overlaps))
        if quad_overlaps
        else 0.0,
        "adj_overlap_poly": float(sum(poly_overlaps) / len(poly_overlaps))
        if poly_overlaps
        else 0.0,
        "sec_per_page_quad": float(sum(timings["defaults-quad"]) / len(timings["defaults-quad"])),
        "sec_per_page_poly": float(sum(timings["defaults-poly"]) / len(timings["defaults-poly"])),
    }
    print(
        f"overall mean_pts={overall['mean_points']:.2f} "
        f"mean_poly_over_quad={overall['mean_poly_over_quad']:.4f} "
        f"adj_overlap_quad={overall['adj_overlap_quad']:.4f} "
        f"adj_overlap_poly={overall['adj_overlap_poly']:.4f} "
        f"sec_per_page_quad={overall['sec_per_page_quad']:.2f} "
        f"sec_per_page_poly={overall['sec_per_page_poly']:.2f}",
        flush=True,
    )
    summary["overall"] = overall
    return summary


def _summarise_sweep(pages: dict, totals: dict, sweep: list[tuple[str, str, str]]) -> dict:
    """Per-ratio summary: metrics, lines lost against off, lines fixed, merges."""
    summary = {}
    print("ratio R P Pdrop F1 F1drop lost fixed merged", flush=True)
    for ratio, gap_name, drop_name in sweep:
        gap, drop = totals[gap_name], totals[drop_name]
        lost: dict[str, list] = {}
        fixed: dict[str, list] = {}
        for page, entry in pages.items():
            off_paired = set(entry["off"]["paired_ids"])
            off_alone = set(entry["off"]["alone_ids"])
            gap_paired = set(entry[gap_name]["paired_ids"])
            gap_alone = set(entry[gap_name]["alone_ids"])
            page_lost = sorted(off_paired - gap_paired)
            page_fixed = sorted(gap_alone - off_alone)
            if page_lost:
                lost[page] = page_lost
            if page_fixed:
                fixed[page] = page_fixed
        summary[ratio] = {
            "recall": gap["recall"],
            "precision": gap["precision"],
            "precision_drop": drop["precision"],
            "f1": gap["f1"],
            "f1_drop": drop["f1"],
            "lost": lost,
            "fixed": fixed,
            "merged": gap["merged"],
        }
        print(
            f"{ratio} R={gap['recall']:.4f} P={gap['precision']:.4f} "
            f"Pdrop={drop['precision']:.4f} F1={gap['f1']:.4f} "
            f"F1drop={drop['f1']:.4f} lost={sum(len(v) for v in lost.values())} "
            f"fixed={sum(len(v) for v in fixed.values())} merged={gap['merged']}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
