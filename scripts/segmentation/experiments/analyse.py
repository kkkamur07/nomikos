#!/usr/bin/env python
"""Score every variant directory on a common page set and print the tables.

    analyse.py [--pages vat-1r c13 ...] [--variants v0 v1-crop ...] --out-csv out/csv/summary.csv

Restricting to a common page set matters: the cheap variants ran on all 33
pages and the expensive ones on a 12-page subset, and comparing the two
directly would compare page difficulty, not variants.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score as scorelib  # noqa: E402

MAIN = Path("/Users/krishuagarwal/Desktop/Programming/python/greekOCR")
OUT = HERE / "out"
GT_DIR = Path(
    "/Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_gt-recon/human-guide/v4/recon"
)

# The effective scale each variant puts the page at, relative to production.
# Production resizes the whole page to 1800 px; a crop or a horizontal tile
# raises that, a padded canvas lowers it.
SCALE_MULTIPLIER = {
    "v5-s050": 0.50,
    "v5-s075": 0.75,
    "v0": 1.00,
    "v5-s150": 1.50,
    "v2-tiles2": 2.00,
    "v2-tiles3": 3.00,
}

COLUMNS = [
    "variant",
    "group",
    "pages",
    "gt_lines",
    "pred_lines",
    "precision",
    "recall",
    "f1",
    "fragments",
    "merges",
    "spurious",
    "duplicate",
    "fn",
    "columns",
    "exact_columns",
    "exact_share",
    "mean_abs_col_diff",
    "median_pitch_model",
    "seconds_per_page",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--groups", nargs="*", default=["vatican", "napoli", "overall"])
    args = parser.parse_args(argv)

    variants = args.variants or sorted(
        d.name for d in OUT.iterdir() if d.is_dir() and d.name != "csv" and d.name != "overlays"
    )
    only = set(args.pages) if args.pages else None
    rows = []
    for variant in variants:
        directory = OUT / variant
        if not any(directory.glob("*.json")):
            continue
        try:
            summaries = scorelib.run(
                pred_dir=directory,
                gt_dir=args.gt_dir,
                manifest=MAIN / "scripts/coptic/manifest.json",
                repo_root=MAIN,
                out=OUT / "csv" / f"{variant}.csv",
                quiet=True,
                only=only,
            )
        except SystemExit:
            continue
        for summary in summaries:
            if summary["group"] not in args.groups:
                continue
            summary["variant"] = variant
            summary["scale_multiplier"] = SCALE_MULTIPLIER.get(variant, "")
            rows.append({k: summary.get(k, "") for k in COLUMNS + ["scale_multiplier"]})

    if not rows:
        raise SystemExit("no scored variants")
    widths = {k: max([len(k)] + [len(str(r[k])) for r in rows]) for k in COLUMNS}
    print(" ".join(k.ljust(widths[k]) for k in COLUMNS))
    for row in sorted(rows, key=lambda r: (r["group"], -float(r["f1"] or 0))):
        print(" ".join(str(row[k]).ljust(widths[k]) for k in COLUMNS))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS + ["scale_multiplier"])
            writer.writeheader()
            writer.writerows(rows)

    curve = [r for r in rows if r["group"] == "overall" and r["scale_multiplier"] != ""]
    if curve:
        print("\nV5 scale curve (overall):")
        print("multiplier  pitch_model  F1     recall  precision  fragments  exact_share")
        for row in sorted(curve, key=lambda r: r["scale_multiplier"]):
            pitch = float(row["median_pitch_model"]) * float(row["scale_multiplier"])
            print(
                f"{row['scale_multiplier']:<11} {pitch:<12.1f} {row['f1']:<6} "
                f"{row['recall']:<7} {row['precision']:<10} {row['fragments']:<10} "
                f"{row['exact_share']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
