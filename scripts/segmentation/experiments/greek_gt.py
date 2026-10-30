#!/usr/bin/env python
"""Greek (Grec 1360) ground truth: PAGE XML in, the scorers' snapshot JSON out.

    greek_gt.py convert --out-dir <recon dir>
    greek_gt.py score --pred-dir out/greek-v0 --gt-dir <recon dir> [--view all|page1|manual]

Why this file exists
--------------------
The Coptic ground truth arrives as ``<slug>-latest.json`` recon snapshots and
``score.py``/``score_polygons.py`` read it through ``gtlib.load_all``. The Greek
ground truth arrives as five PAGE XML files instead, so this converts them into
the same snapshot shape and then feeds the same two scorers, unchanged, by
swapping in a loader. Nothing here changes what the Coptic runs produce.

Two traps in the source files, both handled here
------------------------------------------------
1. The ``Baseline`` element is a verbatim copy of the ``Coords`` polygon (a known
   exporter bug), so it is IGNORED. A baseline is derived from the polygon by
   ``derive_baseline`` from
   ``nomikos/backend/annotation/application/page_xml_geometry.py``
   (the fixed exporter's own function: it follows the polygon and sits at
   BASELINE_HEIGHT_FRACTION = 0.68 of the local height).
2. ``custom`` is the OLD "source:x; kind:y" format, not Transkribus grammar, so
   it is parsed here with a split rather than by kraken's parser.

What "ground truth" means per page
----------------------------------
Only page 1 is fully human (23 of 23 manual). Page 2 is raw kraken output with
no correction at all. Pages 3 to 5 are mostly raw kraken with 15, 13 and 3
manually corrected lines. So:

* ``--view page1``: the only accuracy measurement on this manuscript.
* ``--view manual``: every manual line across the five pages is scored, every
  kraken-sourced line is marked unpaired, which makes it DON'T CARE in both
  scorers. Predictions landing on a kraken-sourced line are dropped rather than
  counted against the model, so this is a RECALL measurement on human lines;
  precision on this view is not meaningful.
* ``--view all``: agreement with mostly uncorrected kraken output. It measures
  how kraken-like a model is, not how correct it is.

The per-folio count check is the Greek analogue of the Coptic RTF column counts,
except that no transcription docx exists under ``data/dataset/grec1360/`` (the
whole checkout was searched), so the per-folio expected counts are taken from
the ground-truth XML itself and are only independent on page 1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "nomikos"))

import gtlib  # noqa: E402

from backend.annotation.application.page_xml_geometry import derive_baseline  # noqa: E402

NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}
# data/dataset/ is not tracked; set SEG_DATA_ROOT when running from a worktree.
DATA_ROOT = gtlib.data_root()
XML_DIR = DATA_ROOT / "data/dataset/grec1360/xml"
IMAGE_DIR = DATA_ROOT / "data/dataset/grec1360/pages"
# Two folios per image, split near this x. See the grec1360 spread note.
FOLIO_SPLIT = 725
FOLIO_NAMES = ("L", "R")


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def parse_points(raw: str) -> list[list[float]]:
    out = []
    for pair in raw.split():
        x, _, y = pair.partition(",")
        out.append([float(x), float(y)])
    return out


def parse_custom(raw: str | None) -> dict[str, str]:
    """'source:manual; kind:polygon' -> {'source': 'manual', 'kind': 'polygon'}."""
    fields: dict[str, str] = {}
    for part in (raw or "").split(";"):
        key, _, value = part.strip().partition(":")
        if key and value:
            fields[key.strip()] = value.strip()
    return fields


def convert_one(xml_path: Path) -> dict[str, Any]:
    # S314: these five files are our own exporter's output, read from the local
    # checkout, and the brief requires ElementTree rather than kraken's parser
    # because the custom attribute is in the old "source:x; kind:y" format.
    root = ET.parse(xml_path).getroot()  # noqa: S314
    page = root.find("p:Page", NS)
    if page is None:
        raise SystemExit(f"{xml_path}: no Page element")
    lines = []
    for order, el in enumerate(page.findall(".//p:TextLine", NS)):
        coords = el.find("p:Coords", NS)
        if coords is None or not coords.get("points"):
            continue
        polygon = parse_points(coords.get("points", ""))
        custom = parse_custom(el.get("custom"))
        text_el = el.find(".//p:Unicode", NS)
        lines.append(
            {
                "id": el.get("id") or f"_{uuid.uuid4()}",
                "order": order,
                "points": polygon,
                # The file's own Baseline is a copy of Coords, so it is dropped
                # and the exporter's derivation is used instead.
                "baseline": {"points": derive_baseline(polygon), "derived": True},
                "source": custom.get("source", "unknown"),
                "kind": custom.get("kind", ""),
                "text": (text_el.text or "") if text_el is not None else "",
            }
        )
    return {
        "page": {
            "slug": "grec-p" + xml_path.stem.removeprefix("page"),
            "image_filename": page.get("imageFilename", ""),
            "width": int(page.get("imageWidth", 0)),
            "height": int(page.get("imageHeight", 0)),
        },
        # Every line is listed as paired; the loader decides what counts, so the
        # same snapshot serves the "all" and "manual" views.
        "pairing": {"text_lines": [{"paired_line_id": ln["id"]} for ln in lines]},
        "lines": lines,
    }


def convert(out_dir: Path, xml_dir: Path = XML_DIR) -> list[Path]:
    gtlib.require_file(xml_dir, "Grec 1360 PAGE XML directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for xml_path in sorted(xml_dir.glob("page*.xml")):
        data = convert_one(xml_path)
        slug = data["page"]["slug"]
        path = out_dir / f"{slug}-latest.json"
        path.write_text(json.dumps(data))
        written.append(path)
        kinds: dict[str, int] = {}
        for ln in data["lines"]:
            kinds[ln["source"]] = kinds.get(ln["source"], 0) + 1
        print(
            f"{slug}: {len(data['lines'])} lines "
            f"({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}) -> {path}"
        )
    return written


# --------------------------------------------------------------------------
# loading, as gtlib.GTPage so score.py and score_polygons.py work unchanged
# --------------------------------------------------------------------------
def resolve_image(filename: str, image_dir: Path, converted_dir: Path | None) -> Path:
    if converted_dir is not None:
        png = converted_dir / (Path(filename).stem + ".png")
        if png.exists():
            return png
    return image_dir / filename


def load_greek_page(
    path: Path,
    *,
    manual_only: bool,
    image_dir: Path = IMAGE_DIR,
    converted_dir: Path | None = None,
) -> gtlib.GTPage:
    data = json.loads(path.read_text())
    meta = data["page"]
    lines = [
        gtlib.GTLine(
            baseline=[[float(x), float(y)] for x, y in ln["baseline"]["points"]],
            polygon=[[float(x), float(y)] for x, y in ln["points"]],
            paired=(ln["source"] == "manual") if manual_only else True,
        )
        for ln in sorted(data["lines"], key=lambda ln: ln["order"])
    ]
    for ln in lines:
        xs = [p[0] for p in ln.polygon]
        ln.column = 0 if (min(xs) + max(xs)) / 2.0 < FOLIO_SPLIT else 1

    gt = gtlib.GTPage(
        slug=meta["slug"],
        label=meta["slug"],
        manuscript="grec1360",
        image_path=resolve_image(meta["image_filename"], image_dir, converted_dir),
        width=meta["width"],
        height=meta["height"],
        lines=lines,
        ncols=2,
        column_names=FOLIO_NAMES,
        # Folio centres, so column_of_x splits a prediction at the same place.
        centres=[FOLIO_SPLIT / 2.0, FOLIO_SPLIT + (meta["width"] - FOLIO_SPLIT) / 2.0],
        # No transcription docx exists for this manuscript, so the expected
        # per-folio counts come from the XML itself. Independent on page 1 only.
        rtf_counts=[sum(1 for ln in lines if ln.column == c and ln.paired) for c in range(2)],
        text_lines=len(lines),
    )
    gt.pitch_by_column = [_pitch(gt, c) for c in range(2)]
    return gt


def _pitch(gt: gtlib.GTPage, column: int) -> float:
    ys = sorted(
        statistics.median(p[1] for p in ln.baseline) for ln in gt.lines if ln.column == column
    )
    gaps = sorted(b - a for a, b in zip(ys, ys[1:], strict=False) if b - a > 0)
    return statistics.median(gaps) if gaps else 0.0


def loader(*, gt_dir: Path, manual_only: bool, converted_dir: Path | None = None):
    """A drop-in replacement for ``gtlib.load_all``, ignoring the manifest."""

    def load_all(*, gt_dir: Path = gt_dir, manifest_path=None, repo_root=None):  # noqa: ARG001
        return [
            load_greek_page(p, manual_only=manual_only, converted_dir=converted_dir)
            for p in sorted(Path(gt_dir).glob("grec-p*-latest.json"))
        ]

    return load_all


VIEWS = {
    "page1": (False, {"grec-p1"}),
    "manual": (True, None),
    "all": (False, None),
}


def folio_counts(pred_dir: Path, gts: list[gtlib.GTPage]) -> list[tuple[str, int, int, int, int]]:
    rows = []
    for gt in gts:
        path = pred_dir / f"{gt.slug}.json"
        if not path.exists():
            continue
        pred = json.loads(path.read_text())
        counts = [0, 0]
        for ln in pred["lines"]:
            bl = ln.get("baseline") or []
            if len(bl) < 2:
                continue
            xs = [p[0] for p in bl]
            counts[gt.column_of_x(sum(xs) / len(xs))] += 1
        rows.append((gt.slug, counts[0], gt.rtf_counts[0], counts[1], gt.rtf_counts[1]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    conv = sub.add_parser("convert", help="PAGE XML -> recon snapshots")
    conv.add_argument("--xml-dir", type=Path, default=XML_DIR)
    conv.add_argument("--out-dir", required=True, type=Path)

    sc = sub.add_parser("score", help="run score.py and score_polygons.py on a Greek view")
    sc.add_argument("--pred-dir", required=True, type=Path)
    sc.add_argument("--gt-dir", required=True, type=Path)
    sc.add_argument("--view", choices=sorted(VIEWS), default="all")
    sc.add_argument("--converted-dir", type=Path, default=None)
    sc.add_argument("--polygons", action="store_true", help="also run score_polygons.py")

    args = parser.parse_args(argv)
    if args.cmd == "convert":
        convert(args.out_dir, args.xml_dir)
        return 0

    import score
    import score_polygons

    manual_only, only = VIEWS[args.view]
    load_all = loader(gt_dir=args.gt_dir, manual_only=manual_only, converted_dir=args.converted_dir)
    gtlib.load_all = load_all  # score.py and score_polygons.py both call gtlib.load_all
    score_polygons.gtlib.load_all = load_all
    print(f"### view={args.view}  pred={args.pred_dir}")
    if args.view == "manual":
        print(
            "### manual view: kraken-sourced ground-truth lines are DON'T CARE, so "
            "read recall only; precision and the false-positive columns are not meaningful."
        )
    if args.view == "all":
        print(
            "### all view: pages 2 to 5 are mostly UNCORRECTED kraken output, so this is "
            "agreement with kraken, not accuracy."
        )
    score.run(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        manifest=Path("unused-for-greek"),
        repo_root=DATA_ROOT,
        out=None,
        only=only,
    )
    if args.polygons:
        score_polygons.run(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            manifest=Path("unused-for-greek"),
            repo_root=DATA_ROOT,
            out=None,
            only=only,
        )
    gts = load_all(gt_dir=args.gt_dir)
    if only:
        gts = [g for g in gts if g.slug in only]
    print("per folio (predicted vs ground truth): slug | L pred | L gt | R pred | R gt")
    for row in folio_counts(args.pred_dir, gts):
        print("   " + " | ".join(str(v) for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
