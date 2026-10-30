"""Ground truth loading, column derivation and RTF line counts.

The column logic and the RTF blank-line heuristic are ported from
``human-guide/v4/tools/common.py`` (shipped in
``scripts/coptic/pairing-tools-2026-09-18.tar.gz``) so the experiments count
columns the same way the pairing tools do.

Nothing in here knows anything about kraken: it reads a manifest, a directory
of recon JSON files and the RTF transcriptions, and that is all.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Code-owned files (registry.yaml, nomikos/) resolve from this checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Data the repo does not track (data/dataset/, scripts/coptic/manifest.json) lives
# under a data root: --data-root, else $SEG_DATA_ROOT, else this checkout.
DATA_ROOT_ENV = "SEG_DATA_ROOT"
COPTIC_GT_ENV = "COPTIC_GT_DIR"
GREEK_GT_ENV = "GREEK_GT_DIR"

COLS4 = ("LA", "LB", "RA", "RB")
COLS2 = ("L", "R")

# From human-guide/v4/tools/common.py: human rulings about RTF blank-line
# breaks the heuristic misplaces.
RTF_RANGE_OVERRIDES = {
    ("vatican", "8v"): "merge-trailing-singleton-into-last-column",
}


def data_root(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get(DATA_ROOT_ENV)
    return Path(env) if env else REPO_ROOT


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{what} not found at {path}. It is not tracked by git: pass --data-root or set "
            f"{DATA_ROOT_ENV} to the checkout that holds it."
        )
    return path


def coptic_manifest(root: Path) -> Path:
    return require_file(root / "scripts/coptic/manifest.json", "Coptic manifest")


def env_path(name: str) -> Path | None:
    """Default for a --gt-dir style flag: the env var if set, else None."""
    env = os.environ.get(name)
    return Path(env) if env else None


def require_gt_dir(gt_dir: Path | None, env: str) -> Path:
    if gt_dir is None:
        raise SystemExit(f"ground truth directory missing: pass --gt-dir or set {env}")
    if not gt_dir.is_dir():
        raise SystemExit(f"ground truth directory {gt_dir} does not exist")
    return gt_dir


def out_slug(label: str) -> str:
    """'c 10' -> 'c10', '1r' -> 'vat-1r'."""
    if label.startswith("c "):
        return label.replace(" ", "")
    return "vat-" + label


def ms_of_label(label: str) -> str:
    return "napoli" if label.startswith("c ") else "vatican"


def kmeans_1d(values: list[float], k: int, iters: int = 200) -> list[float]:
    xs = sorted(values)
    if len(xs) <= k:
        return xs + [xs[-1]] * (k - len(xs))
    centres = [xs[int((i + 0.5) * len(xs) / k)] for i in range(k)]
    for _ in range(iters):
        groups: list[list[float]] = [[] for _ in range(k)]
        for x in xs:
            groups[min(range(k), key=lambda i: abs(x - centres[i]))].append(x)
        nxt = [statistics.fmean(g) if g else centres[i] for i, g in enumerate(groups)]
        if nxt == centres:
            break
        centres = nxt
    return sorted(centres)


def rtf_column_counts(rtf_path: Path, ms: str, label: str) -> list[int]:
    """Transcription lines per column, from the RTF's blank-line breaks."""
    out = subprocess.run(  # noqa: S603
        ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(rtf_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ranges: list[tuple[int, int]] = []
    start = kept = 0
    brk = False
    for line in out.splitlines():
        if not line.strip():
            brk = True
            continue
        if brk and kept > start:
            ranges.append((start, kept))
            start = kept
        brk = False
        kept += 1
    ranges.append((start, kept))
    if (
        RTF_RANGE_OVERRIDES.get((ms, label)) == "merge-trailing-singleton-into-last-column"
        and len(ranges) > 1
        and ranges[-1][1] - ranges[-1][0] == 1
    ):
        ranges = ranges[:-2] + [(ranges[-2][0], ranges[-1][1])]
    return [b - a for a, b in ranges]


@dataclass
class GTLine:
    baseline: list[list[float]]
    polygon: list[list[float]]
    paired: bool
    column: int = -1


@dataclass
class GTPage:
    slug: str
    label: str
    manuscript: str
    image_path: Path
    width: int
    height: int
    lines: list[GTLine]
    ncols: int
    column_names: tuple[str, ...]
    centres: list[float]
    rtf_counts: list[int]
    text_lines: int
    pitch_by_column: list[float] = field(default_factory=list)

    @property
    def paired_lines(self) -> list[GTLine]:
        return [ln for ln in self.lines if ln.paired]

    @property
    def scale(self) -> float:
        """Model pixels per source pixel at the production 1800 px height."""
        return 1800.0 / self.height

    def pitch_for(self, column: int) -> float:
        if 0 <= column < len(self.pitch_by_column) and self.pitch_by_column[column] > 0:
            return self.pitch_by_column[column]
        good = [p for p in self.pitch_by_column if p > 0]
        return statistics.median(good) if good else 30.0

    def column_of_x(self, x: float) -> int:
        return min(range(self.ncols), key=lambda i: abs(x - self.centres[i]))


def _median_y(points: list[list[float]]) -> float:
    return statistics.median(p[1] for p in points)


def _xc(polygon: list[list[float]]) -> float:
    xs = [p[0] for p in polygon]
    return (min(xs) + max(xs)) / 2.0


def load_gt_page(
    page: dict[str, Any],
    *,
    gt_dir: Path,
    repo_root: Path,
) -> GTPage | None:
    label = page["label"]
    slug = out_slug(label)
    recon = gt_dir / f"{slug}-latest.json"
    if not recon.exists():
        return None
    data = json.loads(recon.read_text())
    paired_ids = {
        t["paired_line_id"] for t in data["pairing"]["text_lines"] if t.get("paired_line_id")
    }
    from PIL import Image

    image_path = repo_root / page["image_path"]
    with Image.open(image_path) as im:
        width, height = im.size

    lines = [
        GTLine(
            baseline=[[float(x), float(y)] for x, y in ln["baseline"]["points"]],
            polygon=[[float(x), float(y)] for x, y in ln["points"]],
            paired=ln["id"] in paired_ids,
        )
        for ln in sorted(data["lines"], key=lambda l: l["order"])
        if ln.get("points") and (ln.get("baseline") or {}).get("points")
    ]

    ncols = 4 if width / height > 1.2 else 2
    names = COLS4 if ncols == 4 else COLS2
    centres = kmeans_1d([_xc(ln.polygon) for ln in lines], ncols)
    for ln in lines:
        ln.column = min(range(ncols), key=lambda i: abs(_xc(ln.polygon) - centres[i]))

    ms = ms_of_label(label)
    rtf_counts = rtf_column_counts(repo_root / page["rtf_path"], ms, label)

    gt = GTPage(
        slug=slug,
        label=label,
        manuscript=ms,
        image_path=image_path,
        width=width,
        height=height,
        lines=lines,
        ncols=ncols,
        column_names=names,
        centres=centres,
        rtf_counts=rtf_counts,
        text_lines=page["text_lines"],
    )
    gt.pitch_by_column = [_column_pitch(gt, c) for c in range(ncols)]
    return gt


def _column_pitch(gt: GTPage, column: int) -> float:
    ys = sorted(_median_y(ln.baseline) for ln in gt.lines if ln.column == column)
    gaps = [b - a for a, b in zip(ys, ys[1:], strict=False) if b - a > 0]
    if not gaps:
        return 0.0
    # Trim the tail: a column break or a deleted line leaves one huge gap.
    gaps.sort()
    return statistics.median(gaps)


def line_height(polygon: list[list[float]]) -> float:
    ys = [p[1] for p in polygon]
    return max(ys) - min(ys)


def load_manifest_pages(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    pages = []
    for ms in ("vatican", "napoli"):
        for page in manifest[ms]["pages"]:
            if page.get("text_lines") and page.get("rtf_path"):
                pages.append(page)
    return pages


def load_all(*, gt_dir: Path, manifest_path: Path, repo_root: Path) -> list[GTPage]:
    out = []
    for page in load_manifest_pages(manifest_path):
        gt = load_gt_page(page, gt_dir=gt_dir, repo_root=repo_root)
        if gt is not None:
            out.append(gt)
    return out
