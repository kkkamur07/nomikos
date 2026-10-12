"""Prepare Coptic PAGE XML exports for TrOCR / Calamari training.

Like Syriac gold, every transcribed page goes to finetuning. There is no
pretraining partition. Line crops are split 80/10/10 per manuscript so
validation and test stay the same size. The same page can contribute lines
to more than one split.

Empty Unicode lines, unlabelled pages, and unreadable polygons are dropped.
Lookalike encodings of the same ink are unified (ASCII period, combining
macron) so the codec does not learn two classes for one glyph. Horizontal
ellipsis is stripped; ASCII colon is kept.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2

from .syriac import crop_polygon, parse_points, save_crop


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "dirty_coptic"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "coptic"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
COPTIC_ROOT = PROCESSED_ROOT / "coptic"
SEED = 1111
PARTITIONS = ("finetuning",)
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
PADDING = 12
MIN_CROP_SIDE = 8
ASCII_FULL_STOP = "."
COPTIC_MIDDLE_DOT = "\u00b7"  # ·, the usual Coptic phrase stop
HORIZONTAL_ELLIPSIS = "\u2026"  # …, editorial damage mark, not page ink
COMBINING_MACRON = "\u0304"  # ̄
COMBINING_CONJOINING_MACRON = "\ufe26"  # ︦, the majority overline in this GT


@dataclass(frozen=True)
class LineAnnotation:
    page_stem: str
    source: str
    text: str
    polygon: list[list[int]]
    y_min: int
    x_min: int


def local_tag(tag: str) -> str:
    """Strip an XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def children(element: ET.Element, name: str) -> list[ET.Element]:
    """Return direct children with the given local tag name."""
    return [child for child in element if local_tag(child.tag) == name]


def descendant(element: ET.Element, *names: str) -> ET.Element | None:
    """Walk a chain of direct children by local tag name."""
    current = element
    for name in names:
        found = next((child for child in current if local_tag(child.tag) == name), None)
        if found is None:
            return None
        current = found
    return current


def slugify(value: str) -> str:
    """Return a stable filename-safe manuscript identifier."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def line_source(custom_attr: str | None) -> str:
    """Read the annotation provenance from the PAGE custom attribute."""
    custom = custom_attr or ""
    if "source:manual" in custom:
        return "manual"
    if "source:kraken" in custom:
        return "kraken"
    return "unknown"


def normalize_coptic_gt(text: str) -> str:
    """Keep page ink and unify duplicate encodings of one glyph.

    ASCII '.' is the same raised phrase stop as middle dot ·. Combining
    macron U+0304 is the same overline as combining conjoining macron
    U+FE26, which is the majority encoding on the nomina sacra.
    Horizontal ellipsis is editorial (broken line ends), not ink; ASCII
    colon stays — it is drawn on the page.
    """
    text = text.replace("\xa0", " ")
    text = text.replace(HORIZONTAL_ELLIPSIS, "")
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace(ASCII_FULL_STOP, COPTIC_MIDDLE_DOT)
    text = text.replace(COMBINING_MACRON, COMBINING_CONJOINING_MACRON)
    return unicodedata.normalize("NFC", text)


def iter_page_lines(xml_path: Path) -> tuple[str, list[LineAnnotation]]:
    """Return the page image filename and transcribed line polygons."""
    root = ET.parse(xml_path).getroot()
    page = next((child for child in root if local_tag(child.tag) == "Page"), None)
    if page is None or not page.get("imageFilename"):
        raise ValueError(f"PAGE XML missing Page/imageFilename: {xml_path}")

    lines: list[LineAnnotation] = []
    for region in children(page, "TextRegion"):
        for line in children(region, "TextLine"):
            text_el = descendant(line, "TextEquiv", "Unicode")
            text = normalize_coptic_gt(text_el.text or "") if text_el is not None else ""
            if not text:
                continue
            coords = next((child for child in line if local_tag(child.tag) == "Coords"), None)
            polygon = parse_points(coords.get("points") if coords is not None else None)
            if len(polygon) < 3:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            lines.append(
                LineAnnotation(
                    page_stem=xml_path.stem,
                    source=line_source(line.get("custom")),
                    text=text,
                    polygon=polygon,
                    y_min=min(ys),
                    x_min=min(xs),
                )
            )

    lines.sort(key=lambda item: (item.y_min, item.x_min))
    return page.get("imageFilename"), lines


def count_empty_lines(xml_path: Path) -> int:
    """Count TextLine elements with no usable transcription."""
    root = ET.parse(xml_path).getroot()
    page = next((child for child in root if local_tag(child.tag) == "Page"), None)
    if page is None:
        return 0
    empty = 0
    for region in children(page, "TextRegion"):
        for line in children(region, "TextLine"):
            text_el = descendant(line, "TextEquiv", "Unicode")
            text = normalize_coptic_gt(text_el.text or "") if text_el is not None else ""
            if not text:
                empty += 1
    return empty


def split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate train/validation/test counts while keeping non-empty splits."""
    train_ratio, val_ratio, test_ratio = ratios
    ratio_total = sum(ratios)
    train = round(total * train_ratio / ratio_total)
    val = round(total * val_ratio / ratio_total)
    test = total - train - val

    if total >= 3:
        counts = [max(1, train), max(1, val), max(1, test)]
        while sum(counts) > total:
            index = max(range(3), key=counts.__getitem__)
            if counts[index] == 1:
                break
            counts[index] -= 1
        while sum(counts) < total:
            counts[0] += 1
        return counts[0], counts[1], counts[2]
    return (max(1, total - 1), 1 if total > 1 else 0, 0)


def split_items(items: list, source_index: int) -> dict[str, list]:
    """Split one manuscript's lines (or pages) 80/10/10 with a stable shuffle."""
    shuffled = list(items)
    random.Random(SEED + source_index).shuffle(shuffled)
    train_count, val_count, test_count = split_counts(len(shuffled), SPLIT_RATIOS)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def write_ground_truth(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    """Write one filename/transcription manifest."""
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for image_name, text in rows:
            output.write(f"{image_name}\t{text}\n")


def parse_ground_truth(path: Path) -> list[tuple[str, str]]:
    """Read and validate one TrOCR manifest."""
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            image_name, text = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid manifest row {path}:{line_number}") from exc
        rows.append((image_name, text))
    return rows


def copy_raw_exports() -> dict[str, Path]:
    """Copy manuscript folders from dirty_coptic into data/raw/coptic."""
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing Coptic source folder: {SOURCE_ROOT}")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    raw_sources: dict[str, Path] = {}
    for source in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir()):
        destination = RAW_ROOT / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        xml_files = sorted(destination.glob("*.xml"))
        if not xml_files:
            raise ValueError(f"No PAGE XML files found in {destination}")
        raw_sources[source.name] = destination
    return raw_sources


def resolve_page_image(source_root: Path, image_name: str) -> Path:
    """Locate the page image named in the PAGE XML."""
    direct = source_root / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem
    for candidate in source_root.iterdir():
        if candidate.is_file() and candidate.stem == stem:
            return candidate
    raise FileNotFoundError(f"Missing page image for {image_name} under {source_root}")


def build_coptic_dataset(raw_sources: dict[str, Path], staging_root: Path) -> dict:
    """Create Coptic line crops and manifests in a staging directory."""
    if staging_root.exists():
        shutil.rmtree(staging_root)
    summary: dict[str, object] = {
        "manuscripts": {},
        "partitions": {},
        "dropped_empty_lines": 0,
        "skipped_unlabelled_pages": [],
        "skipped_invalid_lines": [],
    }
    rows: dict[str, list[tuple[str, str]]] = {split: [] for split in SPLITS}
    image_dir = staging_root / "finetuning" / "image"
    image_dir.mkdir(parents=True, exist_ok=True)

    for source_index, (source_name, source_root) in enumerate(raw_sources.items()):
        xml_files = sorted(source_root.glob("*.xml"))
        source_slug = slugify(source_name)
        source_rows: list[tuple[str, str]] = []
        for xml_path in xml_files:
            summary["dropped_empty_lines"] = int(summary["dropped_empty_lines"]) + count_empty_lines(
                xml_path
            )
            page_image_name, annotations = iter_page_lines(xml_path)
            if not annotations:
                summary["skipped_unlabelled_pages"].append(
                    f"{source_name}/{xml_path.name}"
                )
                continue

            page_image = cv2.imread(
                str(resolve_page_image(source_root, page_image_name)),
                cv2.IMREAD_COLOR,
            )
            if page_image is None:
                raise ValueError(
                    f"Could not read page image: {source_root / page_image_name}"
                )

            for line_index, annotation in enumerate(annotations):
                image_name = (
                    f"{source_slug}__{annotation.page_stem}__{line_index:03d}.jpg"
                )
                try:
                    crop, _ = crop_polygon(
                        page_image,
                        annotation.polygon,
                        PADDING,
                        keep_color=False,
                    )
                except ValueError as exc:
                    summary["skipped_invalid_lines"].append(
                        {
                            "image_name": image_name,
                            "reason": str(exc),
                        }
                    )
                    continue
                if min(crop.shape[:2]) < MIN_CROP_SIDE:
                    summary["skipped_invalid_lines"].append(
                        {
                            "image_name": image_name,
                            "reason": f"crop too small: {crop.shape}",
                        }
                    )
                    continue
                image_path = image_dir / image_name
                if image_path.exists():
                    raise FileExistsError(f"Duplicate Coptic crop name: {image_name}")
                save_crop(image_path, crop, keep_color=False)
                source_rows.append((image_name, annotation.text))

        if len(source_rows) < 3:
            raise ValueError(
                f"{source_name} has fewer than three labelled lines; cannot split "
                "into train, validation, and test."
            )

        assignments = split_items(source_rows, source_index)
        summary["manuscripts"][source_name] = {
            split: len(split_rows) for split, split_rows in assignments.items()
        }
        for split, split_rows in assignments.items():
            rows[split].extend(split_rows)

    partition_root = staging_root / "finetuning"
    for split, manifest_rows in rows.items():
        write_ground_truth(partition_root / f"gt_{split}.txt", manifest_rows)
    summary["partitions"]["finetuning"] = {
        split: len(manifest_rows) for split, manifest_rows in rows.items()
    }
    return summary


def validate_dataset(root: Path) -> dict[str, dict[str, int]]:
    """Ensure every manifest row has an image and no split overlaps."""
    summary: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        partition_root = root / partition
        split_names: dict[str, set[str]] = {}
        summary[partition] = {}
        for split in ("train", "val", "test"):
            rows = parse_ground_truth(partition_root / f"gt_{split}.txt")
            names = {image_name for image_name, _ in rows}
            if len(names) != len(rows):
                raise ValueError(f"Duplicate rows in {partition_root}/gt_{split}.txt")
            for image_name, text in rows:
                if not text.strip():
                    raise ValueError(f"Empty transcription in {partition}/{split}/{image_name}")
                if not (partition_root / "image" / image_name).is_file():
                    raise FileNotFoundError(
                        f"Manifest references missing image: {partition}/{split}/{image_name}"
                    )
            split_names[split] = names
            summary[partition][split] = len(rows)
        if split_names["train"] & split_names["val"]:
            raise ValueError(f"Train/validation overlap under {partition_root}")
        if split_names["train"] & split_names["test"]:
            raise ValueError(f"Train/test overlap under {partition_root}")
        if split_names["val"] & split_names["test"]:
            raise ValueError(f"Validation/test overlap under {partition_root}")
    return summary


def replace_tree(staging: Path, destination: Path) -> None:
    """Replace a dataset tree while retaining a rollback copy during the swap."""
    backup = destination.with_name(f".{destination.name}_backup")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    """Copy raw Coptic pages and write processed line crops plus manifests."""
    raw_sources = copy_raw_exports()
    coptic_staging = PROCESSED_ROOT / ".coptic_staging"
    coptic_summary = build_coptic_dataset(raw_sources, coptic_staging)
    validate_dataset(coptic_staging)
    replace_tree(coptic_staging, COPTIC_ROOT)
    final_coptic = validate_dataset(COPTIC_ROOT)
    report = {
        "raw_root": str(RAW_ROOT),
        "source_root": str(SOURCE_ROOT),
        "coptic": coptic_summary,
        "validated_coptic": final_coptic,
    }
    (COPTIC_ROOT / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
