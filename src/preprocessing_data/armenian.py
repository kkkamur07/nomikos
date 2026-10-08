"""Prepare Armenian PAGE XML exports for TrOCR training.

Each manuscript contributes 70% of its pages to pretraining and 30% to
finetuning. Pages in each partition are further divided into train, validation,
and test sets so line crops from one page never leak across splits.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import cv2

try:  # python -m src.preprocessing_data.<module>
    from .syriac import PAGE_NS, crop_polygon, iter_page_lines, save_crop
except ImportError:  # python src/preprocessing_data/<module>.py
    from syriac import PAGE_NS, crop_polygon, iter_page_lines, save_crop


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    "MS-P-331-ff24r-34r",
    "MS-UCLA-Ms-72-ff64v-73v",
    "Ms_P_172-CanonsDvin719-Partaw768-Dvin644-Karin_complete",
)
RAW_ROOT = REPO_ROOT / "data" / "raw" / "armenian"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
ARMENIAN_ROOT = PROCESSED_ROOT / "armenian"
COMBINED_ROOT = PROCESSED_ROOT / "combined"
SEED = 1111
PRETRAINING_RATIO = 0.7
SPLIT_RATIOS = (0.8, 0.1, 0.1)
PADDING = 12
DROP_CROPS = frozenset(
    {
        "ms_ucla_ms__0011__044.jpg",  # scratch labelled է
        "ms_ucla_ms__0011__046.jpg",  # gutter box labelled գ
        "ms_ucla_ms__0011__047.jpg",  # gutter box labelled դ
        "ms_ucla_ms__0011__048.jpg",  # gutter box labelled մ
    }
)
ASCII_FULL_STOP = "."
ASCII_COLON = ":"
ARMENIAN_MIJAKET = "\u2024"  # ․
ARMENIAN_VERJAKET = "\u0589"  # ։


ARMENIAN_ABBREVIATION_MARK = "\u055f"  # ՟ pativ / titlo
_BRACKET_EXPANSION = re.compile(r"\[[^\[\]]*\]")
_ARMENIAN_LETTER = re.compile(r"[\u0531-\u0587և]")


def normalize_text(text: str) -> str:
    """NFC-normalize transcription text, matching replace_armenian_ms.py."""
    return unicodedata.normalize("NFC", text)


ARMENIAN_HYPHEN = "\u058a"  # ֊
ARMENIAN_ECH_YIWN = "\u0587"  # և
MULTIPLE_SPACES = re.compile(r" {2,}")
TRAILING_SPACES = re.compile(r" +$")
# Manifest rules applied on read after NFC, in order: (report name, pattern,
# replacement). Parentheses and quotes are drawn in the ink and stay.
INK_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # The dash is drawn; the manuscripts already use ֊ for the same stroke.
    ("- (U+002D) -> ֊ (U+058A)", re.compile("-"), ARMENIAN_HYPHEN),
    # Same ligature in the ink. Capital Եւ has no single codepoint and stays.
    ("եւ -> և (U+0587)", re.compile("\u0565\u0582"), ARMENIAN_ECH_YIWN),
    ("double space -> single", MULTIPLE_SPACES, " "),
    ("trailing spaces -> removed", TRAILING_SPACES, ""),
)


def apply_ink_rules(text: str) -> tuple[str, dict[str, int]]:
    """Apply ``INK_RULES``; return the text and the per-rule substitution counts."""
    counts: dict[str, int] = {}
    for name, pattern, replacement in INK_RULES:
        text, count = pattern.subn(replacement, text)
        if count:
            counts[name] = count
    return text, counts


def normalize_manifest_text(text: str) -> str:
    """NFC, then ``INK_RULES``: the form resplit writes to the gt_*.txt manifests."""
    return apply_ink_rules(normalize_text(text))[0]


def normalize_print_punctuation(text: str) -> str:
    """Map Western print stops onto the Armenian marks they look like.

    ASCII '.' is one dot, so it becomes միջակետ ․. ASCII ':' is two dots,
    so it becomes վերջակետ ։. Native Armenian punctuation is left as-is.
    """
    return text.replace(ASCII_FULL_STOP, ARMENIAN_MIJAKET).replace(ASCII_COLON, ARMENIAN_VERJAKET)


def normalize_latin_typos(text: str) -> str:
    """Map leftover Latin keys onto the Armenian letters in the ink.

    On MS_P_172 nomina sacra, q is ի (tall stem plus the abbreviation bar),
    t is յ (bar sitting on Yi), and b is a stray keystroke inside յի.
    Dotless ı is ա in հանապազ and ի in տի. ASCII & is և except զիդ,
    trailing @ is գ, line-final / is ։, and | is a separator to drop.
    """
    text = text.replace(
        "արոզութին․ ևզաղաւթսն զհետ բերել, զոր ի նոյն խորհուրդ արարեալ է "
        "երանելւոյն Յովհաննու Մանդակունւոյ Հայոց կաթողիկոսի՝ | Քարոզութիւն, "
        "Զարթուցեալքս, Աղաւթք, Զքէն գոհանամք և զհանգիստ նորին՝ Նայեա՛ Տէր "
        "ողորմութեամբ։",
        "արոզութին․ ևզաղաւթսն զհետ",
    )
    return (
        text.replace("յqս", "յիս")
        .replace("յqիցին", "յիսիցին")
        .replace("յbի", "յի")
        .replace("tիս", "յիս")
        .replace("հանապıզ", "հանապազ")
        .replace("տı", "տի")
        .replace("&իս", "ևիս")
        .replace("&իդ", "զիդ")
        .replace("․ @", "․ գ")
        .replace("ասի/", "ասի։")
        .replace("այլըստ|ողորմու", "այլըստողորմու")
    )


def to_diplomatic_gt(text: str) -> str:
    """Keep page ink, drop Calfa restorations inside [brackets].

    Letters in [...] were never drawn; a pativ/titlo bar usually was. The bar
    is encoded as U+055F after the last visible letter of that word. The
    ideogram ~ already *is* the drawing for աշխարհ, so ~[աշխարհ] becomes ~
    with no extra mark.
    """
    text = text.replace("{", "[")
    pieces: list[str] = []
    for part in re.split(r"(\s+)", text):
        if not part or part.isspace():
            pieces.append(part)
            continue
        had_brackets = "[" in part or "]" in part
        had_ideogram = "~" in part
        part = _BRACKET_EXPANSION.sub("", part)
        part = part.replace("[", "").replace("]", "")
        if had_brackets and not had_ideogram:
            letters = list(_ARMENIAN_LETTER.finditer(part))
            if letters:
                index = letters[-1].end()
                part = part[:index] + ARMENIAN_ABBREVIATION_MARK + part[index:]
        pieces.append(part)
    return "".join(pieces)


def slugify(value: str) -> str:
    """Return a stable filename-safe manuscript identifier."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def page_metadata(xml_path: Path) -> tuple[str, str]:
    """Return the PAGE image filename and Transkribus download URL."""
    root = ET.parse(xml_path).getroot()
    page = root.find("page:Page", PAGE_NS)
    metadata = root.find(".//page:TranskribusMetadata", PAGE_NS)
    if page is None or not page.get("imageFilename"):
        raise ValueError(f"PAGE XML has no image filename: {xml_path}")
    if metadata is None or not metadata.get("imgUrl"):
        raise ValueError(f"PAGE XML has no Transkribus image URL: {xml_path}")
    return page.get("imageFilename"), metadata.get("imgUrl")


def download_file(url: str, destination: Path) -> None:
    """Download one file atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "greekocr-data-preparation"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def copy_raw_exports() -> dict[str, Path]:
    """Copy source exports into data/raw and download their missing images."""
    raw_sources: dict[str, Path] = {}
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for source_name in SOURCE_NAMES:
        source = REPO_ROOT / source_name
        destination = RAW_ROOT / source_name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif not destination.is_dir():
            raise FileNotFoundError(
                f"Missing Armenian source folder in both {source} and {destination}"
            )
        xml_files = sorted((destination / "page").glob("*.xml"))
        if not xml_files:
            raise ValueError(f"No PAGE XML files found in {destination}")

        for xml_path in xml_files:
            image_name, image_url = page_metadata(xml_path)
            image_path = destination / image_name
            if not image_path.exists():
                download_file(image_url, image_path)
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise ValueError(f"Downloaded page image is unreadable: {image_path}")
        raw_sources[source_name] = destination
    return raw_sources


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
        return tuple(counts)
    return (max(1, total - 1), 1 if total > 1 else 0, 0)


def partition_pages(xml_files: list[Path], source_index: int) -> dict[str, dict[str, list[Path]]]:
    """Split one manuscript into 70/30 partitions and train/val/test subsets."""
    shuffled = list(xml_files)
    random.Random(SEED + source_index).shuffle(shuffled)
    pretraining_count = round(len(shuffled) * PRETRAINING_RATIO)
    pretraining_count = min(max(1, pretraining_count), len(shuffled) - 1)
    partitions = {
        "pretraining": shuffled[:pretraining_count],
        "finetuning": shuffled[pretraining_count:],
    }

    result: dict[str, dict[str, list[Path]]] = {}
    for partition_name, partition_files in partitions.items():
        train_count, val_count, test_count = split_counts(
            len(partition_files),
            SPLIT_RATIOS,
        )
        result[partition_name] = {
            "train": partition_files[:train_count],
            "val": partition_files[train_count : train_count + val_count],
            "test": partition_files[train_count + val_count : train_count + val_count + test_count],
        }
    return result


def write_ground_truth(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    """Write one TrOCR filename/transcription manifest."""
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for image_name, text in rows:
            output.write(f"{image_name}\t{text}\n")


def build_armenian_dataset(raw_sources: dict[str, Path], staging_root: Path) -> dict:
    """Create Armenian line crops and manifests in a staging directory."""
    if staging_root.exists():
        shutil.rmtree(staging_root)
    summary: dict[str, object] = {"manuscripts": {}, "partitions": {}}
    rows: dict[str, dict[str, list[tuple[str, str]]]] = {
        partition: {split: [] for split in ("train", "val", "test")}
        for partition in ("pretraining", "finetuning")
    }

    for source_index, (source_name, source_root) in enumerate(raw_sources.items()):
        xml_files = sorted((source_root / "page").glob("*.xml"))
        assignments = partition_pages(xml_files, source_index)
        manuscript_summary = {
            partition: {split: len(split_files) for split, split_files in split_map.items()}
            for partition, split_map in assignments.items()
        }
        summary["manuscripts"][source_name] = manuscript_summary
        source_slug = slugify(source_name)

        for partition, split_map in assignments.items():
            image_dir = staging_root / partition / "image"
            image_dir.mkdir(parents=True, exist_ok=True)
            for split, split_files in split_map.items():
                for xml_path in split_files:
                    page_image_name, annotations = iter_page_lines(xml_path)
                    page_image = cv2.imread(
                        str(source_root / page_image_name),
                        cv2.IMREAD_COLOR,
                    )
                    if page_image is None:
                        raise ValueError(
                            f"Could not read page image: {source_root / page_image_name}"
                        )

                    for annotation in annotations:
                        image_name = (
                            f"{source_slug}__{annotation.page_stem}"
                            f"__{annotation.line_index:03d}.jpg"
                        )
                        if image_name in DROP_CROPS:
                            continue
                        crop, _ = crop_polygon(
                            page_image,
                            annotation.polygon,
                            PADDING,
                            keep_color=False,
                        )
                        image_path = image_dir / image_name
                        if image_path.exists():
                            raise FileExistsError(f"Duplicate Armenian crop name: {image_name}")
                        save_crop(image_path, crop, keep_color=False)
                        rows[partition][split].append(
                            (
                                image_name,
                                to_diplomatic_gt(
                                    normalize_latin_typos(
                                        normalize_print_punctuation(
                                            normalize_text(annotation.text)
                                        )
                                    )
                                ),
                            )
                        )

    for partition, split_rows in rows.items():
        partition_root = staging_root / partition
        for split, manifest_rows in split_rows.items():
            write_ground_truth(partition_root / f"gt_{split}.txt", manifest_rows)
        summary["partitions"][partition] = {
            split: len(manifest_rows) for split, manifest_rows in split_rows.items()
        }
    return summary


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


def hardlink_or_copy(source: Path, destination: Path) -> None:
    """Hardlink a crop when possible, falling back to a copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_combined_dataset(armenian_staging: Path, combined_staging: Path) -> dict:
    """Rebuild combined pretraining and finetuning datasets from each language."""
    if combined_staging.exists():
        shutil.rmtree(combined_staging)
    language_roots = {
        "greek": PROCESSED_ROOT / "greek",
        "syriac": PROCESSED_ROOT / "syriac",
        "armenian": armenian_staging,
    }
    summary: dict[str, dict[str, int]] = {}

    for partition in ("pretraining", "finetuning"):
        partition_root = combined_staging / partition
        image_dir = partition_root / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        partition_summary: dict[str, int] = {}

        for split in ("train", "val", "test"):
            combined_rows: list[tuple[str, str]] = []
            seen_names: set[str] = set()
            for language, language_root in language_roots.items():
                source_root = language_root / partition
                manifest = source_root / f"gt_{split}.txt"
                if not manifest.exists():
                    raise FileNotFoundError(f"Missing {language} manifest: {manifest}")
                for image_name, text in parse_ground_truth(manifest):
                    if image_name in seen_names:
                        raise ValueError(
                            f"Duplicate image name in combined {partition}/{split}: {image_name}"
                        )
                    seen_names.add(image_name)
                    source_image = source_root / "image" / image_name
                    if not source_image.is_file():
                        raise FileNotFoundError(f"Missing source crop: {source_image}")
                    hardlink_or_copy(source_image, image_dir / image_name)
                    combined_rows.append((image_name, text))
            write_ground_truth(partition_root / f"gt_{split}.txt", combined_rows)
            partition_summary[split] = len(combined_rows)
        summary[partition] = partition_summary
    return summary


def validate_dataset(root: Path) -> dict[str, dict[str, int]]:
    """Ensure every manifest row has an image and no split overlaps."""
    summary: dict[str, dict[str, int]] = {}
    for partition in ("pretraining", "finetuning"):
        partition_root = root / partition
        split_names: dict[str, set[str]] = {}
        summary[partition] = {}
        for split in ("train", "val", "test"):
            rows = parse_ground_truth(partition_root / f"gt_{split}.txt")
            names = {image_name for image_name, _ in rows}
            if len(names) != len(rows):
                raise ValueError(f"Duplicate rows in {partition_root}/gt_{split}.txt")
            for image_name in names:
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
    """Run the complete Armenian migration and remove validated source copies."""
    raw_sources = copy_raw_exports()
    armenian_staging = PROCESSED_ROOT / ".armenian_staging"
    combined_staging = PROCESSED_ROOT / ".combined_staging"

    armenian_summary = build_armenian_dataset(raw_sources, armenian_staging)
    combined_summary = build_combined_dataset(armenian_staging, combined_staging)
    validate_dataset(armenian_staging)
    validate_dataset(combined_staging)

    replace_tree(armenian_staging, ARMENIAN_ROOT)
    replace_tree(combined_staging, COMBINED_ROOT)
    final_armenian = validate_dataset(ARMENIAN_ROOT)
    final_combined = validate_dataset(COMBINED_ROOT)

    for source_name in SOURCE_NAMES:
        source = REPO_ROOT / source_name
        if source.exists():
            shutil.rmtree(source)

    summary = {
        "raw_root": str(RAW_ROOT),
        "armenian": armenian_summary,
        "combined": combined_summary,
        "validated_armenian": final_armenian,
        "validated_combined": final_combined,
        "removed_root_sources": list(SOURCE_NAMES),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
