"""
Export PAGE XML line annotations into a line-crop dataset for OCR training.

This script reads a Transkribus/eScriptorium-style folder containing page images
beside a `page/` directory with PAGE XML files, then writes:

    OUT_ROOT/
      images/{train,val,test}/*.png
      labels/{train,val,test}/*.gt.txt
      manifests/{train,val,test}.jsonl
      summary.json

The `images/` + `labels/` layout can be flattened into a Calamari pack with
`model/preprocessing/prepare_calamari_data.py`.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
READING_ORDER_RE = re.compile(r"readingOrder\s*\{\s*index:(\d+);?\s*\}")

# Edit these values directly before running the script.
SOURCE_DIR = REPO_ROOT / "SMMJ_00036"  #! change the path run the script
OUTPUT_ROOT = None
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
IMAGE_FORMAT = "jpg"
PADDING = 12
KEEP_COLOR = False
OVERWRITE_OUTPUT = False


@dataclass(frozen=True)
class LineAnnotation:
    page_stem: str
    page_image: str
    xml_file: str
    region_id: str
    line_id: str
    line_index: int
    text: str
    polygon: list[list[int]]
    baseline: list[list[int]]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def parse_points(raw: str | None) -> list[list[int]]:
    if not raw:
        return []
    points: list[list[int]] = []
    for pair in raw.split():
        x_str, y_str = pair.split(",", 1)
        points.append([int(round(float(x_str))), int(round(float(y_str)))])
    return points


ASCII_FULL_STOP = "."
ASCII_COLON = ":"
SYRIAC_SUPRALINEAR_FULL_STOP = "\u0701"  # ܁
SYRIAC_SUPRALINEAR_COLON = "\u0703"  # ܃
SYRIAC_QUSHSHAYA = "\u0741"  # ݁
SYRIAC_RUKKAKHA = "\u0742"  # ݂
COMBINING_DOT_ABOVE = "\u0307"  # ̇
COMBINING_DOT_BELOW = "\u0323"  # ̣
ARABIC_TATWEEL = "\u0640"  # ـ, kashida; not Syriac ink
SYRIAC_ABBREVIATION_MARK = "\u070f"  # ܏, format char; often no bar in the crop
SYRIAC_PTHAHA_ABOVE = "\u0730"  # ܰ, West (Greek-letter) a
SYRIAC_PTHAHA_BELOW = "\u0731"  # ܱ
SYRIAC_PTHAHA_DOTTED = "\u0732"  # ܲ, East a
SYRIAC_ZQAPHA_ABOVE = "\u0733"  # ܳ, West (Greek-letter) o/ā
SYRIAC_ZQAPHA_BELOW = "\u0734"  # ܴ
SYRIAC_ZQAPHA_DOTTED = "\u0735"  # ܵ, East ā
SYRIAC_RBASA_ABOVE = "\u0736"  # ܶ, West (Greek-letter) e
SYRIAC_RBASA_BELOW = "\u0737"  # ܷ
SYRIAC_DOTTED_ZLAMA_HORIZONTAL = "\u0738"  # ܸ, East e
SYRIAC_HBASA_ABOVE = "\u073A"  # ܺ, West (Greek-letter) i
SYRIAC_HBASA_BELOW = "\u073B"  # ܻ
SYRIAC_HBASA_ESASA_DOTTED = "\u073C"  # ܼ, East i/u dot below
SYRIAC_ESASA_ABOVE = "\u073D"  # ܽ, West (Greek-letter) u
SYRIAC_ESASA_BELOW = "\u073E"  # ܾ
SYRIAC_RWAHA = "\u073F"  # ܿ, East single dot above
WEST_VOWELS_TO_EAST = str.maketrans(
    {
        SYRIAC_PTHAHA_ABOVE: SYRIAC_PTHAHA_DOTTED,
        SYRIAC_PTHAHA_BELOW: SYRIAC_PTHAHA_DOTTED,
        SYRIAC_ZQAPHA_ABOVE: SYRIAC_ZQAPHA_DOTTED,
        SYRIAC_ZQAPHA_BELOW: SYRIAC_ZQAPHA_DOTTED,
        SYRIAC_RBASA_ABOVE: SYRIAC_DOTTED_ZLAMA_HORIZONTAL,
        SYRIAC_RBASA_BELOW: SYRIAC_DOTTED_ZLAMA_HORIZONTAL,
        SYRIAC_HBASA_ABOVE: SYRIAC_HBASA_ESASA_DOTTED,
        SYRIAC_HBASA_BELOW: SYRIAC_HBASA_ESASA_DOTTED,
        SYRIAC_ESASA_ABOVE: SYRIAC_HBASA_ESASA_DOTTED,
        SYRIAC_ESASA_BELOW: SYRIAC_HBASA_ESASA_DOTTED,
    }
)


def normalize_symbol_encoding(text: str) -> str:
    """Map duplicate encodings of the same ink onto one Unicode character.

    One-dot phrase stop is ܁, not ASCII period. Two stacked dots are ܃,
    not ASCII colon. One dot above a letter is qushshaya (keep ̇); one
    dot below is rukkakha (keep ̣). Slash-colons ܆/܇, paragraph ܀,
    and seyame ̈ stay as-is — those are different inks.
    Arabic tatweel and the abbreviation mark ܏ are dropped: they are not
    letters, and ܏ is often absent from the crop.

    West (Greek-letter) vowels are mapped onto their East dotted
    equivalents so one vowel is one class: ܰ/ܱ -> ܲ (U+0730/0731 -> U+0732),
    ܳ/ܴ -> ܵ (U+0733/0734 -> U+0735), ܶ/ܷ -> ܸ (U+0736/0737 -> U+0738),
    ܺ/ܻ -> ܼ (U+073A/073B -> U+073C), ܽ/ܾ -> ܼ (U+073D/073E -> U+073C).
    After that, the single-dot East vowels fold into the generic dots:
    hbasa-esasa ܼ (U+073C) becomes ̣ (U+0323) and rwaha ܿ (U+073F) becomes
    ̇ (U+0307). Same ink; the model transcribes what is drawn. The two-dot
    East vowels ܲ ܵ ܸ ܹ (U+0732, U+0735, U+0738, U+0739) stay as-is.
    """
    return (
        text.translate(WEST_VOWELS_TO_EAST)
        .replace(SYRIAC_HBASA_ESASA_DOTTED, COMBINING_DOT_BELOW)
        .replace(SYRIAC_RWAHA, COMBINING_DOT_ABOVE)
        .replace(SYRIAC_QUSHSHAYA, COMBINING_DOT_ABOVE)
        .replace(SYRIAC_RUKKAKHA, COMBINING_DOT_BELOW)
        .replace(ASCII_FULL_STOP, SYRIAC_SUPRALINEAR_FULL_STOP)
        .replace(ASCII_COLON, SYRIAC_SUPRALINEAR_COLON)
        .replace(ARABIC_TATWEEL, "")
        .replace(SYRIAC_ABBREVIATION_MARK, "")
    )


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text.strip())
    text = unicodedata.normalize("NFC", text)
    return normalize_symbol_encoding(text)


def reading_order_index(custom_attr: str | None, fallback: int) -> int:
    if not custom_attr:
        return fallback
    match = READING_ORDER_RE.search(custom_attr)
    if match:
        return int(match.group(1))
    return fallback


def iter_page_lines(xml_path: Path) -> tuple[str, list[LineAnnotation]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page_el = root.find("page:Page", PAGE_NS)
    if page_el is None:
        raise ValueError(f"PAGE XML missing Page element: {xml_path}")

    image_filename = page_el.get("imageFilename")
    if not image_filename:
        raise ValueError(f"PAGE XML missing imageFilename: {xml_path}")

    regions = page_el.findall("page:TextRegion", PAGE_NS)
    region_map = {region.get("id", f"region_{idx}"): region for idx, region in enumerate(regions)}

    ordered_region_ids: list[str] = []
    for region_ref in page_el.findall(
        "./page:ReadingOrder/page:OrderedGroup/page:RegionRefIndexed", PAGE_NS
    ):
        region_id = region_ref.get("regionRef")
        if region_id and region_id in region_map:
            ordered_region_ids.append(region_id)

    for region_id in sorted(region_map):
        if region_id not in ordered_region_ids:
            ordered_region_ids.append(region_id)

    lines: list[LineAnnotation] = []
    page_stem = xml_path.stem

    for region_id in ordered_region_ids:
        region = region_map[region_id]
        text_lines = region.findall("page:TextLine", PAGE_NS)
        indexed_lines = sorted(
            enumerate(text_lines),
            key=lambda pair: reading_order_index(pair[1].get("custom"), pair[0]),
        )

        for _, line_el in indexed_lines:
            text_el = line_el.find("./page:TextEquiv/page:Unicode", PAGE_NS)
            text = normalize_text(text_el.text or "") if text_el is not None else ""
            if not text:
                continue

            coords_el = line_el.find("page:Coords", PAGE_NS)
            polygon = parse_points(coords_el.get("points") if coords_el is not None else None)
            if len(polygon) < 3:
                continue

            baseline_el = line_el.find("page:Baseline", PAGE_NS)
            baseline = parse_points(baseline_el.get("points") if baseline_el is not None else None)
            line_index = len(lines)
            lines.append(
                LineAnnotation(
                    page_stem=page_stem,
                    page_image=image_filename,
                    xml_file=xml_path.name,
                    region_id=region_id,
                    line_id=line_el.get("id", f"line_{line_index}"),
                    line_index=line_index,
                    text=text,
                    polygon=polygon,
                    baseline=baseline,
                )
            )

    return image_filename, lines


def split_pages(
    page_stems: list[str], train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, set[str]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to more than zero.")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n_pages = len(page_stems)
    if n_pages == 0:
        raise ValueError("No PAGE XML files found.")

    n_train = int(round(n_pages * train_ratio))
    n_val = int(round(n_pages * val_ratio))
    n_test = n_pages - n_train - n_val

    if n_pages >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        if n_test == 0:
            n_test = 1
        while n_train + n_val + n_test > n_pages:
            largest = max(
                ("train", n_train), ("val", n_val), ("test", n_test), key=lambda item: item[1]
            )[0]
            if largest == "train" and n_train > 1:
                n_train -= 1
            elif largest == "val" and n_val > 1:
                n_val -= 1
            elif largest == "test" and n_test > 1:
                n_test -= 1
            else:
                break
        while n_train + n_val + n_test < n_pages:
            n_train += 1
    else:
        n_train = max(1, n_pages - 1)
        n_val = 1 if n_pages > 1 else 0
        n_test = n_pages - n_train - n_val

    train_pages = set(page_stems[:n_train])
    val_pages = set(page_stems[n_train : n_train + n_val])
    test_pages = set(page_stems[n_train + n_val :])
    return {"train": train_pages, "val": val_pages, "test": test_pages}


def crop_polygon(
    image: np.ndarray, polygon: list[list[int]], padding: int, keep_color: bool
) -> tuple[np.ndarray, list[int]]:
    polygon_arr = np.array(polygon, dtype=np.int32)
    h, w = image.shape[:2]

    x_min = max(0, int(polygon_arr[:, 0].min()) - padding)
    y_min = max(0, int(polygon_arr[:, 1].min()) - padding)
    x_max = min(w - 1, int(polygon_arr[:, 0].max()) + padding)
    y_max = min(h - 1, int(polygon_arr[:, 1].max()) + padding)
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"Invalid crop bounds: {(x_min, y_min, x_max, y_max)}")

    crop = image[y_min : y_max + 1, x_min : x_max + 1].copy()
    if crop.size == 0:
        raise ValueError(f"Empty crop for bounds {(x_min, y_min, x_max, y_max)}")
    shifted = polygon_arr - np.array([[x_min, y_min]], dtype=np.int32)

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)

    masked = np.full_like(crop, 255)
    masked[mask == 255] = crop[mask == 255]

    if keep_color:
        return masked, [x_min, y_min, x_max, y_max]

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    return gray, [x_min, y_min, x_max, y_max]


def ensure_empty_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {path}\n"
                    "Set OVERWRITE_OUTPUT = True to reuse it."
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_crop(path: Path, crop: np.ndarray, keep_color: bool) -> None:
    if crop.size == 0:
        raise ValueError(f"Empty crop for {path}")
    if keep_color:
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        image = Image.fromarray(crop)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=82, optimize=True)
    elif suffix == ".png":
        image.save(path, compress_level=9, optimize=True)
    else:
        image.save(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_output_root(source_dir: Path) -> Path:
    return (REPO_ROOT / "data" / "syriacData" / source_dir.name).resolve()


def main() -> None:
    source_dir = _resolve_path(SOURCE_DIR)
    if not source_dir.is_dir():
        raise SystemExit(f"SOURCE_DIR is not a directory: {source_dir}")

    page_dir = source_dir / "page"
    if not page_dir.is_dir():
        raise SystemExit(f"Missing PAGE XML directory: {page_dir}")

    output_root = (
        _default_output_root(source_dir) if OUTPUT_ROOT is None else _resolve_path(OUTPUT_ROOT)
    )
    ensure_empty_output(output_root, overwrite=bool(OVERWRITE_OUTPUT))

    xml_files = sorted(page_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No PAGE XML files found under {page_dir}")

    split_map = split_pages(
        [xml_path.stem for xml_path in xml_files],
        train_ratio=float(TRAIN_RATIO),
        val_ratio=float(VAL_RATIO),
        test_ratio=float(TEST_RATIO),
    )

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    summary = {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "splits": {},
        "image_format": IMAGE_FORMAT,
        "padding": int(PADDING),
        "grayscale": not bool(KEEP_COLOR),
        "skipped_invalid_lines": [],
    }

    image_format = str(IMAGE_FORMAT)
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"Unsupported image_format: {image_format}")
    ext = ".jpg" if image_format == "jpeg" else f".{image_format}"

    for xml_path in xml_files:
        page_image_name, lines = iter_page_lines(xml_path)
        page_stem = xml_path.stem
        split = next(name for name, page_set in split_map.items() if page_stem in page_set)

        image_path = source_dir / page_image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read page image: {image_path}")

        for line in lines:
            try:
                crop, bbox = crop_polygon(
                    image,
                    line.polygon,
                    int(PADDING),
                    keep_color=bool(KEEP_COLOR),
                )
            except ValueError as exc:
                summary["skipped_invalid_lines"].append(
                    {
                        "page_stem": page_stem,
                        "line_id": line.line_id,
                        "line_index": line.line_index,
                        "reason": str(exc),
                    }
                )
                continue
            base_name = f"{page_stem}__{line.line_index:03d}"
            image_rel = Path("images") / split / f"{base_name}{ext}"
            label_rel = Path("labels") / split / f"{base_name}.gt.txt"

            image_out = output_root / image_rel
            label_out = output_root / label_rel

            save_crop(image_out, crop, keep_color=bool(KEEP_COLOR))
            with label_out.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{line.text}\n")

            manifest_rows[split].append(
                {
                    "page_stem": page_stem,
                    "page_image": page_image_name,
                    "xml_file": xml_path.name,
                    "region_id": line.region_id,
                    "line_id": line.line_id,
                    "line_index": line.line_index,
                    "text": line.text,
                    "polygon": line.polygon,
                    "baseline": line.baseline,
                    "bbox": bbox,
                    "image_relpath": image_rel.as_posix(),
                    "label_relpath": label_rel.as_posix(),
                }
            )

    for split in ("train", "val", "test"):
        write_jsonl(manifests_dir / f"{split}.jsonl", manifest_rows[split])
        split_pages_sorted = sorted(split_map[split])
        summary["splits"][split] = {
            "pages": split_pages_sorted,
            "page_count": len(split_pages_sorted),
            "line_count": len(manifest_rows[split]),
        }

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
