"""Language provenance for language-specific OCR evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


LANGUAGES = ("greek", "armenian", "syriac", "coptic")


def language_labels(data_root: Path, split: str, image_names: Sequence[str]) -> list[str]:
    """Return each sample's language from the processed data layout.

    Combined manifests retain the original image names. Their language is
    resolved against all language-specific manifests for the partition, rather
    than inferred from a filename convention. The language and combined
    datasets can use independently assigned train/validation/test splits.
    """
    if data_root.parent.name in LANGUAGES:
        return [data_root.parent.name] * len(image_names)
    if data_root.parent.name != "combined":
        return ["unknown"] * len(image_names)

    memberships: dict[str, str] = {}
    processed_root = data_root.parent.parent
    for language in LANGUAGES:
        language_root = processed_root / language / data_root.name
        for source_split in ("train", "val", "test"):
            manifest = language_root / f"gt_{source_split}.txt"
            if not manifest.is_file():
                continue
            for line_number, line in enumerate(
                manifest.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line:
                    continue
                try:
                    image_name, _ = line.split("\t", 1)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid language manifest row {manifest}:{line_number}"
                    ) from error
                if image_name in memberships:
                    raise ValueError(
                        f"Duplicate image name across language manifests: {image_name}"
                    )
                memberships[image_name] = language

    missing = [image_name for image_name in image_names if image_name not in memberships]
    if missing:
        raise ValueError(
            f"Unable to resolve language for {len(missing)} samples in {data_root}/{split}."
        )
    return [memberships[image_name] for image_name in image_names]


def language_indices(labels: Sequence[str]) -> dict[str, list[int]]:
    """Return dataset indices for each recognized language."""
    indices = {language: [] for language in LANGUAGES}
    for index, language in enumerate(labels):
        if language in indices:
            indices[language].append(index)
    return {language: values for language, values in indices.items() if values}
