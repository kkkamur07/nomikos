"""Hub staging tree layout, validation, and card generation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nomikos_inference.hub.artifacts import find_hub_artifact
from nomikos_inference.publish.paths import DEFAULT_STAGING_ROOT


@dataclass(frozen=True, slots=True)
class ModelStagingRef:
    script: str
    architecture: str
    model_version: str
    registry_tag: str

    @property
    def registry_model_id(self) -> str:
        return f"{self.script}-{self.architecture}-{self.model_version}"

    @property
    def hub_repo_slug(self) -> str:
        return hub_repo_slug(self.script, self.architecture)

    def weights_source(self, *, namespace: str) -> str:
        return f"hf://{namespace}/{self.hub_repo_slug}@{self.registry_tag}"


def hub_repo_slug(script: str, architecture: str) -> str:
    if architecture in {"blla", "blla-segment", "blla_segment"}:
        return "segmentation-blla"
    return f"{script}-htr-{architecture}"


# Hub model card `language` metadata requires ISO 639 codes, not script slugs.
_SCRIPT_LANGUAGE_CODES: dict[str, str] = {
    "greek": "grc",
    "syriac": "syr",
}


def hub_language_code(script: str) -> str | None:
    if script == "segmentation":
        return None
    return _SCRIPT_LANGUAGE_CODES.get(script, script)


def model_staging_dir(
    ref: ModelStagingRef,
    *,
    staging_root: Path | None = None,
) -> Path:
    root = staging_root or DEFAULT_STAGING_ROOT
    return root / "models" / ref.script / ref.architecture / ref.model_version / ref.registry_tag


def validate_model_staging(
    staging_dir: Path,
    *,
    architecture: str,
) -> None:
    if not staging_dir.is_dir():
        raise ValueError(f"model staging directory not found: {staging_dir}")

    try:
        find_hub_artifact(staging_dir, architecture=architecture)
    except FileNotFoundError as exc:
        raise ValueError(
            f"model staging directory {staging_dir} is missing a supported Hub artifact "
            f"for architecture {architecture!r}"
        ) from exc


def build_model_card(
    ref: ModelStagingRef,
    *,
    namespace: str,
    task: str,
    registry_model_id: str | None = None,
) -> str:
    model_id = registry_model_id or ref.registry_model_id
    weights_source = f"hf://{namespace}/{ref.hub_repo_slug}@{ref.registry_tag}"
    title_script = ref.script.capitalize()

    language_code = hub_language_code(ref.script)
    is_segmentation = task == "segment"
    task_tag = "layout-analysis" if is_segmentation else "handwritten-text-recognition"
    usage_description = (
        "General-purpose page segmentation checkpoint for manuscript images."
        if is_segmentation
        else f"Handwritten text recognition checkpoint for **{ref.script}** manuscripts."
    )
    title = "Document Segmentation" if is_segmentation else f"{title_script} HTR"
    language_frontmatter = f"language:\n- {language_code}\n" if language_code else ""
    return f"""---
{language_frontmatter}tags:
- {task_tag}
- {ref.architecture}
- ocr
library_name: {ref.architecture}
---

# {title} ({ref.architecture})

{usage_description} Published from the nomikos **Hub staging tree**.

| Field | Value |
|-------|-------|
| **registry model id** | `{model_id}` |
| **registry tag** | `{ref.registry_tag}` |
| **script** | `{ref.script}` |
| **architecture** | `{ref.architecture}` |
| **model version** | `{ref.model_version}` |
| **task** | `{task}` |
| **weights source** | `{weights_source}` |

## Usage

Resolve this checkpoint through the inference **Registry** with:

```yaml
weights_source: {weights_source}
```

Prefetch into the **Hub cache** without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py {model_id} --registry-tag {ref.registry_tag}
```
"""


@dataclass(frozen=True, slots=True)
class DatasetStagingRef:
    dataset_slug: str

    def repo_id(self, *, namespace: str) -> str:
        return f"{namespace}/{self.dataset_slug}"


_SLUG_COMPONENT_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"


def validate_dataset_slug(*, dataset_slug: str, script: str) -> None:
    """Require the searchable Hub dataset-slug conventions from CONTEXT."""
    if not re.fullmatch(_SLUG_COMPONENT_PATTERN, script):
        raise ValueError(f"script must be a lowercase slug: {script!r}")

    expected = re.compile(
        rf"{re.escape(script)}-(?:manuscript-lines|{_SLUG_COMPONENT_PATTERN}-htr-lines)"
    )
    if not expected.fullmatch(dataset_slug):
        raise ValueError(
            "dataset slug must follow "
            f"{script}-manuscript-lines or {script}-{{corpus}}-htr-lines: {dataset_slug!r}"
        )


def dataset_staging_dir(
    ref: DatasetStagingRef,
    *,
    staging_root: Path | None = None,
) -> Path:
    root = staging_root or DEFAULT_STAGING_ROOT
    return root / "datasets" / ref.dataset_slug


def _require_dataset_layout(staging_dir: Path) -> tuple[Path, Path]:
    if not staging_dir.is_dir():
        raise ValueError(f"dataset staging directory not found: {staging_dir}")
    images_dir = staging_dir / "images"
    labels_path = staging_dir / "labels.csv"
    if not images_dir.is_dir():
        raise ValueError(f"dataset staging directory missing images/: {staging_dir}")
    if not labels_path.is_file():
        raise ValueError(f"dataset staging directory missing labels.csv: {staging_dir}")
    return images_dir, labels_path


def _collect_staged_images(images_dir: Path, staging_dir: Path) -> set[str]:
    image_paths = {
        path.relative_to(staging_dir).as_posix() for path in images_dir.rglob("*") if path.is_file()
    }
    if not image_paths:
        raise ValueError(f"dataset images/ directory is empty: {images_dir}")
    return image_paths


def _check_label_row(
    row: dict[str, str | None],
    row_number: int,
    image_paths: set[str],
    labelled_paths: set[str],
) -> None:
    image = (row.get("image") or "").strip()
    transcription = (row.get("transcription") or "").strip()
    if not image:
        raise ValueError(f"dataset labels.csv row {row_number} has no image path")
    if not transcription:
        raise ValueError(f"dataset labels.csv row {row_number} has no transcription")
    image_path = PurePosixPath(image)
    if image_path.is_absolute() or ".." in image_path.parts or image_path.parts[:1] != ("images",):
        raise ValueError(
            f"dataset labels.csv row {row_number} image must be a relative images/ path"
        )
    normalized_image = image_path.as_posix()
    if normalized_image not in image_paths:
        raise ValueError(f"dataset labels.csv row {row_number} references missing crop: {image}")
    if normalized_image in labelled_paths:
        raise ValueError(f"dataset labels.csv contains duplicate crop pairing: {normalized_image}")
    labelled_paths.add(normalized_image)


def _read_labelled_paths(labels_path: Path, image_paths: set[str]) -> set[str]:
    labelled_paths: set[str] = set()
    with labels_path.open(encoding="utf-8", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        if reader.fieldnames is None or not {"image", "transcription"}.issubset(reader.fieldnames):
            raise ValueError(
                f"dataset labels.csv must contain image,transcription columns: {labels_path}"
            )
        for row_number, row in enumerate(reader, start=2):
            _check_label_row(row, row_number, image_paths, labelled_paths)
    return labelled_paths


def validate_dataset_staging(staging_dir: Path) -> None:
    images_dir, labels_path = _require_dataset_layout(staging_dir)
    image_paths = _collect_staged_images(images_dir, staging_dir)
    labelled_paths = _read_labelled_paths(labels_path, image_paths)
    unpaired_images = image_paths - labelled_paths
    if unpaired_images:
        unpaired = ", ".join(sorted(unpaired_images))
        raise ValueError(f"dataset images without labels.csv pairings: {unpaired}")


def build_dataset_readme(
    ref: DatasetStagingRef,
    *,
    namespace: str,
    script: str,
    crop_format: str = "Image line crops with UTF-8 transcriptions",
    provenance: str = "Curated from nomikos annotation exports.",
) -> str:
    return f"""---
language:
- {hub_language_code(script)}
tags:
- handwritten-text-recognition
- manuscript
- line-crops
task_categories:
- image-to-text
---

# {ref.dataset_slug}

Labelled manuscript line crops for training and evaluating **{script}** handwritten text recognition models.

| Field | Value |
|-------|-------|
| **Hub dataset slug** | `{ref.dataset_slug}` |
| **script** | `{script}` |
| **crop format** | {crop_format} |
| **provenance** | {provenance} |

## Pairing convention

`labels.csv` has UTF-8 `image,transcription` columns. Each `image` value is a
relative path below `images/` (for example `images/ms-001/line-0001.png`), and
must name exactly one crop. Every crop under `images/` has exactly one row in
`labels.csv`. Use PNG line crops where possible; preserve the original crop
dimensions and do not bake model-specific resizing into this dataset.

## Relationship to inference models

This **Hub dataset repo** holds training/evaluation material only. Inference loads weights from separate **Hub model repos** (for example `{script}-htr-calamari`) via `hf://` **weights sources** in the inference **Registry**.

One dataset can support multiple future **registry model ids**; it is not itself a
Registry entry and it must not contain inference weights. After publishing a new
model generation, update `nomikos_inference/publish/collection.yaml` so the `nomos` **Hub
collection** lists both the dataset and the corresponding **Hub model repo**.

## Staging layout

Stage publish-ready files under:

```
nomikos_inference/publish/artifacts/staging/datasets/{ref.dataset_slug}/
  images/
  labels.csv
```

Then publish with:

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py {ref.dataset_slug} --script {script} --upload
```
"""
