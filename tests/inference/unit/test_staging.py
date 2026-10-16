"""Dataset staging validation: every rejection branch, pinned by message."""

from __future__ import annotations

from pathlib import Path

import pytest

from nomikos_inference.publish.staging import (
    DatasetStagingRef,
    ModelStagingRef,
    build_dataset_readme,
    build_model_card,
    validate_dataset_slug,
    validate_dataset_staging,
    validate_model_staging,
)


def _valid_tree(root: Path) -> Path:
    (root / "images" / "ms-001").mkdir(parents=True)
    (root / "images" / "ms-001" / "line-0001.png").write_bytes(b"png")
    (root / "images" / "ms-001" / "line-0002.png").write_bytes(b"png")
    (root / "labels.csv").write_text(
        "image,transcription\n"
        "images/ms-001/line-0001.png,hello\n"
        "images/ms-001/line-0002.png,world\n",
        encoding="utf-8",
    )
    return root


def test_valid_staging_passes(tmp_path: Path) -> None:
    assert validate_dataset_staging(_valid_tree(tmp_path / "staging")) is None


def test_missing_staging_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging directory not found"):
        validate_dataset_staging(tmp_path / "absent")


def test_missing_images_dir(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "labels.csv").write_text("image,transcription\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing images/"):
        validate_dataset_staging(root)


def test_missing_labels_csv(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    (root / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing labels.csv"):
        validate_dataset_staging(root)


def test_empty_images_dir(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    (root / "images").mkdir(parents=True)
    (root / "labels.csv").write_text("image,transcription\n", encoding="utf-8")
    with pytest.raises(ValueError, match="images/ directory is empty"):
        validate_dataset_staging(root)


def test_labels_missing_columns(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text("image\nimages/ms-001/line-0001.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain image,transcription columns"):
        validate_dataset_staging(root)


def test_row_without_image(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text("image,transcription\n,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2 has no image path"):
        validate_dataset_staging(root)


def test_row_without_transcription(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text(
        "image,transcription\nimages/ms-001/line-0001.png,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="row 2 has no transcription"):
        validate_dataset_staging(root)


@pytest.mark.parametrize(
    "bad_image",
    [
        "/abs/images/ms-001/line-0001.png",
        "images/../labels.csv",
        "crops/ms-001/line-0001.png",
    ],
)
def test_row_with_non_images_relative_path(tmp_path: Path, bad_image: str) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text(f"image,transcription\n{bad_image},hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a relative images/ path"):
        validate_dataset_staging(root)


def test_row_referencing_missing_crop(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text(
        "image,transcription\nimages/ms-001/ghost.png,hello\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="references missing crop"):
        validate_dataset_staging(root)


def test_duplicate_crop_pairing(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text(
        "image,transcription\n"
        "images/ms-001/line-0001.png,hello\n"
        "images/ms-001/line-0001.png,again\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate crop pairing"):
        validate_dataset_staging(root)


def test_unpaired_images_are_listed(tmp_path: Path) -> None:
    root = _valid_tree(tmp_path / "staging")
    (root / "labels.csv").write_text(
        "image,transcription\nimages/ms-001/line-0001.png,hello\n", encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match=r"images without labels.csv pairings: images/ms-001/line-0002.png"
    ):
        validate_dataset_staging(root)


def test_model_card_pins_registry_fields() -> None:
    ref = ModelStagingRef(
        script="syriac", architecture="calamari", model_version="v1", registry_tag="stable"
    )
    card = build_model_card(ref, namespace="nomikos", task="transcribe")
    assert "`syriac-calamari-v1`" in card
    assert "hf://nomikos/syriac-htr-calamari@stable" in card
    assert "handwritten-text-recognition" in card
    assert "- syr" in card
    assert "# Syriac HTR (calamari)" in card


def test_model_card_segmentation_has_no_language() -> None:
    ref = ModelStagingRef(
        script="segmentation",
        architecture="blla-segment",
        model_version="v1",
        registry_tag="stable",
    )
    card = build_model_card(ref, namespace="nomikos", task="segment")
    assert "layout-analysis" in card
    assert "language:" not in card
    assert "# Document Segmentation (blla-segment)" in card


def test_dataset_readme_names_slug_and_script() -> None:
    ref = DatasetStagingRef(dataset_slug="syriac-lines")
    readme = build_dataset_readme(ref, namespace="nomikos", script="syriac")
    assert "syriac-lines" in readme
    assert "syriac" in readme
    assert "labels.csv" in readme


def test_dataset_slug_conventions() -> None:
    validate_dataset_slug(dataset_slug="syriac-manuscript-lines", script="syriac")
    validate_dataset_slug(dataset_slug="syriac-abc-htr-lines", script="syriac")
    with pytest.raises(ValueError, match="dataset slug must follow"):
        validate_dataset_slug(dataset_slug="syriac-lines", script="syriac")
    with pytest.raises(ValueError, match="dataset slug must follow"):
        validate_dataset_slug(dataset_slug="Syriac_Lines", script="syriac")


def test_model_staging_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging directory not found"):
        validate_model_staging(tmp_path / "absent", architecture="calamari")
    with pytest.raises(ValueError, match="missing a supported Hub artifact"):
        validate_model_staging(tmp_path, architecture="calamari")
