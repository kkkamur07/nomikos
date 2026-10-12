"""Line-image dataset and CTC batching for PyTorch Calamari."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ...metrics.languages import language_labels
from ...augmentation.augmentation import build_variant_plan, plan_for_augmented_variant
from .augmentation import augment_legacy_line_image, jitter_syriac_line_width
from .codec import CharacterCodec

_SYRIAC_2024_PREFIX = "syriac_2024_"
_SYRIAC_2024_TRAIN_REPEATS = 2


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass(frozen=True)
class LineSample:
    image_path: Path
    text: str
    language: str


class CalamariLineDataset(Dataset[dict[str, object]]):
    """Read paired line images and ``.gt.txt`` transcriptions for one split."""

    def __init__(self, root: Path, split: str, codec: CharacterCodec, line_height: int) -> None:
        self.codec = codec
        self.line_height = line_height
        self.samples = collect_samples(root, split)
        self._decoded_images: dict[int, Tensor] = {}
        self._encoded_targets: dict[int, Tensor] = {}
        if not self.samples:
            raise ValueError(f"No labeled Calamari samples found for split {split!r} in {root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "image": self._decoded_image(index, sample.image_path),
            "targets": self._encoded_target(index, sample.text),
            "text": sample.text,
            "language": sample.language,
        }

    def _decoded_image(self, index: int, image_path: Path) -> Tensor:
        cached = self._decoded_images.get(index)
        if cached is not None:
            return cached
        image = _load_line_image(image_path, self.line_height)
        self._decoded_images[index] = image
        return image

    def _encoded_target(self, index: int, text: str) -> Tensor:
        cached = self._encoded_targets.get(index)
        if cached is not None:
            return cached
        encoded = self.codec.encode(text)
        self._encoded_targets[index] = encoded
        return encoded


class CalamariAugmentedDataset(Dataset[dict[str, object]]):
    """Expose each training sample together with legacy Calamari augmentations."""

    def __init__(
        self,
        dataset: Dataset[dict[str, object]],
        copies: Mapping[str, int],
        probability: float = 1.0,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Calamari augmentation probability must be between zero and one.")
        self.dataset = dataset
        self.plan = build_variant_plan(copies)
        self.n_augmentations = len(self.plan)
        self.probability = probability

    def __len__(self) -> int:
        return len(self.dataset) * self._variants_per_sample

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, variant = divmod(index, self._variants_per_sample)
        sample = dict(self.dataset[sample_index])
        image = sample["image"]
        if not isinstance(image, Tensor):
            raise TypeError("Calamari augmentation requires tensor images.")
        if variant:
            strength, operation_count = plan_for_augmented_variant(variant, self.plan)
            # "none" copies are exact duplicates of the original: no jitter, no ops.
            if strength != "none" and numpy.random.random() < self.probability:
                if sample.get("language") == "syriac":
                    image = jitter_syriac_line_width(image, str(sample.get("text", "")))
                image = augment_legacy_line_image(image, strength, operation_count)
        sample["image"] = image
        return sample

    @property
    def _variants_per_sample(self) -> int:
        return self.n_augmentations + 1 if self.probability > 0.0 else 1


class RemappedDataset(Dataset[dict[str, object]]):
    """Address ``dataset`` through an index list so repeats share the same cache."""

    def __init__(self, dataset: Dataset[dict[str, object]], indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        if not indices:
            raise ValueError("RemappedDataset requires at least one index.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.dataset[self.indices[index]]


def repeat_syriac_2024_train(dataset: CalamariLineDataset) -> Dataset[dict[str, object]]:
    """See each ``syriac_2024_`` train line twice without copying files or GT."""
    indices: list[int] = []
    for index, sample in enumerate(dataset.samples):
        repeats = (
            _SYRIAC_2024_TRAIN_REPEATS
            if sample.image_path.name.startswith(_SYRIAC_2024_PREFIX)
            else 1
        )
        indices.extend([index] * repeats)
    if len(indices) == len(dataset.samples):
        return dataset
    return RemappedDataset(dataset, indices)


def collect_samples(root: Path, split: str) -> list[LineSample]:
    """Support canonical flat packs and the source ``images/labels`` layout."""
    root = _resolve_virtual_root(root)
    manifest = root / f"gt_{split}.txt"
    trocr_images = root / "image"
    if manifest.is_file() and trocr_images.is_dir():
        samples: list[tuple[Path, str]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                image_name, text = line.split("\t", 1)
                samples.append((trocr_images / image_name, text))
        return _with_languages(root, split, samples)

    flat_split = root / split
    images_dir = flat_split if flat_split.is_dir() else root / "images" / split
    labels_dir = flat_split if flat_split.is_dir() else root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []

    samples = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.gt.txt"
        if label_path.is_file():
            samples.append((image_path, label_path.read_text(encoding="utf-8")))
    return _with_languages(root, split, samples)


def _with_languages(root: Path, split: str, samples: list[tuple[Path, str]]) -> list[LineSample]:
    languages = language_labels(root, split, [image_path.name for image_path, _ in samples])
    return [
        LineSample(image_path=image_path, text=text, language=language)
        for (image_path, text), language in zip(samples, languages, strict=True)
    ]


def _resolve_virtual_root(root: Path) -> Path:
    marker = root / "source.txt"
    if not marker.is_file():
        return root
    source = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
    if not source.is_absolute():
        source = marker.parent / source
    resolved = source.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Calamari source marker points to a missing directory: {resolved}")
    return resolved


def collate_ctc(samples: list[dict[str, object]]) -> dict[str, object]:
    """Pad variable-width images and concatenate CTC label targets."""
    images = [sample["image"] for sample in samples]
    targets = [sample["targets"] for sample in samples]
    if not all(isinstance(image, Tensor) for image in images) or not all(
        isinstance(target, Tensor) for target in targets
    ):
        raise TypeError("Invalid Calamari batch.")
    typed_images = [image for image in images if isinstance(image, Tensor)]
    typed_targets = [target for target in targets if isinstance(target, Tensor)]
    widths = torch.tensor([image.shape[0] for image in typed_images], dtype=torch.long)
    height = typed_images[0].shape[1]
    batch = torch.zeros((len(typed_images), int(widths.max()), height, 1), dtype=torch.float32)
    labels = torch.zeros(
        (len(typed_targets), max(target.numel() for target in typed_targets)),
        dtype=torch.long,
    )
    for index, image in enumerate(typed_images):
        batch[index, : image.shape[0]] = image
    for index, target in enumerate(typed_targets):
        labels[index, : target.numel()] = target
    return {
        "image": batch,
        "image_lengths": widths,
        "targets": torch.cat(typed_targets),
        "target_lengths": torch.tensor(
            [target.numel() for target in typed_targets], dtype=torch.long
        ),
        # Padded labels are used solely by Hugging Face Trainer's evaluation
        # loop; CTC loss continues to consume the concatenated targets above.
        "labels": labels,
        "texts": [str(sample["text"]) for sample in samples],
        "languages": [str(sample["language"]) for sample in samples],
    }


def to_ink_bright(pixels: numpy.ndarray) -> numpy.ndarray:
    """Return a uint8 line with ink = 255 and paper = 0.

    All stored crops are paper-bright (the Esteban crops were re-polarised on
    disk on 2026-10-08).
    """
    return 255 - numpy.asarray(pixels, dtype=numpy.uint8)


def load_line_image_ink_bright(path: Path, line_height: int) -> Tensor:
    """Load a line as an ink-bright ``(width, height, 1)`` uint8 tensor at ``line_height``."""
    with Image.open(path) as source:
        image = source.convert("L")
        width = max(1, round(image.width * line_height / image.height))
        image = image.resize((width, line_height), Image.Resampling.BILINEAR)
        pixels = to_ink_bright(numpy.asarray(image))
    return torch.from_numpy(pixels.T.copy()).unsqueeze(-1)


def _load_line_image(path: Path, line_height: int) -> Tensor:
    return load_line_image_ink_bright(path, line_height)
