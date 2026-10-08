"""Line-image datasets and batching for TrOCR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, TrOCRProcessor

from ...metrics.languages import language_labels
from ...augmentation.augmentation import build_variant_plan, plan_for_augmented_variant
from .augmentation import LineAugmentation


def read_ground_truth(path: Path) -> list[tuple[str, str]]:
    """Read ``image-name<TAB>transcription`` manifest rows."""
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            samples.append(tuple(line.split("\t", 1)))
    return samples


class LineDataset(Dataset):
    """OCR line images backed by a ``gt_<split>.txt`` manifest."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        augmentation: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.image_dir = data_dir / "image"
        self.samples = read_ground_truth(data_dir / f"gt_{split}.txt")
        self.languages = language_labels(
            data_dir, split, [image_name for image_name, _ in self.samples]
        )
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_name, text = self.samples[index]
        image = Image.open(self.image_dir / image_name).convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image)
        return {"image": image, "text": text}


class TrOCRAugmentedDataset(Dataset):
    """Expose one original and multiple independently augmented line images."""

    def __init__(
        self,
        dataset: Dataset,
        augmentation: LineAugmentation,
        copies: Mapping[str, int],
    ) -> None:
        self.dataset = dataset
        self.augmentation = augmentation
        self.plan = build_variant_plan(copies)
        self.n_augmentations = len(self.plan)

    def __len__(self) -> int:
        return len(self.dataset) * self._variants_per_sample

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, variant = divmod(index, self._variants_per_sample)
        sample = dict(self.dataset[sample_index])
        if variant:
            strength, operation_count = plan_for_augmented_variant(variant, self.plan)
            if strength == "none":
                # Identity copy: the original image, untouched.
                return sample
            image = sample["image"]
            if not isinstance(image, Image.Image):
                raise TypeError("TrOCR augmentation requires PIL images.")
            sample["image"] = self.augmentation.apply(image, strength, operation_count)
        return sample

    @property
    def _variants_per_sample(self) -> int:
        return self.n_augmentations + 1


@dataclass
class TrOCRCollator:
    """Convert image/text examples into TrOCR pixel values and labels."""

    processor: TrOCRProcessor
    max_target_length: int

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        tokenizer: PreTrainedTokenizerBase = self.processor.tokenizer
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define EOS and PAD tokens.")
        if self.max_target_length < 1:
            raise ValueError("max_target_length must be at least one.")

        images = [feature["image"] for feature in features]
        texts = [str(feature["text"]) for feature in features]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length - 1,
            padding=False,
        )
        labels = torch.full(
            (len(texts), self.max_target_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(encoded.input_ids):
            sequence = [*token_ids, tokenizer.eos_token_id]
            labels[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}
