"""Evaluate a trained TrOCR checkpoint on randomly selected samples."""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from transformers import TrOCRProcessor

from ..metrics.metrics import edit_distance
from ..models.trocr.dataloader import read_ground_truth
from ..models.trocr.model_builder import load_model


@hydra.main(version_base=None, config_path="../../config/trocr", config_name="configs")
def main(cfg: DictConfig) -> None:
    data_dir = Path(to_absolute_path(cfg.data.dir)).expanduser().resolve()
    checkpoint = Path(to_absolute_path(cfg.evaluation.checkpoint)).expanduser().resolve()
    split = str(cfg.evaluation.split)
    samples = read_ground_truth(data_dir / f"gt_{split}.txt")
    selected = random.Random(cfg.evaluation.seed).sample(
        samples,
        min(cfg.evaluation.num_samples, len(samples)),
    )
    processor = TrOCRProcessor.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(checkpoint)).to(device).eval()

    total_edits = 0
    total_reference_characters = 0
    with torch.inference_mode():
        for index, (image_name, reference) in enumerate(selected, start=1):
            image = Image.open(data_dir / "image" / image_name).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(
                pixel_values,
                max_length=cfg.evaluation.max_length,
                num_beams=cfg.evaluation.num_beams,
            )
            prediction = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            edits = edit_distance(reference, prediction)
            sample_cer = edits / len(reference) if reference else float(bool(prediction))
            total_edits += edits
            total_reference_characters += len(reference)
            print(f"[{index}] {image_name}")
            print(f"  reference:  {reference}")
            print(f"  prediction: {prediction}")
            print(f"  CER: {sample_cer:.4f}")

    cer = total_edits / total_reference_characters if total_reference_characters else 0.0
    print(f"Cumulative CER: {cer:.4f} ({cer * 100:.2f}%)")


if __name__ == "__main__":
    main()
