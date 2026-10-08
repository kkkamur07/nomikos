"""Checkpoint evaluation helpers for trained TrOCR Hugging Face directories."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor

from ...metrics.metrics import compute_sequence_length_metrics, compute_text_metrics
from ..trocr.dataloader import read_ground_truth
from ..trocr.model_builder import load_model


class _ImageTextDataset(Dataset):
    """Pairs of line images and reference transcriptions for offline scoring."""

    def __init__(self, data_root: Path, split: str) -> None:
        self.image_dir = data_root / "image"
        self.samples = read_ground_truth(data_root / f"gt_{split}.txt")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_name, text = self.samples[index]
        image = Image.open(self.image_dir / image_name).convert("RGB")
        return {"image": image, "text": text}


def _load_processor(checkpoint: Path) -> TrOCRProcessor:
    """Load processor; ``best/`` dirs often omit ``preprocessor_config.json``."""
    if (checkpoint / "preprocessor_config.json").is_file():
        return TrOCRProcessor.from_pretrained(checkpoint)

    from transformers import AutoImageProcessor, AutoTokenizer

    image_sources = (
        checkpoint.parent / "final",
        Path("trocr_checkpoints/trocr-base-handwritten"),
    )
    image_processor = None
    for source in image_sources:
        if (source / "preprocessor_config.json").is_file():
            image_processor = AutoImageProcessor.from_pretrained(source)
            break
    if image_processor is None:
        raise FileNotFoundError(
            f"No preprocessor_config.json next to {checkpoint} or in final/base fallbacks"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _max_length(checkpoint: Path, processor: TrOCRProcessor) -> int:
    generation = checkpoint / "generation_config.json"
    if generation.is_file():
        payload = json.loads(generation.read_text(encoding="utf-8"))
        value = payload.get("max_length")
        if isinstance(value, int) and value > 0:
            return value
    model_max = getattr(processor.tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 100_000:
        return model_max
    return 160


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path,
    *,
    split: str,
    batch_size: int,
    workers: int,
    device: str,
    num_beams: int = 1,
) -> dict[str, float]:
    """Evaluate a TrOCR ``best/`` or ``final/`` directory on one dataset split."""
    checkpoint = Path(checkpoint)
    data_root = Path(data_root)
    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    processor = _load_processor(checkpoint)
    model = load_model(str(checkpoint)).to(torch_device).eval()
    max_length = _max_length(checkpoint, processor)
    dataset = _ImageTextDataset(data_root, split)

    def collate(batch: list[dict[str, object]]) -> dict[str, object]:
        images = [row["image"] for row in batch]
        texts = [str(row["text"]) for row in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values, "texts": texts}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
    )

    references: list[str] = []
    hypotheses: list[str] = []
    use_amp = torch_device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(torch_device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        pixel_values,
                        max_length=max_length,
                        num_beams=num_beams,
                    )
            else:
                generated = model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            decoded = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            references.extend(batch["texts"])
            hypotheses.extend(decoded)

    metrics = compute_text_metrics(references, hypotheses)
    metrics.update(compute_sequence_length_metrics(references, hypotheses))
    # Match Calamari's result schema: keep a loss key even though we only decode.
    metrics.setdefault("loss", float("nan"))
    return metrics


def count_parameters(checkpoint: Path) -> int:
    """Parameter count of a TrOCR checkpoint directory."""
    model = load_model(str(checkpoint))
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    del model
    return total


def timed_evaluate(
    checkpoint: Path,
    data_root: Path,
    *,
    split: str,
    batch_size: int,
    workers: int,
    device: str,
    num_beams: int = 1,
    warm: bool = False,
) -> tuple[dict[str, float], float]:
    """Run :func:`evaluate_checkpoint` and return ``(metrics, wall_seconds)``."""
    start = time.perf_counter()
    metrics = evaluate_checkpoint(
        checkpoint,
        data_root,
        split=split,
        batch_size=batch_size,
        workers=workers,
        device=device,
        num_beams=num_beams,
    )
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if warm:
        # Caller may discard the second metrics; we only care about the timer.
        _ = elapsed
    return metrics, elapsed
