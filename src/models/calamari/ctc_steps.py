"""Print per-timestep CTC softmax dumps for a few labelled lines."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .codec import CharacterCodec
from .data import collate_ctc
from .model import CalamariTorchModel


def ctc_step_report(
    model: CalamariTorchModel,
    dataset: Dataset[dict[str, object]],
    codec: CharacterCodec,
    device: torch.device,
    *,
    n_lines: int = 3,
    top_k: int = 3,
) -> str:
    """Format greedy CTC distributions for the first ``n_lines`` dataset items."""
    if n_lines < 1 or len(dataset) == 0:
        return ""
    count = min(n_lines, len(dataset))
    batch = collate_ctc([dataset[index] for index in range(count)])
    image = _tensor(batch["image"], "image").to(device)
    image_lengths = _tensor(batch["image_lengths"], "image_lengths").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image, image_lengths=image_lengths)
    logits = outputs["logits"].detach().cpu()
    lengths = outputs["out_len"].detach().cpu()
    blocks = [
        codec.format_frame_distributions(
            logits[index],
            int(lengths[index]),
            reference=str(batch["texts"][index]),
            top_k=top_k,
        )
        for index in range(count)
    ]
    return "\n\n".join(
        f"--- eval CTC steps sample {index + 1} ---\n{block}"
        for index, block in enumerate(blocks)
    )


def write_ctc_step_report(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")


def _tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Calamari batch {name} must be a Tensor.")
    return value
