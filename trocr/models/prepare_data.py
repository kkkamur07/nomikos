"""Create the line-image manifest layout consumed by TrOCR training."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@hydra.main(version_base=None, config_path="../../../config/trocr", config_name="configs")
def main(cfg: DictConfig) -> None:
    raw_root = Path(to_absolute_path(cfg.preparation.raw_root)).expanduser().resolve()
    output_dir = Path(to_absolute_path(cfg.preparation.output_dir)).expanduser().resolve()
    image_dir = output_dir / "image"
    for split in ("train", "val", "test"):
        manifest = raw_root / "manifests" / f"{split}.jsonl"
        rows = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"gt_{split}.txt").open("w", encoding="utf-8") as output:
            for row in rows:
                source = raw_root / row["image_relpath"]
                image_name = Path(row["image_relpath"]).name
                link_or_copy(source, image_dir / image_name)
                output.write(f"{image_name}\t{row['text']}\n")
        print(f"{split}: wrote {len(rows)} samples")


if __name__ == "__main__":
    main()
