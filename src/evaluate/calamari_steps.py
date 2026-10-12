"""Dump CTC per-step softmax tables from a saved Calamari checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.calamari.checkpoint import load_calamari_checkpoint
from ..models.calamari.codec import CharacterCodec
from ..models.calamari.ctc_steps import ctc_step_report, write_ctc_step_report
from ..models.calamari.data import CalamariLineDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show CTC softmax distributions for a few lines from an old or new checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Processed split root (gt_*.txt + image/)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    model, metadata = load_calamari_checkpoint(args.checkpoint)
    codec = CharacterCodec(metadata.charset)
    dataset = CalamariLineDataset(args.data, args.split, codec, metadata.line_height)
    device = torch.device(args.device)
    report = ctc_step_report(
        model.to(device), dataset, codec, device, n_lines=args.n, top_k=args.top_k
    )
    print(report)
    if args.out is not None:
        write_ctc_step_report(report, args.out)


if __name__ == "__main__":
    main()
