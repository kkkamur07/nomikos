"""Smoke tests for the TrOCR logit distillation script."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

SCRIPT = (Path(__file__).resolve().parents[2]
          / "scripts" / "distill" / "distill_trocr.py")
LETTERS = "αβγδεζηθABXYZabxyz019"


def make_teacher_dir(path: Path) -> Path:
    """Build a tiny random teacher from the real base configs."""
    source = Path(snapshot_download(
        "microsoft/trocr-base-handwritten",
        allow_patterns=["*.json", "merges.txt"]))
    config = VisionEncoderDecoderConfig.from_pretrained(str(source))
    config.encoder.num_hidden_layers = 2
    config.decoder.decoder_layers = 2
    teacher = VisionEncoderDecoderModel(config)
    processor = TrOCRProcessor.from_pretrained(str(source))
    teacher.save_pretrained(path)
    processor.save_pretrained(path)
    return path


def make_data_dir(path: Path, n_train: int = 16, n_train_offset: int = 0) -> Path:
    """Generate synthetic line images and manifests in the repo layout."""
    rng = random.Random(7)  # noqa: S311 - deterministic synthetic test data
    image_dir = path / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    for index in range(n_train + 8):
        text = "".join(rng.choice(LETTERS) for _ in range(rng.randint(6, 14)))
        name = f"line-{index + n_train_offset:04d}.png"
        image = Image.new("RGB", (384, 64), "white")
        ImageDraw.Draw(image).text((10, 20), text, fill="black")
        image.save(image_dir / name)
        split = "train" if index < n_train else "val"
        rows[split].append(f"{name}\t{text}")
    (path / "gt_train.txt").write_text("\n".join(rows["train"]) + "\n",
                                        encoding="utf-8")
    (path / "gt_val.txt").write_text("\n".join(rows["val"]) + "\n",
                                      encoding="utf-8")
    return path


def run_script(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd, capture_output=True, text=True, timeout=1200, env=env)


def test_distill_smoke(tmp_path: Path) -> None:
    teacher_dir = make_teacher_dir(tmp_path / "teacher")
    data_dir = make_data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"
    result = run_script([
        "--teacher", str(teacher_dir), "--data-dir", str(data_dir),
        "--out", str(out_dir), "--mode", "smoke",
        "--max-steps", "4", "--eval-every", "2", "--batch-size", "4",
        "--enc-hidden", "64", "--enc-layers", "1", "--enc-heads", "2",
        "--dec-dim", "64", "--dec-layers", "1", "--dec-heads", "2",
        "--dec-ffn", "128", "--embed-init", "pca",
        "--no-bf16", "--num-workers", "0",
    ], cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]
    assert "missing=0" in result.stdout

    records = [(out_dir / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    parsed = [json.loads(line) for line in records[0]]
    teacher_records = [r for r in parsed if r.get("type") == "teacher_eval"]
    train_records = [r for r in parsed if r.get("type") == "train"]
    assert len(teacher_records) == 1
    assert len(train_records) >= 1
    assert all(math.isfinite(r["loss_kd"]) for r in train_records)

    model = VisionEncoderDecoderModel.from_pretrained(out_dir / "final")
    processor = TrOCRProcessor.from_pretrained(out_dir / "final")
    image = Image.open(data_dir / "image" / "line-0000.png").convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    with torch.no_grad():
        generated = model.generate(pixel_values, max_length=50, num_beams=1)
    assert torch.is_tensor(generated) and generated.ndim == 2


def test_grad_accum_counts_optimizer_steps(tmp_path: Path) -> None:
    teacher_dir = make_teacher_dir(tmp_path / "teacher")
    data_dir = make_data_dir(tmp_path / "data")
    out_dir = tmp_path / "out"
    result = run_script([
        "--teacher", str(teacher_dir), "--data-dir", str(data_dir),
        "--out", str(out_dir), "--mode", "smoke",
        "--max-steps", "3", "--eval-every", "3", "--log-every", "1",
        "--batch-size", "2", "--grad-accum", "2", "--warmup-ratio", "0",
        "--enc-hidden", "64", "--enc-layers", "1", "--enc-heads", "2",
        "--dec-dim", "64", "--dec-layers", "1", "--dec-heads", "2",
        "--dec-ffn", "128", "--embed-init", "pca",
        "--no-bf16", "--num-workers", "0",
    ], cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]

    parsed = [json.loads(line) for line in (out_dir / "metrics.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    train_records = [r for r in parsed if r.get("type") == "train"]
    assert [r["step"] for r in train_records] == [1, 2, 3]
    assert train_records[-1]["lr"] < train_records[0]["lr"]
    eval_records = [r for r in parsed if r.get("type") == "eval"]
    assert any(r["step"] == 3 for r in eval_records)


def test_param_check_defaults(tmp_path: Path) -> None:
    teacher_dir = make_teacher_dir(tmp_path / "teacher")
    result = run_script([
        "--teacher", str(teacher_dir), "--data-dir", str(tmp_path),
        "--out", str(tmp_path / "out"), "--mode", "smoke", "--param-check",
    ], cwd=tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]
    match = re.search(r"total=(\d+)", result.stdout)
    assert match is not None, result.stdout[-2000:]
    total = int(match.group(1))
    assert 20_000_000 <= total <= 25_000_000
