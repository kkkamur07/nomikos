#!/usr/bin/env python3
"""VRAM smoke test: find the sweet-spot batch size for Calamari and TrOCR on RTX 3090.

Also estimates total training time for every cluster training script.

Run via SLURM (recommended):
    sbatch scripts/cluster/smoke_test.sh

Or interactively on a GPU node:
    uv run --extra train python scripts/smoke_test.py
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_VRAM_TOTAL_MB: int = 0
_SEP = "─" * 70
_N_AUGMENTATIONS = 3  # matches config/*/augmentation/default.yaml
_OVERHEAD_FACTOR = 1.15  # eval + checkpoint + data-loading overhead


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _vram_total_mb() -> int:
    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)


def _peak_mb() -> int:
    return torch.cuda.max_memory_allocated() // (1024 * 1024)


def _row(bs: int, peak_mb: int | None, step_s: float | None, status: str) -> None:
    total = _VRAM_TOTAL_MB or 1
    if status == "OK" and peak_mb is not None and step_s is not None:
        pct = peak_mb / total * 100
        bar_w = 20
        filled = round(pct / 100 * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(
            f"  batch {bs:4d} │ {peak_mb:6,}/{total:,} MB [{bar}] {pct:4.1f}%"
            f" │ {step_s:5.2f} s/step │ ✓"
        )
    else:
        print(f"  batch {bs:4d} │ {'── ' + status + ' ──':^55} │")


def _count_samples(lang: str, split: str = "train") -> int:
    """Count lines in gt_{split}.txt for a language's pretraining data."""
    gt = REPO_ROOT / "data" / "processed" / lang / "pretraining" / f"gt_{split}.txt"
    try:
        return sum(1 for _ in gt.open(encoding="utf-8"))
    except FileNotFoundError:
        return 0


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# Calamari smoke test
# ─────────────────────────────────────────────────────────────────────────────


def smoke_calamari(
    batch_sizes: list[int],
    image_width: int = 800,
    image_height: int = 48,
    num_classes: int = 426,
    steps: int = 3,
) -> tuple[int, float]:
    """Forward+backward over dummy line images.

    Returns (recommended_batch_size, seconds_per_step).
    """
    from src.models.calamari.config import default_model_config
    from src.models.calamari.model import CalamariTorchModel

    print(f"\n{'Calamari  CNN+BiLSTM':^70}")
    print(
        f"{'2.6 M params · FP32 · line images ' + str(image_width) + '×' + str(image_height):^70}"
    )
    print(_SEP)

    device = torch.device("cuda")
    best_bs, best_step_s = batch_sizes[0], 0.0

    # Load model once and warm up CUDA kernels at smallest batch
    cfg = default_model_config(classes=num_classes)
    model = CalamariTorchModel(cfg).to(device)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    _wx = torch.randint(
        0, 256, (batch_sizes[0], image_width, image_height, 1), dtype=torch.uint8, device=device
    )
    _wl = torch.full((batch_sizes[0],), image_width, dtype=torch.long, device=device)
    _wt = torch.randint(1, num_classes, (batch_sizes[0] * 10,), device=device)
    _wll = torch.full((batch_sizes[0],), 10, dtype=torch.long, device=device)
    _wout = model(_wx, _wl)
    loss_fn(
        _wout["logits"].log_softmax(-1).permute(1, 0, 2), _wt, _wout["out_len"], _wll
    ).backward()
    del _wx, _wl, _wt, _wll, _wout
    torch.cuda.synchronize()

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

            def _cal_step(
                _bs: int = bs,
                _opt: torch.optim.Optimizer = opt,
                _model: CalamariTorchModel = model,
            ) -> float:
                x = torch.randint(
                    0,
                    256,
                    (_bs, image_width, image_height, 1),
                    dtype=torch.uint8,
                    device=device,
                )
                lengths = torch.full((_bs,), image_width, dtype=torch.long, device=device)
                tgt = torch.randint(1, num_classes, (_bs * 10,), device=device)
                tgt_lens = torch.full((_bs,), 10, dtype=torch.long, device=device)
                t0 = time.perf_counter()
                _opt.zero_grad(set_to_none=True)
                out = _model(x, lengths)
                log_probs = out["logits"].log_softmax(-1).permute(1, 0, 2)
                loss = loss_fn(log_probs, tgt, out["out_len"], tgt_lens)
                loss.backward()
                _opt.step()
                torch.cuda.synchronize()
                return time.perf_counter() - t0

            elapsed = sum(_cal_step() for _ in range(steps))  # timed steps
            step_s = elapsed / steps
            _row(bs, _peak_mb(), step_s, "OK")
            best_bs, best_step_s = bs, step_s
            del opt
        except RuntimeError as exc:
            tag = "OOM" if "out of memory" in str(exc).lower() else f"ERR: {exc}"
            _row(bs, None, None, tag)
            torch.cuda.empty_cache()
            break

    del model
    torch.cuda.empty_cache()
    print(f"\n  ➜  Recommended batch_size = {best_bs}  ({best_step_s:.3f} s/step)")
    return best_bs, best_step_s


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _load_trocr_model(checkpoint: Path, freeze_encoder: bool, device):
    """Load the TrOCR checkpoint for VRAM probing with sane decoding defaults."""
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(str(checkpoint))
    dec_cfg = model.config.decoder
    if not getattr(model.config, "pad_token_id", None):
        model.config.pad_token_id = getattr(dec_cfg, "pad_token_id", None) or getattr(
            dec_cfg, "eos_token_id", 1
        )
    if not getattr(model.config, "decoder_start_token_id", None):
        model.config.decoder_start_token_id = getattr(dec_cfg, "bos_token_id", None) or 0
    real_vocab = dec_cfg.vocab_size
    model = model.half().to(device).train()
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, trainable, real_vocab


def _warmup_trocr_kernels(model, *, batch_size: int, label_seq_len: int, device) -> None:
    """One untimed forward at the smallest batch size to compile CUDA kernels."""
    _warmup_pv = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
    _warmup_lb = torch.zeros(batch_size, label_seq_len, dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        model(pixel_values=_warmup_pv, labels=_warmup_lb)
    del _warmup_pv, _warmup_lb
    torch.cuda.synchronize()


def _trocr_step(
    model, optimizer, trainable, *, batch_size: int, label_seq_len: int, vocab_size: int, device
) -> float:
    """One timed forward+backward over dummy 384x384 images."""
    pixel_values = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
    labels = torch.randint(0, vocab_size, (batch_size, label_seq_len), device=device)
    labels[:, label_seq_len // 2 :] = -100
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=pixel_values, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _probe_trocr_batch(
    model, trainable, *, batch_size: int, label_seq_len: int, vocab_size: int, device, steps: int
) -> tuple[float | None, str]:
    """Time one batch size; returns (seconds per step, status) without raising."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    try:
        elapsed = sum(
            _trocr_step(
                model,
                optimizer,
                trainable,
                batch_size=batch_size,
                label_seq_len=label_seq_len,
                vocab_size=vocab_size,
                device=device,
            )
            for _ in range(steps)
        )
        return elapsed / steps, "OK"
    except RuntimeError as exc:
        tag = "OOM" if "out of memory" in str(exc).lower() else f"ERR: {exc}"
        torch.cuda.empty_cache()
        return None, tag


def smoke_trocr(
    checkpoint: Path,
    batch_sizes: list[int],
    freeze_encoder: bool,
    label_seq_len: int = 24,
    steps: int = 2,
) -> tuple[int, float]:
    """Forward+backward over dummy 384×384 images.

    Returns (recommended_batch_size, seconds_per_step).
    """
    model_tag = checkpoint.name
    param_count = "282.9 M" if "base" in model_tag else "28.9 M"
    enc_note = "encoder frozen" if freeze_encoder else "full fine-tune"
    print(f"\n{'TrOCR  ' + model_tag:^70}")
    print(f"{param_count + ' params · FP16 · ' + enc_note + ' · 384×384':^70}")
    print(_SEP)

    device = torch.device("cuda")
    best_bs, best_step_s = batch_sizes[0], 0.0

    # Load model once, so we don't re-pay disk I/O and kernel compilation per batch
    print("  loading model …", flush=True)
    model, trainable, real_vocab = _load_trocr_model(checkpoint, freeze_encoder, device)

    # Warm up CUDA kernels once at the smallest batch size before timing
    _warmup_trocr_kernels(
        model, batch_size=batch_sizes[0], label_seq_len=label_seq_len, device=device
    )
    print("  kernels compiled — probing batch sizes …", flush=True)

    for bs in batch_sizes:
        step_s, status = _probe_trocr_batch(
            model,
            trainable,
            batch_size=bs,
            label_seq_len=label_seq_len,
            vocab_size=real_vocab,
            device=device,
            steps=steps,
        )
        _row(bs, _peak_mb() if status == "OK" else None, step_s, status)
        if status != "OK" or step_s is None:
            break
        best_bs, best_step_s = bs, step_s

    del model
    torch.cuda.empty_cache()
    print(f"\n  ➜  Recommended batch_size = {best_bs}  ({best_step_s:.3f} s/step)")
    return best_bs, best_step_s


# ─────────────────────────────────────────────────────────────────────────────
# Training-time estimation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class JobSpec:
    script: str
    model: str  # "calamari" | "trocr-base" | "trocr-small"
    language: str
    epochs: int


_JOBS: list[JobSpec] = [
    # ── Calamari ──────────────────────────────────────────────────────────
    JobSpec("calamari/pretraining/armenian", "calamari", "armenian", 40),
    JobSpec("calamari/pretraining/greek", "calamari", "greek", 40),
    JobSpec("calamari/pretraining/syriac", "calamari", "syriac", 40),
    JobSpec("calamari/pretraining/combined", "calamari", "combined", 60),
    # ── TrOCR-base ────────────────────────────────────────────────────────
    JobSpec("trocr/pretraining/armenian", "trocr-base", "armenian", 40),
    JobSpec("trocr/pretraining/greek", "trocr-base", "greek", 40),
    JobSpec("trocr/pretraining/syriac", "trocr-base", "syriac", 50),
    JobSpec("trocr/pretraining/combined", "trocr-base", "combined", 60),
]


def print_time_estimates(
    calamari_bs: int,
    calamari_step_s: float,
    trocr_base_bs: int,
    trocr_base_step_s: float,
) -> None:
    print(f"\n\n{'TRAINING TIME ESTIMATES  (RTX 3090, cc-gpu-n03)':^70}")
    print("=" * 70)
    print(
        f"  {'Script':<38} {'Samples':>8}  {'Steps/ep':>8}  "
        f"{'Epoch':>7}  {'Total':>8}  {'SLURM limit':>11}"
    )
    print(_SEP)

    step_map = {"calamari": calamari_step_s, "trocr-base": trocr_base_step_s}
    bs_map = {"calamari": calamari_bs, "trocr-base": trocr_base_bs}

    slurm_limits = {
        "calamari/pretraining/armenian": "10:00:00",
        "calamari/pretraining/greek": "10:00:00",
        "calamari/pretraining/syriac": "10:00:00",
        "calamari/pretraining/combined": "10:00:00",
        "trocr/pretraining/armenian": "10:00:00",
        "trocr/pretraining/greek": "10:00:00",
        "trocr/pretraining/syriac": "20:00:00",
        "trocr/pretraining/combined": "10:00:00",
    }

    for job in _JOBS:
        step_s = step_map.get(job.model)
        bs = bs_map.get(job.model)
        if step_s is None or bs is None or step_s == 0:
            print(f"  {job.script:<38}  (no timing data)")
            continue

        base_samples = _count_samples(job.language)
        eff_samples = base_samples * (_N_AUGMENTATIONS + 1)
        if eff_samples == 0:
            print(f"  {job.script:<38}  (dataset not found locally)")
            continue

        steps_per_epoch = math.ceil(eff_samples / bs)
        epoch_s = steps_per_epoch * step_s
        total_s = epoch_s * job.epochs * _OVERHEAD_FACTOR
        limit = slurm_limits.get(job.script, "?")

        print(
            f"  {job.script:<38} {eff_samples:>8,}  {steps_per_epoch:>8,}  "
            f"{_fmt_duration(epoch_s):>7}  {_fmt_duration(total_s):>8}  {limit:>11}"
        )

    print()
    print(f"  Batch sizes used:  Calamari={calamari_bs}  TrOCR-base={trocr_base_bs}")
    print(f"  Overhead factor:   ×{_OVERHEAD_FACTOR} (eval + checkpointing + data-load)")
    print(
        f"  Aug multiplier:    ×{_N_AUGMENTATIONS + 1}  (n_augmentations={_N_AUGMENTATIONS} + original)"
    )
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("✗  CUDA not available — run this on a GPU node.")

    global _VRAM_TOTAL_MB
    _VRAM_TOTAL_MB = _vram_total_mb()
    props = torch.cuda.get_device_properties(0)

    print("=" * 70)
    print("  VRAM SMOKE TEST")
    print(f"  GPU    : {props.name}")
    print(f"  VRAM   : {_VRAM_TOTAL_MB:,} MB")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 70)

    trocr_base = REPO_ROOT / "trocr_checkpoints" / "trocr-base-handwritten"
    trocr_small = REPO_ROOT / "trocr_checkpoints" / "trocr-small-handwritten"

    # ── 1. Calamari: probe every 64 images up to 512 ─────────────────────────
    cal_bs, cal_step_s = smoke_calamari(batch_sizes=[64, 128, 256, 320, 448, 512])

    # ── 2. TrOCR-base (full fine-tune, all cluster pretraining scripts) ──────
    trocr_base_bs, trocr_base_step_s = (0, 0.0)
    if trocr_base.exists():
        trocr_base_bs, trocr_base_step_s = smoke_trocr(
            trocr_base,
            batch_sizes=[8, 16, 24, 32, 40],
            freeze_encoder=False,
        )
    else:
        print(f"\n[TrOCR-base] checkpoint not found at {trocr_base}, skipping.")

    # ── 3. TrOCR-small (full fine-tune) ──────────────────────────────────────
    if trocr_small.exists():
        smoke_trocr(
            trocr_small,
            batch_sizes=[32, 48, 64, 80],
            freeze_encoder=False,
        )
    else:
        print(f"\n[TrOCR-small] checkpoint not found at {trocr_small}, skipping.")

    # ── 4. Time estimates for all cluster scripts ─────────────────────────────
    print_time_estimates(cal_bs, cal_step_s, trocr_base_bs, trocr_base_step_s)

    print("\n  Done. Update config YAML files with the batch sizes above.")


if __name__ == "__main__":
    main()
