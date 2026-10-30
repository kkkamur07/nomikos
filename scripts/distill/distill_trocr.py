"""Logit distillation of a fine-tuned TrOCR-base teacher into a 20-25M student.

Plain knowledge distillation: KL divergence on temperature-softened teacher
logits plus cross-entropy on the ground truth labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from src.metrics.metrics import compute_text_metrics

MIN_PARAMS = 20_000_000
MAX_PARAMS = 25_000_000

_jsonl_fh = None


def log(message: str, record: dict[str, Any] | None = None) -> None:
    """Write a message to stdout and optionally a record to metrics.jsonl."""
    print(message, flush=True)
    if record is not None and _jsonl_fh is not None:
        _jsonl_fh.write(json.dumps(record) + "\n")
        _jsonl_fh.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distil a fine-tuned TrOCR-base teacher into a small student."
    )
    parser.add_argument("--teacher", type=Path, required=True, help="Teacher checkpoint directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        required=True,
        help="Data directory with image/ and gt_*.txt (repeatable).",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for checkpoints and metrics."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "live"],
        required=True,
        help="smoke: tiny debug run; live: full training.",
    )
    parser.add_argument("--train-gt", default="gt_train.txt", help="Train manifest file name.")
    parser.add_argument("--val-gt", default="gt_val.txt", help="Validation manifest file name.")
    parser.add_argument(
        "--image-subdir", default="image", help="Image subdirectory name inside each data dir."
    )
    parser.add_argument("--enc-hidden", type=int, default=192, help="Student encoder hidden size.")
    parser.add_argument("--enc-layers", type=int, default=12, help="Student encoder layers.")
    parser.add_argument("--enc-heads", type=int, default=3, help="Student encoder attention heads.")
    parser.add_argument("--dec-dim", type=int, default=256, help="Student decoder model dimension.")
    parser.add_argument("--dec-layers", type=int, default=4, help="Student decoder layers.")
    parser.add_argument("--dec-heads", type=int, default=4, help="Student decoder attention heads.")
    parser.add_argument("--dec-ffn", type=int, default=1024, help="Student decoder FFN dimension.")
    parser.add_argument(
        "--encoder-init",
        choices=["scratch", "deit-tiny"],
        default="scratch",
        help="Init student encoder from scratch or deit-tiny.",
    )
    parser.add_argument(
        "--embed-init",
        choices=["pca", "random"],
        default="pca",
        help="Init student token embedding via PCA or randomly.",
    )
    parser.add_argument(
        "--param-check", action="store_true", help="Print student parameter counts and exit."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dataset stats and parameter counts, then exit.",
    )
    parser.add_argument("--temperature", type=float, default=2.5, help="Distillation temperature.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Weight of the KD loss (1-alpha for cross-entropy).",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay.")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Fraction of steps for linear warmup."
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (default: 8 smoke, 32 live).",
    )
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Epochs (default: 30 in both modes)."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap on training steps (default: 30 smoke, none live).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate every N steps (default: 15 smoke, 500 live).",
    )
    parser.add_argument(
        "--log-every", type=int, default=None, help="Log every N steps (default: 5 smoke, 50 live)."
    )
    parser.add_argument(
        "--max-target-length", type=int, default=50, help="Maximum label length including EOS."
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=1111, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap train samples (default: 256 smoke, all live).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Cap val samples (default: 64 smoke, all live).",
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 for the teacher and student on CUDA.",
    )
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable bfloat16.")
    parser.add_argument("--resume", action="store_true", help="Resume from <out>/last.pt.")
    return parser


def apply_mode_presets(args: argparse.Namespace) -> None:
    """Fill unset training-size flags from the smoke/live presets."""
    presets = {
        "smoke": {
            "batch_size": 8,
            "epochs": 30,
            "max_steps": 30,
            "eval_every": 15,
            "log_every": 5,
            "max_train_samples": 256,
            "max_val_samples": 64,
        },
        "live": {
            "batch_size": 32,
            "epochs": 30,
            "max_steps": None,
            "eval_every": 500,
            "log_every": 50,
            "max_train_samples": None,
            "max_val_samples": None,
        },
    }
    for key, value in presets[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class DistillLineDataset(Dataset):
    """Line images and texts mirroring the repo LineDataset layout."""

    def __init__(self, data_dir: Path, gt_file: str, image_subdir: str) -> None:
        self.image_dir = data_dir / image_subdir
        rows = (data_dir / gt_file).read_text(encoding="utf-8").splitlines()
        self.missing = 0
        self.samples: list[tuple[str, str]] = []
        for row in rows:
            if not row:
                continue
            image_name, text = row.split("\t", 1)
            if not (self.image_dir / image_name).is_file():
                self.missing += 1
                continue
            self.samples.append((image_name, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_name, text = self.samples[index]
        image = Image.open(self.image_dir / image_name).convert("RGB")
        return {"image": image, "text": text}


def make_collator(processor: TrOCRProcessor, max_target_length: int) -> Any:
    """Batch images and labels with the repo label convention."""
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define EOS and PAD tokens.")
    if max_target_length < 1:
        raise ValueError("max_target_length must be at least one.")

    def collate(features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        images = [feature["image"] for feature in features]
        texts = [str(feature["text"]) for feature in features]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length - 1,
            padding=False,
        )
        labels = torch.full(
            (len(texts), max_target_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(encoded.input_ids):
            sequence = [*token_ids, tokenizer.eos_token_id]
            labels[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

    return collate


def patch_encoder_decoder_ids(
    config: VisionEncoderDecoderConfig,
) -> VisionEncoderDecoderConfig:
    """Backfill top-level token ids from the decoder subconfig.

    The released TrOCR configs keep these ids only inside the decoder
    subconfig, but the encoder-decoder forward reads them top-level.
    """
    decoder = config.decoder
    defaults = (("decoder_start_token_id", 2), ("pad_token_id", 1), ("eos_token_id", 2))
    for name, default in defaults:
        try:
            getattr(config, name)
        except AttributeError:
            setattr(config, name, getattr(decoder, name, default))
    return config


def build_student(
    teacher: VisionEncoderDecoderModel, args: argparse.Namespace
) -> VisionEncoderDecoderModel:
    """Build the small student by shrinking copies of the teacher configs."""
    enc_cfg = deepcopy(teacher.config.encoder)
    enc_cfg.hidden_size = args.enc_hidden
    enc_cfg.num_hidden_layers = args.enc_layers
    enc_cfg.num_attention_heads = args.enc_heads
    enc_cfg.intermediate_size = 4 * args.enc_hidden
    dec_cfg = deepcopy(teacher.config.decoder)
    dec_cfg.d_model = args.dec_dim
    dec_cfg.decoder_layers = args.dec_layers
    dec_cfg.decoder_attention_heads = args.dec_heads
    dec_cfg.decoder_ffn_dim = args.dec_ffn
    dec_cfg.cross_attention_hidden_size = args.enc_hidden
    dec_cfg.tie_word_embeddings = True
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
    teacher_decoder = teacher.config.decoder
    config.decoder_start_token_id = getattr(
        teacher.config,
        "decoder_start_token_id",
        getattr(teacher_decoder, "decoder_start_token_id", 2),
    )
    config.pad_token_id = getattr(
        teacher.config, "pad_token_id", getattr(teacher_decoder, "pad_token_id", 1)
    )
    config.eos_token_id = getattr(
        teacher.config, "eos_token_id", getattr(teacher_decoder, "eos_token_id", 2)
    )
    return VisionEncoderDecoderModel(config)


def count_params(model: torch.nn.Module) -> tuple[int, int, int]:
    """Count unique parameters (tied weights once), split encoder/decoder."""
    seen: set[int] = set()
    total = 0
    encoder = 0
    decoder = 0
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
        if name.startswith("encoder."):
            encoder += param.numel()
        elif name.startswith("decoder."):
            decoder += param.numel()
    return total, encoder, decoder


def tied_embed_size(model: torch.nn.Module, vocab_size: int) -> int:
    """Return the parameter count of the tied token embedding."""
    for name, param in model.named_parameters():
        if param.shape[0] == vocab_size and ("embed" in name or "shared" in name):
            return param.numel()
    return 0


def find_embed_weight(
    model: torch.nn.Module, vocab_size: int, embed_dim: int
) -> torch.nn.Parameter | None:
    for name, param in model.named_parameters():
        if tuple(param.shape) == (vocab_size, embed_dim) and ("embed" in name or "shared" in name):
            return param
    return None


def init_embed_pca(
    student: VisionEncoderDecoderModel,
    teacher: VisionEncoderDecoderModel,
    dec_dim: int,
) -> None:
    """Init the student token embedding from the teacher via top PCs."""
    vocab_size = int(teacher.config.decoder.vocab_size)
    teacher_dim = int(teacher.config.decoder.d_model)
    teacher_emb = find_embed_weight(teacher, vocab_size, teacher_dim)
    if teacher_emb is None:
        raise SystemExit("Could not find the teacher token embedding.")
    with torch.no_grad():
        teacher_mat = teacher_emb.detach().float()
        centered = teacher_mat - teacher_mat.mean(dim=0, keepdim=True)
        total = centered.pow(2).sum()
        _, singular, vecs = torch.pca_lowrank(centered, q=dec_dim)
        explained = float(singular.pow(2).sum() / total)
        projected = centered @ vecs
        teacher_norm = teacher_mat.norm(dim=1).mean()
        student_norm = projected.norm(dim=1).mean()
        projected = projected * (teacher_norm / student_norm)
        copied = 0
        seen: set[int] = set()
        for _name, param in student.named_parameters():
            if tuple(param.shape) == (vocab_size, dec_dim) and id(param) not in seen:
                seen.add(id(param))
                if "embed" in _name or "shared" in _name or "lm_head" in _name:
                    param.copy_(projected)
                    copied += 1
    log(f"embed-init pca: explained variance ratio {explained:.4f}, copied into {copied} tensor(s)")


def init_encoder_deit(student: VisionEncoderDecoderModel, args: argparse.Namespace) -> None:
    """Load deit-tiny weights into the student encoder, resizing positions."""
    from transformers import DeiTModel

    reference = DeiTModel.from_pretrained("facebook/deit-tiny-patch16-224").config
    expected = (reference.hidden_size, reference.num_hidden_layers, reference.num_attention_heads)
    actual = (args.enc_hidden, args.enc_layers, args.enc_heads)
    if expected != actual:
        raise SystemExit(
            f"deit-tiny shape {expected} does not match "
            f"--enc-hidden/--enc-layers/--enc-heads {actual}."
        )
    donor = DeiTModel.from_pretrained("facebook/deit-tiny-patch16-224")
    donor_state = donor.state_dict()
    target = student.encoder.state_dict()
    pos_key = "embeddings.position_embeddings"
    student_pos = target[pos_key]
    donor_pos = donor_state["deit." + pos_key]
    _, student_tokens, _ = student_pos.shape
    _, donor_tokens, _ = donor_pos.shape
    student_grid = math.isqrt(int(student_tokens) - 1)
    donor_grid = math.isqrt(int(donor_tokens) - 1)
    if student_grid * student_grid + 1 != student_tokens:
        raise SystemExit("Student position embeddings are not a square grid.")
    cls_token, grid = donor_pos[:, :1, :], donor_pos[:, 1:, :]
    grid = grid.reshape(1, donor_grid, donor_grid, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid, size=(student_grid, student_grid), mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, student_grid * student_grid, -1)
    target[pos_key].copy_(torch.cat([cls_token, grid], dim=1))
    copied = 1
    for key, value in target.items():
        if key == pos_key:
            continue
        donor_key = "deit." + key
        if donor_key not in donor_state:
            continue
        donor_value = donor_state[donor_key]
        if tuple(donor_value.shape) == tuple(value.shape):
            value.copy_(donor_value)
            copied += 1
    log(f"encoder-init deit-tiny: copied {copied} tensor(s)")


def report_param_counts(student: VisionEncoderDecoderModel) -> int:
    """Print student parameter counts and return the total."""
    total, encoder, decoder = count_params(student)
    vocab_size = int(student.config.decoder.vocab_size)
    tied = tied_embed_size(student, vocab_size)
    log(f"params total={total} encoder={encoder} decoder={decoder} tied_embed={tied}")
    return total


def check_param_range(total: int) -> None:
    if MIN_PARAMS <= total <= MAX_PARAMS:
        log(f"param-check passed: {total} within [{MIN_PARAMS}, {MAX_PARAMS}]")
    else:
        log(f"param-check FAILED: {total} outside [{MIN_PARAMS}, {MAX_PARAMS}]")
        raise SystemExit(1)


def load_split(
    data_dirs: list[Path], gt_file: str, image_subdir: str, cap: int | None
) -> tuple[Dataset, list[DistillLineDataset]]:
    datasets = [DistillLineDataset(d, gt_file, image_subdir) for d in data_dirs]
    combined: Dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if cap is not None:
        combined = Subset(combined, range(min(cap, len(combined))))
    return combined, datasets


def dry_run(
    args: argparse.Namespace,
    teacher: VisionEncoderDecoderModel,
) -> None:
    for data_dir in args.data_dir:
        train_ds = DistillLineDataset(data_dir, args.train_gt, args.image_subdir)
        val_ds = DistillLineDataset(data_dir, args.val_gt, args.image_subdir)
        log(
            f"{data_dir}: train={len(train_ds)} val={len(val_ds)} "
            f"missing_train={train_ds.missing} missing_val={val_ds.missing}"
        )
        for name, text in train_ds.samples[:3]:
            log(f"  sample {name}\t{text}")
        first_image = Image.open(train_ds.image_dir / train_ds.samples[0][0]).convert("RGB")
        log(f"  first image size={first_image.size}")
    student = build_student(teacher, args)
    check_param_range(report_param_counts(student))


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KD loss plus cross-entropy, computed in float32 over masked tokens."""
    student_flat = student_logits.float().reshape(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.float().reshape(-1, teacher_logits.size(-1))
    labels_flat = labels.reshape(-1)
    mask = labels_flat != -100
    student_masked = student_flat[mask]
    teacher_masked = teacher_flat[mask]
    soft_teacher = F.softmax(teacher_masked / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_masked / temperature, dim=-1)
    loss_kd = (
        F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * temperature * temperature
    )
    loss_ce = F.cross_entropy(student_flat, labels_flat, ignore_index=-100)
    total = alpha * loss_kd + (1.0 - alpha) * loss_ce
    return total, loss_kd, loss_ce


@torch.no_grad()
def evaluate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    loader: DataLoader,
    device: torch.device,
    max_target_length: int,
    use_bf16: bool = False,
) -> dict[str, float]:
    """Greedy-decode the val set and score CER, WER, exact match and CE."""
    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    ce_sum = 0.0
    ce_tokens = 0
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with context:
            outputs = model(pixel_values=pixel_values, labels=labels)
            generated = model.generate(pixel_values, max_length=max_target_length, num_beams=1)
        ce_sum += float(outputs.loss.float()) * int((labels != -100).sum())
        ce_tokens += int((labels != -100).sum())
        texts = processor.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = labels.clone()
        decoded_labels[decoded_labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.tokenizer.batch_decode(
            decoded_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predictions.extend(texts)
        references.extend(refs)
    metrics = compute_text_metrics(references, predictions)
    metrics["val_loss_ce"] = ce_sum / max(ce_tokens, 1)
    return metrics


def run_eval_and_save(
    student: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    state: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    use_bf16: bool = False,
) -> None:
    metrics = evaluate(student, processor, val_loader, device, args.max_target_length, use_bf16)
    record = {"type": "eval", "step": step, **metrics}
    log(
        f"eval step={step} cer={metrics['cer']:.4f} wer={metrics['wer']:.4f} "
        f"exact_match={metrics['exact_match']:.4f} "
        f"val_loss_ce={metrics['val_loss_ce']:.4f}",
        record,
    )
    state["step"] = step
    torch.save(
        state
        | {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_cer": state["best_cer"],
        },
        args.out / "last.pt",
    )
    if metrics["cer"] < state["best_cer"]:
        state["best_cer"] = metrics["cer"]
        best_dir = args.out / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        log(f"new best CER {metrics['cer']:.4f}, saved to {best_dir}")
    student.train()


def train(args: argparse.Namespace) -> None:
    global _jsonl_fh
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    teacher = VisionEncoderDecoderModel.from_pretrained(args.teacher)
    patch_encoder_decoder_ids(teacher.config)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    if use_bf16:
        teacher.to(torch.bfloat16)
    processor = TrOCRProcessor.from_pretrained(str(args.teacher))

    student = build_student(teacher, args)
    total, encoder, decoder = count_params(student)
    log(f"student params total={total} encoder={encoder} decoder={decoder}")
    if args.encoder_init == "deit-tiny":
        init_encoder_deit(student, args)
    if args.embed_init == "pca":
        init_embed_pca(student, teacher, args.dec_dim)
    student.to(device)
    student.train()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "run_config.json", "w", encoding="utf-8") as handle:

        def stringify(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [stringify(item) for item in value]
            return value

        resolved = {k: stringify(v) for k, v in vars(args).items()}
        resolved["param_total"] = total
        resolved["param_encoder"] = encoder
        resolved["param_decoder"] = decoder
        json.dump(resolved, handle, indent=2)

    collate = make_collator(processor, args.max_target_length)
    train_ds, train_parts = load_split(
        args.data_dir, args.train_gt, args.image_subdir, args.max_train_samples
    )
    val_ds, val_parts = load_split(
        args.data_dir, args.val_gt, args.image_subdir, args.max_val_samples
    )
    for data_dir, part in zip(args.data_dir, train_parts, strict=True):
        log(f"data {data_dir} train={len(part)} missing={part.missing}")
    for data_dir, part in zip(args.data_dir, val_parts, strict=True):
        log(f"data {data_dir} val={len(part)} missing={part.missing}")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)  # type: ignore[arg-type]
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)  # type: ignore[arg-type]

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    if args.max_steps is not None:
        total_steps = int(args.max_steps)
    else:
        total_steps = int(args.epochs) * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_factor(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    state: dict[str, Any] = {"best_cer": float("inf"), "step": 0}
    start_step = 0
    if args.resume and (args.out / "last.pt").is_file():
        checkpoint = torch.load(args.out / "last.pt", map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        state["best_cer"] = checkpoint.get("best_cer", float("inf"))
        start_step = int(checkpoint.get("step", 0))
        log(f"resumed from step {start_step}")

    with open(args.out / "metrics.jsonl", "a", encoding="utf-8") as handle:
        _jsonl_fh = handle
        teacher_metrics = evaluate(
            teacher, processor, val_loader, device, args.max_target_length, use_bf16
        )
        log(
            f"teacher cer={teacher_metrics['cer']:.4f} "
            f"wer={teacher_metrics['wer']:.4f} "
            f"exact_match={teacher_metrics['exact_match']:.4f}",
            {"type": "teacher_eval", **teacher_metrics},
        )

        step = start_step
        micro = 0
        optimizer.zero_grad()
        stop = False
        for _epoch in range(int(args.epochs)):
            if stop:
                break
            for batch in train_loader:
                if step >= total_steps:
                    stop = True
                    break
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                )
                with context:
                    student_out = student(pixel_values=pixel_values, labels=labels)
                    with torch.no_grad():
                        teacher_out = teacher(pixel_values=pixel_values, labels=labels)
                    loss, loss_kd, loss_ce = distillation_loss(
                        student_out.logits, teacher_out.logits, labels, args.temperature, args.alpha
                    )
                    loss = loss / args.grad_accum
                loss.backward()
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % args.log_every == 0 or step == total_steps:
                        lr = optimizer.param_groups[0]["lr"]
                        log(
                            f"step={step} lr={lr:.2e} "
                            f"loss={float(loss) * args.grad_accum:.4f} "
                            f"loss_kd={float(loss_kd):.4f} "
                            f"loss_ce={float(loss_ce):.4f}",
                            {
                                "type": "train",
                                "step": step,
                                "lr": lr,
                                "loss": float(loss) * args.grad_accum,
                                "loss_kd": float(loss_kd),
                                "loss_ce": float(loss_ce),
                            },
                        )
                    if step % args.eval_every == 0:
                        run_eval_and_save(
                            student,
                            processor,
                            val_loader,
                            device,
                            args,
                            step,
                            state,
                            optimizer,
                            scheduler,
                            use_bf16,
                        )
            run_eval_and_save(
                student,
                processor,
                val_loader,
                device,
                args,
                step,
                state,
                optimizer,
                scheduler,
                use_bf16,
            )
        _jsonl_fh = None

    final_dir = args.out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    log(f"saved final student to {final_dir}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    apply_mode_presets(args)
    teacher = VisionEncoderDecoderModel.from_pretrained(args.teacher)
    patch_encoder_decoder_ids(teacher.config)
    if args.param_check:
        check_param_range(report_param_counts(build_student(teacher, args)))
        return
    if args.dry_run:
        dry_run(args, teacher)
        return
    del teacher
    train(args)


if __name__ == "__main__":
    main()
