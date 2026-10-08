"""Train or fine-tune the Calamari CTC recognizer with Hydra."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import hydra
import numpy
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..logging.wandb import WandbLogger
from ..models.calamari.evaluate import evaluate_checkpoint
from ..models.calamari.trainer import CalamariTrainingSettings, train_calamari


LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _copies(node: DictConfig) -> dict[str, int]:
    """Plain ``{tier: count}`` dict from the ``augmentation.copies`` config node."""
    copies = OmegaConf.to_container(node, resolve=True)
    if not isinstance(copies, dict):
        raise TypeError("augmentation.copies must be a mapping of tier to copy count.")
    return {str(tier): int(count) for tier, count in copies.items()}


@hydra.main(version_base=None, config_path="../../config/calamari", config_name="configs")
def main(cfg: DictConfig) -> None:
    _set_seed(int(cfg.training.seed))
    data_root = Path(to_absolute_path(cfg.data.dir)).expanduser().resolve()
    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else "calamari"
    output_dir = Path(to_absolute_path(cfg.output.root)).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg, "output.root", str(output_dir), merge=False)
    checkpoint = (
        Path(to_absolute_path(cfg.training.checkpoint)).expanduser().resolve()
        if cfg.training.checkpoint is not None
        else None
    )
    patience = cfg.training.get("early_stopping_patience")
    settings = CalamariTrainingSettings(
        mode=str(cfg.training.mode),
        checkpoint=checkpoint,
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.batch_size),
        workers=int(cfg.training.num_workers),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        line_height=int(cfg.model.line_height),
        device=str(cfg.training.device),
        temperature=float(cfg.model.temperature),
        lstm_layers=int(cfg.model.lstm_layers),
        dropout_rate=float(cfg.model.dropout_rate),
        conv0_filters=int(cfg.model.conv0_filters),
        conv1_filters=int(cfg.model.conv1_filters),
        train_split=str(cfg.data.train_split),
        validation_split=str(cfg.data.validation_split),
        copies=_copies(cfg.augmentation.copies),
        augmentation_probability=float(cfg.augmentation.probability),
        ema_decay=float(cfg.training.ema_decay),
        logging_steps=int(cfg.logging.steps),
        warmup_ratio=float(cfg.training.warmup_ratio),
        checkpoint_top_k=int(cfg.training.checkpoint_top_k),
        early_stopping_patience=int(patience) if patience is not None else None,
        early_stopping_min_delta=float(cfg.training.get("early_stopping_min_delta", 0.0)),
        seed=int(cfg.training.seed),
    )
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    log_dir = (
        Path(to_absolute_path(cfg.logging.root)).expanduser().resolve()
        / "calamari"
        / output_dir.name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("Resolved Calamari configuration must be a mapping.")
    wandb_logger = WandbLogger(
        enabled=bool(cfg.wandb.enabled),
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity is not None else None,
        name=str(cfg.wandb.name) if cfg.wandb.name is not None else None,
        mode=str(cfg.wandb.mode),
        group=str(cfg.wandb.group) if cfg.wandb.get("group") is not None else None,
        save_dir=log_dir,
        config=resolved_config,
    )
    metrics_file = log_dir / "metrics.jsonl"

    def report(metrics: dict[str, float]) -> None:
        print(json.dumps(metrics, sort_keys=True))
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        wandb_logger.log_metrics(metrics, step=int(metrics["step"]))

    test_metrics: dict[str, float] | None = None
    try:
        _, _, best = train_calamari(data_root, output_dir, settings, report=report)
        if bool(cfg.evaluation.get("run_test_after_training", False)):
            test_metrics = _run_test_evaluation(
                cfg, data_root, output_dir, best, metrics_file, wandb_logger
            )
    finally:
        wandb_logger.finish()
    summary: dict[str, object] = {"best": best, "checkpoint": str(output_dir / "best.pt")}
    if test_metrics is not None:
        summary["test"] = test_metrics
    print(json.dumps(summary, sort_keys=True))


def _run_test_evaluation(
    cfg: DictConfig,
    data_root: Path,
    output_dir: Path,
    best: dict[str, float],
    metrics_file: Path,
    wandb_logger: WandbLogger,
) -> dict[str, float] | None:
    """Score ``best.pt`` on the test split; failures only warn so training still succeeds."""
    checkpoint = output_dir / "best.pt"
    split = str(cfg.data.test_split)
    try:
        metrics = evaluate_checkpoint(
            checkpoint,
            data_root,
            split=split,
            batch_size=int(cfg.training.batch_size),
            workers=int(cfg.training.num_workers),
            device=str(cfg.training.device),
        )
        print(
            json.dumps(
                {"test": metrics, "split": split, "checkpoint": str(checkpoint)}, sort_keys=True
            )
        )
        (output_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        step = int(best.get("step", 0))
        epoch = float(best.get("epoch", 0.0))
        record: dict[str, float] = {f"test_{key}": value for key, value in metrics.items()}
        record["step"] = step
        record["epoch"] = epoch
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        wandb_logger.log_test_metrics(metrics, step=step, epoch=epoch)
    except Exception:
        LOGGER.warning(
            "Test evaluation of %s on split %r failed.", checkpoint, split, exc_info=True
        )
        return None
    return metrics


if __name__ == "__main__":
    main()
