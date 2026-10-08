"""Training metrics for the Hugging Face TrOCR trainer."""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, TrainerCallback

from .checkpoint_mapping import OPTIMIZER_RESET_MARKER
from .encoder.lora import LoRALinear
from .token_metrics import compute_token_metrics

LOGGER = logging.getLogger(__name__)


OCR_METRIC_NAMES = (
    "cer",
    "wer",
    "exact_match",
    "sroie_precision",
    "sroie_recall",
    "sroie_f1",
)
TRAIN_OCR_METRIC_NAMES = OCR_METRIC_NAMES


class MetricsCsvCallback(TrainerCallback):
    """Write one row per validation epoch to a local CSV file."""

    fields = (
        "epoch",
        "step",
        "train_loss",
        "encoder_grad_norm",
        "encoder_lora_grad_norm",
        "decoder_grad_norm",
        "decoder_input_embedding_grad_norm",
        "DL_layer_1_grad_norm",
        "DL_layer_2_grad_norm",
        "DL_layer_3_grad_norm",
        "DL_layer_4_grad_norm",
        "eval_loss",
        *(f"train_{name}" for name in TRAIN_OCR_METRIC_NAMES),
        *(f"eval_{name}" for name in OCR_METRIC_NAMES),
        "learning_rate",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.latest_train_loss = None
        self.latest_encoder_grad_norm = None
        self.latest_encoder_lora_grad_norm = None
        self.latest_decoder_grad_norm = None
        self.latest_decoder_input_embedding_grad_norm = None
        self.latest_decoder_layer_grad_norms = {position: None for position in range(1, 5)}
        self.latest_train_ocr_metrics = {name: None for name in TRAIN_OCR_METRIC_NAMES}
        self.latest_learning_rate = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return control
        self.latest_train_loss = logs.get("train_loss", self.latest_train_loss)
        self.latest_encoder_grad_norm = logs.get(
            "train_encoder_grad_norm", self.latest_encoder_grad_norm
        )
        self.latest_encoder_lora_grad_norm = logs.get(
            "train_encoder_lora_grad_norm", self.latest_encoder_lora_grad_norm
        )
        self.latest_decoder_grad_norm = logs.get(
            "train_decoder_grad_norm", self.latest_decoder_grad_norm
        )
        self.latest_decoder_input_embedding_grad_norm = logs.get(
            "train_decoder_input_embedding_grad_norm",
            self.latest_decoder_input_embedding_grad_norm,
        )
        for position in self.latest_decoder_layer_grad_norms:
            metric_name = f"train_DL_layer_{position}_grad_norm"
            self.latest_decoder_layer_grad_norms[position] = logs.get(
                metric_name, self.latest_decoder_layer_grad_norms[position]
            )
        for name in TRAIN_OCR_METRIC_NAMES:
            metric_name = f"train_{name}"
            self.latest_train_ocr_metrics[name] = logs.get(
                metric_name, self.latest_train_ocr_metrics[name]
            )
        self.latest_learning_rate = logs.get("learning_rate", self.latest_learning_rate)
        if "eval_loss" in logs and self.writer is not None:
            self.writer.writerow(
                {
                    "epoch": logs.get("epoch", state.epoch),
                    "step": state.global_step,
                    "train_loss": self.latest_train_loss,
                    "encoder_grad_norm": self.latest_encoder_grad_norm,
                    "encoder_lora_grad_norm": self.latest_encoder_lora_grad_norm,
                    "decoder_grad_norm": self.latest_decoder_grad_norm,
                    "decoder_input_embedding_grad_norm": (
                        self.latest_decoder_input_embedding_grad_norm
                    ),
                    **{
                        f"DL_layer_{position}_grad_norm": value
                        for position, value in self.latest_decoder_layer_grad_norms.items()
                    },
                    "eval_loss": logs["eval_loss"],
                    **{
                        f"train_{name}": value
                        for name, value in self.latest_train_ocr_metrics.items()
                    },
                    **{f"eval_{name}": logs.get(f"eval_{name}") for name in OCR_METRIC_NAMES},
                    "learning_rate": self.latest_learning_rate,
                }
            )
            self.file.flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.file is not None:
            self.file.close()
        return control


class TrOCRTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that logs generated OCR and gradient metrics."""

    _SUPPRESSED_KEYS = frozenset(
        {
            "eval_steps_per_second",
            "eval_samples_per_second",
            "eval_runtime",
        }
    )

    def __init__(
        self,
        *args,
        checkpoint_top_k: int = 2,
        language_eval_datasets: dict[str, Dataset] | None = None,
        metric_reporter: Callable[[dict[str, float], int], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if checkpoint_top_k not in (1, 2):
            raise ValueError("checkpoint_top_k must be either 1 or 2.")
        self.model_accepts_loss_kwargs = False
        self.checkpoint_top_k = checkpoint_top_k
        self._pending_train_metrics: dict[str, float] = {}
        self._latest_checkpoint_metric: float | None = None
        self._ranked_checkpoint_metrics: list[float] = []
        self.language_eval_datasets = language_eval_datasets or {}
        self.metric_reporter = metric_reporter
        self._load_checkpoint_ranking()

    def _load_optimizer_and_scheduler(self, checkpoint: str) -> None:
        """Restore the scheduler after a compatibility resume resets AdamW."""
        checkpoint_dir = Path(checkpoint)
        if not (checkpoint_dir / OPTIMIZER_RESET_MARKER).is_file():
            super()._load_optimizer_and_scheduler(checkpoint)
            return

        scheduler_path = checkpoint_dir / "scheduler.pt"
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("Optimizer and scheduler must exist before resume.")
        if not scheduler_path.is_file():
            raise FileNotFoundError("The compatibility resume checkpoint is missing scheduler.pt.")

        scheduler_state = torch.load(
            scheduler_path,
            map_location="cpu",
            weights_only=False,
        )
        self.lr_scheduler.load_state_dict(scheduler_state)
        learning_rates = self.lr_scheduler.get_last_lr()
        if len(learning_rates) != len(self.optimizer.param_groups):
            raise RuntimeError("Resumed scheduler and optimizer have different parameter groups.")
        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            parameter_group["lr"] = learning_rate
        LOGGER.warning("Resuming model, trainer, and scheduler state with fresh AdamW momentum.")

    @property
    def _checkpoint_manifest_path(self) -> Path:
        return Path(self.args.output_dir) / "checkpoint_ranking.json"

    def _load_checkpoint_ranking(self) -> None:
        if not self._checkpoint_manifest_path.is_file():
            return
        manifest = json.loads(self._checkpoint_manifest_path.read_text(encoding="utf-8"))
        self._ranked_checkpoint_metrics = [
            float(checkpoint["eval_cer"])
            for checkpoint in manifest.get("checkpoints", [])
            if (Path(self.args.output_dir) / checkpoint["name"]).is_dir()
        ][: self.checkpoint_top_k]

    def _write_checkpoint_ranking(self) -> None:
        names = ("best", "second-best")
        checkpoints = [
            {"name": names[index], "eval_cer": metric}
            for index, metric in enumerate(self._ranked_checkpoint_metrics)
        ]
        self._checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_manifest_path.write_text(
            json.dumps(
                {
                    "metric": "eval_cer",
                    "greater_is_better": False,
                    "checkpoints": checkpoints,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _update_checkpoint_state_path(
        checkpoint_dir: Path,
        best_checkpoint_dir: Path,
    ) -> None:
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.is_file():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["best_model_checkpoint"] = str(best_checkpoint_dir)
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _determine_best_metric(self, metrics, trial) -> bool:
        metric = metrics.get("eval_cer")
        self._latest_checkpoint_metric = float(metric) if metric is not None else None
        return super()._determine_best_metric(metrics, trial)

    def _save_checkpoint(self, model, trial) -> None:
        metric = self._latest_checkpoint_metric
        if metric is None or not math.isfinite(metric):
            return

        insertion_index = sum(
            existing_metric <= metric for existing_metric in self._ranked_checkpoint_metrics
        )
        if (
            len(self._ranked_checkpoint_metrics) >= self.checkpoint_top_k
            and insertion_index >= self.checkpoint_top_k
        ):
            return

        super()._save_checkpoint(model, trial)
        run_dir = Path(self._get_output_dir(trial=trial))
        candidate_dir = run_dir / f"checkpoint-{self.state.global_step}"
        best_dir = run_dir / "best"
        second_best_dir = run_dir / "second-best"

        if insertion_index == 0:
            if self.checkpoint_top_k == 2:
                if second_best_dir.exists():
                    shutil.rmtree(second_best_dir)
                if best_dir.exists():
                    best_dir.rename(second_best_dir)
                    self._update_checkpoint_state_path(
                        second_best_dir,
                        second_best_dir,
                    )
            elif best_dir.exists():
                shutil.rmtree(best_dir)
            candidate_dir.rename(best_dir)
            self._update_checkpoint_state_path(best_dir, best_dir)
        else:
            if second_best_dir.exists():
                shutil.rmtree(second_best_dir)
            candidate_dir.rename(second_best_dir)
            self._update_checkpoint_state_path(second_best_dir, best_dir)

        self._ranked_checkpoint_metrics.insert(insertion_index, metric)
        self._ranked_checkpoint_metrics = self._ranked_checkpoint_metrics[: self.checkpoint_top_k]
        self.state.best_model_checkpoint = str(best_dir)
        self._write_checkpoint_ranking()

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        filtered = {k: v for k, v in logs.items() if k not in self._SUPPRESSED_KEYS}
        if "loss" in filtered and self._pending_train_metrics:
            filtered = {**self._pending_train_metrics, **filtered}
            self._pending_train_metrics.clear()
        if "loss" in filtered:
            filtered["train_loss"] = filtered.pop("loss")
        # HuggingFace injects ``epoch`` inside ``Trainer.log``. Set it here first so
        # the W&B reporter (and ``define_metric(..., step_metric="epoch")``) see the
        # real epoch instead of defaulting every point to 0.
        if self.state.epoch is not None:
            filtered["epoch"] = float(self.state.epoch)
        self._report_metrics(filtered)
        super().log(filtered, start_time)

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        """Evaluate the val split, plus per-language subsets when there are two or more."""
        metrics = super().evaluate(*args, **kwargs)
        language_metrics = self._language_eval_metrics()
        if language_metrics:
            self.log(language_metrics)
            metrics.update(language_metrics)
        return metrics

    def _language_eval_metrics(self) -> dict[str, float]:
        if len(self.language_eval_datasets) < 2:
            return {}
        language_metrics: dict[str, float] = {}
        for language, dataset in self.language_eval_datasets.items():
            output = self.evaluation_loop(
                self._language_eval_dataloader(language, dataset),
                description=f"{language} evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                metric_key_prefix=f"eval_{language}",
            )
            language_metrics.update(output.metrics)
        return language_metrics

    def _language_eval_dataloader(self, language: str, dataset: Dataset):
        cache_key = f"language_{language}"
        cached = getattr(self, "_eval_dataloaders", {}).get(cache_key)
        if cached is not None and self.args.dataloader_persistent_workers:
            return cached
        return self._get_dataloader(
            dataset=dataset,
            description=f"{language} evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=cache_key,
        )

    def _report_metrics(self, metrics: dict[str, float]) -> None:
        """Forward standard metric names through the configured reporter."""
        if not self.state.is_world_process_zero or self.metric_reporter is None:
            return
        self.metric_reporter(metrics, step=int(self.state.global_step))

    def _gradient_norm(self, parameters) -> float:
        squared_norm = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in parameters
            if parameter.grad is not None
        )
        accelerator = getattr(self, "accelerator", None)
        scaler = getattr(accelerator, "scaler", None)
        if scaler is None:
            scaler = getattr(self, "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None else 1.0
        return squared_norm**0.5 / scale

    def _gradient_metrics(self) -> dict[str, float]:
        decoder_layers = self.model.decoder.model.decoder.layers
        if len(decoder_layers) < 4:
            raise ValueError("TrOCR decoder must have at least four transformer layers.")

        lora_parameters = (
            parameter
            for module in self.model.encoder.modules()
            if isinstance(module, LoRALinear)
            for parameter in (*module.lora_a.parameters(), *module.lora_b.parameters())
        )
        metrics = {
            "train_encoder_grad_norm": self._gradient_norm(self.model.encoder.parameters()),
            "train_encoder_lora_grad_norm": self._gradient_norm(lora_parameters),
            "train_decoder_grad_norm": self._gradient_norm(self.model.decoder.parameters()),
            "train_decoder_input_embedding_grad_norm": self._gradient_norm(
                self.model.decoder.get_input_embeddings().parameters()
            ),
        }
        for position, layer in enumerate(decoder_layers[-4:], start=1):
            metrics[f"train_DL_layer_{position}_grad_norm"] = self._gradient_norm(
                layer.parameters()
            )
        return metrics

    def training_step(self, model, inputs, num_items_in_batch=None):
        should_log = (self.state.global_step + 1) % self.args.logging_steps == 0
        metrics = {}
        if should_log:
            batch = self._prepare_inputs(dict(inputs))
            was_training = model.training
            model.eval()
            with torch.no_grad(), self.autocast_smart_context_manager():
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"],
                    max_length=(
                        self.args.generation_max_length or model.generation_config.max_length
                    ),
                )
            batch_metrics = compute_token_metrics(
                generated_ids,
                batch["labels"],
                self.processing_class,
            )
            metrics.update(
                {f"train_{name}": batch_metrics[name] for name in TRAIN_OCR_METRIC_NAMES}
            )
            if was_training:
                model.train()

        loss = super().training_step(model, inputs, num_items_in_batch)
        if should_log:
            metrics.update(self._gradient_metrics())
            self._pending_train_metrics = metrics
        return loss
