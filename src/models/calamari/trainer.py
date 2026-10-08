"""Hugging Face Trainer integration for the PyTorch CTC Calamari recognizer."""

#! need to change the zero weight and bias
from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ...augmentation.augmentation import DEFAULT_COPIES
from ...early_stopping import EarlyStoppingCallback

from ...metrics.languages import language_indices
from ...metrics.metrics import (
    compute_reference_length_counts,
    compute_sequence_length_metrics,
    compute_text_metrics,
)
from .checkpoint import load_calamari_checkpoint, save_calamari_checkpoint
from .codec import CharacterCodec
from .config import default_model_config
from .ctc_steps import ctc_step_report, write_ctc_step_report
from .data import (
    CalamariAugmentedDataset,
    CalamariLineDataset,
    collect_samples,
    collate_ctc,
    repeat_syriac_2024_train,
)
from .model import CalamariTorchModel


TRAIN_OCR_METRIC_NAMES = (
    "cer",
    "wer",
    "exact_match",
    "sroie_precision",
    "sroie_recall",
    "sroie_f1",
)
EVAL_OCR_METRIC_NAMES = TRAIN_OCR_METRIC_NAMES
_EMA_FILENAME = "calamari_ema.pt"
_METADATA_FILENAME = "calamari_metadata.json"


@dataclass(frozen=True)
class CalamariTrainingSettings:
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    line_height: int
    device: str
    temperature: float
    lstm_layers: int
    dropout_rate: float = 0.3
    conv0_filters: int = 40
    conv1_filters: int = 60
    checkpoint: Path | None = None
    mode: str = "train"
    train_split: str = "train"
    validation_split: str = "val"
    copies: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_COPIES))
    augmentation_probability: float = 1.0
    ema_decay: float = 0.99
    logging_steps: int = 10
    warmup_ratio: float = 0.09
    checkpoint_top_k: int = 1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    seed: int = 1111


class ExponentialMovingAverage:
    """Maintain an evaluation-ready exponential moving average of model weights."""

    def __init__(self, model: CalamariTorchModel, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                "Calamari ema_decay must be greater than or equal to zero and less than one."
            )
        self.decay = decay
        self.model = copy.deepcopy(model).requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: CalamariTorchModel) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        for name, target_value in target.items():
            source_value = source[name].detach()
            if torch.is_floating_point(target_value):
                target_value.lerp_(source_value, 1.0 - self.decay)
            else:
                target_value.copy_(source_value)


class _EMACallback(TrainerCallback):
    """Update Calamari's EMA after each completed optimizer step."""

    def __init__(self, ema: ExponentialMovingAverage) -> None:
        self.ema = ema

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.ema.model.to(next(model.parameters()).device)
            self.ema.update(model)
        return control


class _ReportCallback(TrainerCallback):
    """Forward normalized Trainer logs to the CLI's JSONL and W&B reporter."""

    def __init__(self, report: Callable[[dict[str, float]], None] | None) -> None:
        self.report = report

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.report is None or not state.is_world_process_zero or not logs:
            return control
        metrics = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        metrics["step"] = float(state.global_step)
        if state.epoch is not None:
            metrics.setdefault("epoch", float(state.epoch))
        self.report(metrics)
        return control


class _TrainerCompatibleConfig:
    """Expose Calamari's immutable config through Trainer's mutable interface."""

    def __init__(self, calamari_config: object) -> None:
        self._calamari_config = calamari_config
        self.use_cache = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._calamari_config, name)


class CalamariTrainer(Trainer):
    """Adapt Hugging Face Trainer's lifecycle to Calamari's CTC contract."""

    def __init__(
        self,
        *args: Any,
        codec: CharacterCodec,
        ema: ExponentialMovingAverage,
        line_height: int,
        temperature: float,
        language_eval_datasets: dict[str, Dataset[dict[str, object]]],
        **kwargs: Any,
    ) -> None:
        model = kwargs.get("model")
        if not isinstance(model, CalamariTorchModel):
            raise TypeError("CalamariTrainer requires a CalamariTorchModel.")
        self.lstm_layers = sum(layer.kind == "bilstm" for layer in model.config.layers)
        model.config = _TrainerCompatibleConfig(model.config)
        super().__init__(*args, **kwargs)
        self.codec = codec
        self.ema = ema
        self.line_height = line_height
        self.temperature = temperature
        self.language_eval_datasets = language_eval_datasets
        self.loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
        self.latest_train_text_metrics: dict[str, float] | None = None
        self.latest_train_loss: float | None = None
        self.latest_learning_rate: float | None = None

    def compute_loss(
        self,
        model: CalamariTorchModel,
        inputs: dict[str, object],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        del num_items_in_batch
        outputs = model(
            _tensor(inputs["image"], "image"),
            image_lengths=_tensor(inputs["image_lengths"], "image_lengths"),
        )
        logits = outputs["logits"]
        loss = self.loss_function(
            logits.log_softmax(-1).transpose(0, 1),
            _tensor(inputs["targets"], "targets"),
            outputs["out_len"],
            _tensor(inputs["target_lengths"], "target_lengths"),
        )
        if model.training:
            texts = [str(text) for text in inputs["texts"]]
            decoded = self.codec.decode_logits(logits, outputs["out_len"])
            self.latest_train_text_metrics = compute_text_metrics(texts, decoded)
            self.latest_train_loss = float(loss.detach().cpu())
            if self.optimizer is not None:
                self.latest_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: CalamariTorchModel,
        inputs: dict[str, object],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | tuple[Tensor, Tensor] | None, Tensor | None]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        return (
            loss.detach(),
            (outputs["logits"].detach(), outputs["out_len"].detach()),
            _tensor(inputs["labels"], "labels").detach(),
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        normalized = dict(logs)
        if "loss" in normalized:
            normalized["train_loss"] = normalized.pop("loss")
        if isinstance(normalized.get("train_loss"), int | float):
            self.latest_train_loss = float(normalized["train_loss"])
        if isinstance(normalized.get("learning_rate"), int | float):
            self.latest_learning_rate = float(normalized["learning_rate"])
        if "eval_loss" in normalized:
            if self.latest_train_loss is not None:
                normalized.setdefault("train_loss", self.latest_train_loss)
            if self.latest_learning_rate is not None:
                normalized.setdefault("learning_rate", self.latest_learning_rate)
        if self.latest_train_text_metrics is not None:
            normalized.update(
                {
                    f"train_{name}": self.latest_train_text_metrics[name]
                    for name in TRAIN_OCR_METRIC_NAMES
                }
            )
        super().log(normalized, start_time)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate EMA weights while retaining raw weights for checkpoint resumes.

        Combined and multilingual val splits also report per-language metrics.
        A single-language val set is already covered by ``eval_*``, so the
        extra ``eval_<language>_*`` pass is skipped.
        """
        model = self.accelerator.unwrap_model(self.model)
        raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.ema.model.state_dict())
        try:
            metrics = super().evaluate(*args, **kwargs)
            language_metrics = self._language_eval_metrics()
            if language_metrics:
                self.log(language_metrics)
                metrics.update(language_metrics)
            self._print_eval_ctc_steps()
            return metrics
        finally:
            model.load_state_dict(raw_state)

    def _print_eval_ctc_steps(self, n_lines: int = 3, top_k: int = 3) -> None:
        """Print greedy CTC softmax rivals for a few val lines while EMA weights are loaded."""
        dataset = self.eval_dataset
        if dataset is None or len(dataset) == 0:
            return
        model = self.accelerator.unwrap_model(self.model)
        device = next(model.parameters()).device
        report = ctc_step_report(model, dataset, self.codec, device, n_lines=n_lines, top_k=top_k)
        if not report:
            return
        print(report)
        epoch = self.state.epoch if self.state.epoch is not None else 0
        write_ctc_step_report(
            report,
            Path(self.args.output_dir)
            / "eval_ctc_steps"
            / f"epoch-{epoch:g}-step-{self.state.global_step}.txt",
        )

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

    def _language_eval_dataloader(
        self, language: str, dataset: Dataset[dict[str, object]]
    ) -> DataLoader[dict[str, object]]:
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

    def _save_checkpoint(self, model: nn.Module, trial: Any) -> None:
        super()._save_checkpoint(model, trial)
        if not self.is_world_process_zero():
            return
        checkpoint_dir = Path(self.args.output_dir) / (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        torch.save(self.ema.model.state_dict(), checkpoint_dir / _EMA_FILENAME)
        _write_metadata(
            checkpoint_dir / _METADATA_FILENAME,
            self.codec,
            self.line_height,
            self.temperature,
            self.lstm_layers,
        )

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module | None = None
    ) -> None:
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        ema_path = Path(resume_from_checkpoint) / _EMA_FILENAME
        if not ema_path.is_file():
            raise ValueError(f"Calamari Trainer checkpoint is missing EMA state: {ema_path}.")
        self.ema.model.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))

    def load_ema_checkpoint(self, checkpoint: Path) -> None:
        """Load the EMA weights selected by Hugging Face best-checkpoint tracking."""
        self.ema.model.load_state_dict(
            torch.load(checkpoint / _EMA_FILENAME, map_location="cpu", weights_only=True)
        )


def train_calamari(
    data_root: Path,
    output_dir: Path,
    settings: CalamariTrainingSettings,
    *,
    report: Callable[[dict[str, float]], None] | None = None,
) -> tuple[CalamariTorchModel, CharacterCodec, dict[str, float]]:
    """Train or fine-tune Calamari through Hugging Face Trainer and save ``best.pt``."""
    if settings.mode not in {"train", "finetune"}:
        raise ValueError("Calamari training.mode must be 'train' or 'finetune'.")
    if settings.logging_steps <= 0:
        raise ValueError("Calamari logging_steps must be greater than zero.")
    if not 0.0 <= settings.warmup_ratio < 1.0:
        raise ValueError(
            "Calamari warmup_ratio must be greater than or equal to zero and less than one."
        )
    if settings.checkpoint_top_k <= 0:
        raise ValueError("Calamari checkpoint_top_k must be greater than zero.")
    if settings.early_stopping_patience is not None and settings.early_stopping_patience <= 0:
        raise ValueError("Calamari early_stopping_patience must be greater than zero or null.")
    if settings.early_stopping_min_delta < 0.0:
        raise ValueError("Calamari early_stopping_min_delta must be greater than or equal to zero.")

    train_samples = collect_samples(data_root, settings.train_split)
    if not train_samples:
        raise ValueError(f"No Calamari training samples found in {data_root}.")
    validation_samples = collect_samples(data_root, settings.validation_split)
    model, codec = _initial_model([*train_samples, *validation_samples], settings)
    train_lines = CalamariLineDataset(data_root, settings.train_split, codec, settings.line_height)
    if settings.mode == "train":
        train_lines = repeat_syriac_2024_train(train_lines)
    train_dataset = CalamariAugmentedDataset(
        train_lines,
        settings.copies,
        probability=settings.augmentation_probability,
    )
    validation_dataset = CalamariLineDataset(
        data_root, settings.validation_split, codec, settings.line_height
    )
    language_eval_datasets = {
        language: Subset(validation_dataset, indices)
        for language, indices in language_indices(
            [sample.language for sample in validation_dataset.samples]
        ).items()
    }

    # The architecture has lazy LSTM and classifier layers, which must exist
    # before Trainer creates its optimizer or restores a checkpoint.
    _materialize_model(model, collate_ctc([train_dataset[0]]), torch.device("cpu"))
    ema = ExponentialMovingAverage(model, settings.ema_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        max_grad_norm=5.0,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=settings.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=settings.checkpoint_top_k,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        dataloader_num_workers=settings.workers,
        dataloader_persistent_workers=settings.workers > 0,
        dataloader_prefetch_factor=8 if settings.workers > 0 else None,
        dataloader_pin_memory=_resolve_device(settings.device).type == "cuda",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=settings.seed,
        use_cpu=_resolve_device(settings.device).type == "cpu",
        fp16=_resolve_device(settings.device).type == "cuda",
    )
    # Per-epoch evaluation above produces the eval_cer our EarlyStoppingCallback reads.
    callbacks: list[TrainerCallback] = [_EMACallback(ema), _ReportCallback(report)]
    if settings.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                patience=settings.early_stopping_patience,
                min_delta=settings.early_stopping_min_delta,
            )
        )
    trainer = CalamariTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_ctc,
        compute_metrics=lambda prediction: _compute_ctc_metrics(prediction, codec),
        codec=codec,
        ema=ema,
        line_height=settings.line_height,
        temperature=settings.temperature,
        language_eval_datasets=language_eval_datasets,
        callbacks=callbacks,
    )
    resume_checkpoint = _trainer_checkpoint(settings)
    if report is not None:
        length_charts = {"step": 0.0, "epoch": 0.0}
        length_charts.update(
            {
                f"train_{key}": value
                for key, value in compute_reference_length_counts(
                    [sample.text for sample in train_samples]
                ).items()
            }
        )
        length_charts.update(
            {
                f"eval_{key}": value
                for key, value in compute_reference_length_counts(
                    [sample.text for sample in validation_samples]
                ).items()
            }
        )
        report(length_charts)
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    best_checkpoint = (
        Path(trainer.state.best_model_checkpoint)
        if trainer.state.best_model_checkpoint is not None
        else None
    )
    if best_checkpoint is not None:
        trainer.load_ema_checkpoint(best_checkpoint)
    ema.model.eval()
    ema_device = next(ema.model.parameters()).device
    save_calamari_checkpoint(
        output_dir / "best.pt",
        ema.model.cpu(),
        charset=codec.charset,
        line_height=settings.line_height,
        temperature=ema.model.config.temperature,
    )
    ema.model.to(ema_device)
    best_metrics = _best_metrics(trainer.state.log_history)
    return ema.model, codec, best_metrics


@torch.no_grad()
def evaluate_model(
    model: CalamariTorchModel,
    loader: DataLoader[dict[str, object]],
    codec: CharacterCodec,
    device: torch.device,
    loss_function: nn.CTCLoss | None = None,
) -> dict[str, float]:
    """Evaluate a model using CTC-decoded line transcription metrics."""
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    losses: list[float] = []
    for batch in loader:
        loss, decoded, texts = _batch_loss(model, batch, codec, device, loss_function)
        predictions.extend(decoded)
        references.extend(texts)
        if loss_function is not None:
            losses.append(float(loss.cpu()))
    metrics = compute_text_metrics(references, predictions)
    metrics.update(compute_sequence_length_metrics(references, predictions))
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    return metrics


def _initial_model(
    samples: list[object], settings: CalamariTrainingSettings
) -> tuple[CalamariTorchModel, CharacterCodec]:
    resume_checkpoint = _trainer_checkpoint(settings)
    if resume_checkpoint is not None:
        metadata = _read_metadata(resume_checkpoint / _METADATA_FILENAME)
        codec = CharacterCodec(tuple(metadata["charset"]))
        return (
            CalamariTorchModel(
                default_model_config(
                    classes=codec.classes,
                    temperature=float(metadata["temperature"]),
                    lstm_layers=_metadata_lstm_layers(metadata),
                    dropout_rate=settings.dropout_rate,
                    conv0_filters=settings.conv0_filters,
                    conv1_filters=settings.conv1_filters,
                )
            ),
            codec,
        )
    if settings.mode == "finetune":
        if settings.checkpoint is None:
            raise ValueError("Fine-tuning requires training.checkpoint.")
        model, metadata = load_calamari_checkpoint(
            settings.checkpoint, dropout_rate=settings.dropout_rate
        )
        if metadata.line_height != settings.line_height:
            raise ValueError(
                "Fine-tuning line height must match the checkpoint: "
                f"{metadata.line_height}, received {settings.line_height}."
            )
        codec = CharacterCodec(metadata.charset)
        unsupported = sorted(
            {character for sample in samples for character in sample.text} - set(codec.charset)
        )
        if unsupported:
            model, codec = _expand_checkpoint_charset(
                model,
                codec,
                unsupported,
                line_height=settings.line_height,
            )
        return model, codec
    codec = CharacterCodec.from_texts(sample.text for sample in samples)
    return (
        CalamariTorchModel(
            default_model_config(
                classes=codec.classes,
                temperature=settings.temperature,
                lstm_layers=settings.lstm_layers,
                dropout_rate=settings.dropout_rate,
                conv0_filters=settings.conv0_filters,
                conv1_filters=settings.conv1_filters,
            )
        ),
        codec,
    )


def _expand_checkpoint_charset(
    model: CalamariTorchModel,
    codec: CharacterCodec,
    additional_characters: list[str],
    *,
    line_height: int,
) -> tuple[CalamariTorchModel, CharacterCodec]:
    """Extend a pretrained checkpoint's classifier without changing known outputs."""
    expanded_charset = (*codec.charset, *additional_characters)
    lstm_layers = sum(layer.kind == "bilstm" for layer in model.config.layers)
    dropout_rate = next(
        (
            float(layer.rate)
            for layer in model.config.layers
            if layer.kind == "dropout" and layer.rate is not None
        ),
        0.3,
    )
    conv_filters = [
        int(layer.filters)
        for layer in model.config.layers
        if layer.kind == "conv2d" and layer.filters is not None
    ]
    if len(conv_filters) != 2:
        raise ValueError(
            "Cannot expand a Calamari checkpoint without exactly two convolution blocks."
        )
    expanded_model = CalamariTorchModel(
        default_model_config(
            classes=len(expanded_charset),
            temperature=model.config.temperature,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate,
            conv0_filters=conv_filters[0],
            conv1_filters=conv_filters[1],
        )
    )
    expanded_model.eval()
    with torch.no_grad():
        expanded_model(
            torch.zeros((1, 8, line_height, 1), dtype=torch.float32),
            image_lengths=torch.tensor([8]),
        )

    if not isinstance(model.logits, nn.Linear) or not isinstance(expanded_model.logits, nn.Linear):
        raise TypeError("Calamari classifier must be materialized before extending its charset.")
    with torch.no_grad():
        source_state = model.state_dict()
        expanded_state = expanded_model.state_dict()
        for name, value in expanded_state.items():
            if name in {"logits.weight", "logits.bias"}:
                continue
            source_value = source_state.get(name)
            if source_value is None or source_value.shape != value.shape:
                raise ValueError(f"Cannot expand incompatible pretrained parameter: {name}.")
            value.copy_(source_value)

        known_characters = len(codec.charset) - 1
        expanded_model.logits.weight[:known_characters].copy_(
            model.logits.weight[:known_characters]
        )
        expanded_model.logits.bias[:known_characters].copy_(model.logits.bias[:known_characters])
        expanded_model.logits.weight[-1].copy_(model.logits.weight[-1])
        expanded_model.logits.bias[-1].copy_(model.logits.bias[-1])
        expanded_model.logits.weight[known_characters:-1].zero_()
        expanded_model.logits.bias[known_characters:-1].fill_(
            float(model.logits.bias[:known_characters].mean())
        )

    return expanded_model, CharacterCodec(expanded_charset)


def _trainer_checkpoint(settings: CalamariTrainingSettings) -> Path | None:
    checkpoint = settings.checkpoint
    if (
        settings.mode == "train"
        and checkpoint is not None
        and checkpoint.is_dir()
        and (checkpoint / "trainer_state.json").is_file()
    ):
        return checkpoint
    return None


def _compute_ctc_metrics(prediction: Any, codec: CharacterCodec) -> dict[str, float]:
    logits, output_lengths = prediction.predictions
    hypotheses = codec.decode_logits(
        torch.as_tensor(logits),
        torch.as_tensor(output_lengths),
    )
    labels = torch.as_tensor(prediction.label_ids)
    references = [
        "".join(codec.charset[int(token)] for token in row.tolist() if token > 0) for row in labels
    ]
    metrics = compute_text_metrics(references, hypotheses)
    metrics.update(compute_sequence_length_metrics(references, hypotheses))
    return metrics


def _best_metrics(history: list[dict[str, Any]]) -> dict[str, float]:
    evaluations = [
        metrics for metrics in history if isinstance(metrics.get("eval_cer"), int | float)
    ]
    if not evaluations:
        raise RuntimeError("Calamari training did not produce evaluation metrics.")
    best = min(evaluations, key=lambda metrics: float(metrics["eval_cer"]))
    return {
        key: float(value)
        for key, value in best.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _write_metadata(
    path: Path,
    codec: CharacterCodec,
    line_height: int,
    temperature: float,
    lstm_layers: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "charset": codec.charset,
                "line_height": line_height,
                "temperature": temperature,
                "lstm_layers": lstm_layers,
            }
        ),
        encoding="utf-8",
    )


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Calamari Trainer checkpoint is missing metadata: {path}.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid Calamari Trainer checkpoint metadata: {path}.")
    return metadata


def _metadata_lstm_layers(metadata: dict[str, object]) -> int:
    lstm_layers = metadata.get("lstm_layers", 1)
    if not isinstance(lstm_layers, int) or isinstance(lstm_layers, bool) or lstm_layers < 1:
        raise ValueError("Calamari Trainer checkpoint has an invalid LSTM layer count.")
    return lstm_layers


def _loader(
    dataset: Dataset[dict[str, object]], settings: CalamariTrainingSettings, *, shuffle: bool
) -> DataLoader[dict[str, object]]:
    extra: dict[str, object] = {}
    if settings.workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 8
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.workers,
        pin_memory=_resolve_device(settings.device).type == "cuda",
        collate_fn=collate_ctc,
        **extra,
    )


def _materialize_model(
    model: CalamariTorchModel, batch: dict[str, object], device: torch.device
) -> None:
    image = _tensor(batch["image"], "image").to(device, non_blocking=device.type == "cuda")
    lengths = _tensor(batch["image_lengths"], "image_lengths").to(
        device, non_blocking=device.type == "cuda"
    )
    with torch.no_grad():
        model(image, image_lengths=lengths)


def _batch_loss(
    model: CalamariTorchModel,
    batch: dict[str, object],
    codec: CharacterCodec,
    device: torch.device,
    loss_function: nn.CTCLoss | None,
) -> tuple[Tensor, list[str], list[str]]:
    pinned = device.type == "cuda"
    image = _tensor(batch["image"], "image").to(device, non_blocking=pinned)
    image_lengths = _tensor(batch["image_lengths"], "image_lengths").to(device, non_blocking=pinned)
    targets = _tensor(batch["targets"], "targets").to(device, non_blocking=pinned)
    target_lengths = _tensor(batch["target_lengths"], "target_lengths").to(
        device, non_blocking=pinned
    )
    outputs = model(image, image_lengths=image_lengths)
    output_lengths = outputs["out_len"]
    logits = outputs["logits"]
    if loss_function is None:
        loss = torch.zeros((), device=device)
    else:
        loss = loss_function(
            logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths
        )
    texts = [str(value) for value in batch["texts"]]
    return loss, codec.decode_logits(logits, output_lengths), texts


def _tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Calamari batch {name} must be a Tensor.")
    return value


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)
