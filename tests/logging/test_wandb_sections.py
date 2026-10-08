"""W&B section layout: flat trainer keys -> train/eval/test/length/... sections."""

from __future__ import annotations

from typing import Any

import pytest

from src.logging import wandb as wandb_logging
from src.logging.wandb import (
    WandbLogger,
    _sequence_length_scope_and_metric,
    _wandb_metric_key,
)

# Realistic rows, mirroring var/logs/**/metrics.jsonl (train row, eval row, test row).
_TRAIN_ROW = {
    "epoch": 1.0,
    "step": 100,
    "train_cer": 0.2,
    "train_wer": 0.5,
    "train_exact_match": 0.1,
    "train_loss": 3.0,
    "train_runtime": 12.0,
    "train_samples_per_second": 50.0,
    "train_steps_per_second": 2.0,
    "train_sequence_length_002_009_samples": 7.0,
    "total_flos": 1e12,
    "grad_norm": 1.5,
    "learning_rate": 1e-4,
}
_EVAL_ROW = {
    "epoch": 1.0,
    "step": 100,
    "eval_cer": 0.2,
    "eval_wer": 0.5,
    "eval_exact_match": 0.1,
    "eval_loss": 3.0,
    "eval_runtime": 4.0,
    "eval_samples_per_second": 30.0,
    "eval_steps_per_second": 1.0,
    "eval_sroie_f1": 0.4,
    "eval_sroie_precision": 0.4,
    "eval_sroie_recall": 0.4,
    "eval_sequence_length_002_009_cer": 0.1,
    "eval_sequence_length_002_009_character_errors": 3.0,
    "eval_sequence_length_002_009_reference_characters": 30.0,
    "eval_sequence_length_002_009_samples": 5.0,
}
_TEST_ROW = {
    "epoch": 3.0,
    "step": 25,
    "test_cer": 0.1,
    "test_wer": 0.3,
    "test_exact_match": 0.5,
    "test_loss": 2.0,
    "test_sroie_f1": 0.4,
    "test_sroie_precision": 0.4,
    "test_sroie_recall": 0.4,
    "test_sequence_length_002_009_cer": 0.2,
    "test_sequence_length_002_009_character_errors": 3.0,
    "test_sequence_length_002_009_reference_characters": 15.0,
    "test_sequence_length_002_009_samples": 7.0,
}

_EXPECTED = {
    "epoch": "epoch",
    "step": "step",
    "train_cer": "train/cer",
    "train_wer": "train/wer",
    "train_exact_match": "train/exact_match",
    "train_loss": "train/loss",
    "eval_cer": "eval/cer",
    "eval_wer": "eval/wer",
    "eval_exact_match": "eval/exact_match",
    "eval_loss": "eval/loss",
    "test_cer": "test/cer",
    "test_wer": "test/wer",
    "test_exact_match": "test/exact_match",
    "test_loss": "extra/test_loss",
    "grad_norm": "gradient normalization/grad_norm",
    "learning_rate": "extra/learning_rate",
    "eval_sroie_f1": "extra/eval_sroie_f1",
    "eval_sroie_precision": "extra/eval_sroie_precision",
    "eval_sroie_recall": "extra/eval_sroie_recall",
    "test_sroie_f1": "extra/test_sroie_f1",
    "test_sroie_precision": "extra/test_sroie_precision",
    "test_sroie_recall": "extra/test_sroie_recall",
    "train_runtime": "system/train_runtime",
    "train_samples_per_second": "system/train_samples_per_second",
    "train_steps_per_second": "system/train_steps_per_second",
    "eval_runtime": "system/eval_runtime",
    "eval_samples_per_second": "system/eval_samples_per_second",
    "eval_steps_per_second": "system/eval_steps_per_second",
    "total_flos": "system/total_flos",
}


@pytest.mark.parametrize("row", [_TRAIN_ROW, _EVAL_ROW, _TEST_ROW], ids=["train", "eval", "test"])
def test_every_realistic_key_lands_in_one_section(row: dict[str, float]) -> None:
    allowed = {"train", "eval", "test", "length", "gradient normalization", "extra", "system"}
    for key in row:
        mapped = _wandb_metric_key(key)
        if key in {"epoch", "step"}:
            assert mapped == key
            continue
        section = mapped.split("/", 1)[0]
        assert section in allowed, (key, mapped)
        if "sequence_length_" in key:
            assert mapped == f"length/{key}"
        else:
            assert mapped == _EXPECTED[key]


def test_language_and_misc_keys() -> None:
    for language in ("greek", "armenian", "syriac", "coptic"):
        assert _wandb_metric_key(f"eval_{language}_cer") == f"{language}/cer"
        assert _wandb_metric_key(f"eval_{language}_loss") == f"{language}/loss"
        assert _wandb_metric_key(f"eval_{language}_sroie_f1") == f"extra/eval_{language}_sroie_f1"
        assert _sequence_length_scope_and_metric(
            f"eval_{language}_sequence_length_002_009_cer"
        ) == (language, "sequence_length_002_009_cer")
    assert _wandb_metric_key("train_encoder_grad_norm") == (
        "gradient normalization/train_encoder_grad_norm"
    )
    assert _wandb_metric_key("something_new") == "extra/something_new"
    assert _wandb_metric_key("train_something_new") == "extra/train_something_new"


class _FakeRun:
    def __init__(self) -> None:
        self.step = 40
        self.summary: dict[str, float] = {}
        self.logged: list[tuple[dict[str, Any], int]] = []

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.logged.append((payload, step))


def _logger(monkeypatch) -> tuple[WandbLogger, _FakeRun]:
    monkeypatch.setattr(wandb_logging, "_cer_bar_chart", lambda rows, title: "bar")
    monkeypatch.setattr(wandb_logging, "_reference_length_histogram", lambda rows, title: "hist")
    logger = WandbLogger.__new__(WandbLogger)
    run = _FakeRun()
    logger._run = run
    logger._sequence_length_rows = {}
    return logger, run


def test_log_metrics_length_charts_and_no_bin_scalars(monkeypatch) -> None:
    logger, run = _logger(monkeypatch)
    logger.log_metrics({**_TRAIN_ROW, **_EVAL_ROW}, step=100)
    payload, step = run.logged[-1]
    assert step == 100
    assert payload["length/eval_cer"] == "bar"
    assert payload["length/eval_reference_length"] == "hist"
    assert payload["length/train_cer"] == "bar"  # monkeypatched; real chart skips cer-less bins
    assert payload["length/train_reference_length"] == "hist"
    assert not any("sequence_length" in key for key in payload)
    assert not any(key.startswith("charts/") for key in payload)
    assert payload["gradient normalization/grad_norm"] == 1.5
    assert payload["system/total_flos"] == 1e12


def test_log_test_metrics_row_without_summary(monkeypatch) -> None:
    logger, run = _logger(monkeypatch)
    metrics = {key.removeprefix("test_"): value for key, value in _TEST_ROW.items()}
    metrics.pop("epoch")
    metrics.pop("step")

    logger.log_test_metrics(metrics, step=25, epoch=3.0)

    assert run.summary == {}
    payload, step = run.logged[-1]
    assert step == 41  # best step predates the run's current step; W&B needs monotonic steps
    assert payload["test/cer"] == 0.1
    assert payload["test/wer"] == 0.3
    assert payload["test/exact_match"] == 0.5
    assert "test/loss" not in payload
    assert payload["extra/test_loss"] == 2.0
    assert payload["extra/test_sroie_f1"] == 0.4
    assert payload["step"] == 25.0
    assert payload["epoch"] == 3.0
    assert payload["length/test_cer"] == "bar"
    assert payload["length/test_reference_length"] == "hist"
    assert not any("sequence_length" in key for key in payload)


def test_log_test_metrics_noop_when_disabled() -> None:
    logger = WandbLogger.__new__(WandbLogger)
    logger._run = None
    logger._sequence_length_rows = {}
    logger.log_test_metrics({"cer": 0.1}, step=1, epoch=1.0)
    logger.log_metrics({"train_cer": 0.1}, step=1)
