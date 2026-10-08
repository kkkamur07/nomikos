"""Early stopping on a lower-is-better evaluation metric (eval CER)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

LOGGER = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    """Track a lower-is-better metric and decide when training should stop.

    ``update`` is called once per evaluation. Evaluation happens once per epoch,
    so ``step`` is the epoch number and ``patience`` counts epochs.

    ``min_delta`` is an absolute difference in CER expressed as a fraction in
    ``[0, 1]``; ``0.00001`` therefore means ``0.001 %`` CER. A value counts as an
    improvement only when it beats the best value by strictly more than
    ``min_delta`` (``best - value > min_delta``), which matches the semantics of
    ``transformers.EarlyStoppingCallback`` with ``early_stopping_threshold``.
    Training stops once ``patience`` consecutive evaluations fail to improve.
    """

    patience: int
    min_delta: float = 0.0
    best: float | None = None
    best_step: int | None = None
    wait: int = 0
    stopped_step: int | None = None

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"Early stopping patience must be at least one, got {self.patience}.")
        if self.min_delta < 0.0:
            raise ValueError(f"Early stopping min_delta must not be negative, got {self.min_delta}.")

    def update(self, value: float, step: int) -> bool:
        """Record ``value`` observed at ``step`` and return whether to stop."""
        if self.best is None or self.best - value > self.min_delta:
            self.best = value
            self.best_step = step
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_step = step
            return True
        return False

    @property
    def should_stop(self) -> bool:
        """Return whether patience has been exhausted."""
        return self.stopped_step is not None

    def summary(self) -> str:
        """Return a one-line description of the tracker state."""
        best = "none" if self.best is None else f"{self.best:.6f}"
        return f"best={best} best_step={self.best_step} wait={self.wait}/{self.patience}"


class EarlyStoppingCallback(TrainerCallback):
    """Stop a Hugging Face ``Trainer`` when ``metric`` stops improving.

    Reads the metric directly from the evaluation ``metrics`` dict, so it does not
    depend on ``metric_for_best_model`` or ``load_best_model_at_end``. The tracker
    is exposed as ``tracker`` so callers can read ``best``/``best_step`` after
    training.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, metric: str = "eval_cer") -> None:
        self.metric = metric
        self.tracker = EarlyStopping(patience=patience, min_delta=min_delta)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update the tracker with the latest metric and request a stop if needed."""
        metrics = metrics or {}
        if self.metric not in metrics:
            raise ValueError(
                f"Early stopping metric {self.metric!r} missing from evaluation metrics; "
                f"available keys: {sorted(metrics)}."
            )
        value = float(metrics[self.metric])
        step = int(round(state.epoch)) if state.epoch is not None else state.global_step
        stop = self.tracker.update(value, step)
        LOGGER.debug("Early stopping on %s at epoch %d: %s", self.metric, step, self.tracker.summary())
        if stop:
            control.should_training_stop = True
            LOGGER.info(
                "Early stopping at epoch %d: %s did not improve by more than %g for %d epochs "
                "(best %g at epoch %s).",
                step,
                self.metric,
                self.tracker.min_delta,
                self.tracker.patience,
                self.tracker.best,
                self.tracker.best_step,
            )
