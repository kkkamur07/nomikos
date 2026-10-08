from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


_LANGUAGES = ("greek", "armenian", "syriac", "coptic")
_SEQUENCE_LENGTH_METRIC = re.compile(
    r"sequence_length_(\d{3})_(\d{3})_(samples|reference_characters|character_errors|cer)$"
)
# Headline metrics per section; anything else for that phase falls through to ``extra/``.
_HEADLINE_METRICS = frozenset({"cer", "wer", "exact_match", "loss"})
_TEST_HEADLINE_METRICS = frozenset({"cer", "wer", "exact_match"})
_SYSTEM_SUFFIXES = ("_runtime", "_samples_per_second", "_steps_per_second")
_SYSTEM_KEYS = frozenset({"total_flos"})
_AXIS_KEYS = frozenset({"epoch", "step"})
_SECTIONS = ("train", "eval", "test", "length", "gradient normalization", "extra", "system")


def _evaluation_scope_and_metric(key: str) -> tuple[str, str] | None:
    """Return the W&B section and flat metric suffix for evaluation keys."""
    if not key.startswith("eval_"):
        return None
    metric = key.removeprefix("eval_")
    for language in _LANGUAGES:
        prefix = f"{language}_"
        if metric.startswith(prefix):
            return language, metric.removeprefix(prefix)
    return "eval", metric


def _sequence_length_scope_and_metric(key: str) -> tuple[str, str] | None:
    """Return the chart scope for train, test or evaluation sequence-length keys."""
    for scope in ("train", "test"):
        prefix = f"{scope}_"
        if key.startswith(prefix):
            metric = key.removeprefix(prefix)
            if _SEQUENCE_LENGTH_METRIC.fullmatch(metric):
                return scope, metric
            return None
    evaluation_metric = _evaluation_scope_and_metric(key)
    if evaluation_metric is None:
        return None
    scope, metric = evaluation_metric
    if _SEQUENCE_LENGTH_METRIC.fullmatch(metric):
        return scope, metric
    return None


def _wandb_metric_key(key: str) -> str:
    """Map a flat trainer metric key into exactly one W&B section.

    Sections: ``train/``, ``eval/``, ``<language>/``, ``test/``, ``length/``,
    ``gradient normalization/``, ``system/`` and ``extra/`` (catch-all, keeps the
    original key). ``epoch`` and ``step`` stay top-level as x-axes.
    """
    if key in _AXIS_KEYS:
        return key
    if _sequence_length_scope_and_metric(key) is not None:
        # Bin scalars are folded into the ``length/`` charts and not logged as scalars.
        return f"length/{key}"
    if key.endswith("grad_norm"):
        return f"gradient normalization/{key}"
    if key in _SYSTEM_KEYS or key.endswith(_SYSTEM_SUFFIXES):
        return f"system/{key}"
    if key.startswith("train_"):
        metric = key.removeprefix("train_")
        if metric in _HEADLINE_METRICS:
            return f"train/{metric}"
    elif key.startswith("test_"):
        metric = key.removeprefix("test_")
        if metric in _TEST_HEADLINE_METRICS:
            return f"test/{metric}"
    else:
        evaluation_metric = _evaluation_scope_and_metric(key)
        if evaluation_metric is not None:
            scope, metric = evaluation_metric
            if metric in _HEADLINE_METRICS:
                return f"{scope}/{metric}"
    return f"extra/{key}"


def _sequence_length_rows(
    metrics: Mapping[str, float],
) -> tuple[dict[str, list[tuple[object, ...]]], set[str]]:
    """Collect scalar sequence-length metrics into per-split bar-chart rows."""
    values: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    sequence_keys: set[str] = set()
    for key, value in metrics.items():
        scoped = _sequence_length_scope_and_metric(str(key))
        if scoped is None:
            continue
        scope, metric = scoped
        match = _SEQUENCE_LENGTH_METRIC.fullmatch(metric)
        if match is None:
            continue
        lower, upper, name = match.groups()
        values.setdefault(scope, {}).setdefault((int(lower), int(upper)), {})[name] = float(value)
        sequence_keys.add(str(key))

    rows_by_scope: dict[str, list[tuple[object, ...]]] = {}
    for scope, bins in values.items():
        rows_by_scope[scope] = [
            (
                f"{lower:03d}-{upper:03d}",
                bin_metrics.get("cer"),
                bin_metrics.get("character_errors", 0.0),
                bin_metrics.get("reference_characters", 0.0),
                bin_metrics.get("samples", 0.0),
            )
            for (lower, upper), bin_metrics in sorted(bins.items())
        ]
    return rows_by_scope, sequence_keys


def _cer_bar_chart(rows: list[tuple[object, ...]], title: str) -> Any:
    """CER by length, with sample count (and errors) on hover."""
    import wandb

    cer_rows = [row for row in rows if row[1] is not None]
    if not cer_rows:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        table = wandb.Table(
            columns=["Reference character length", "CER", "Samples"],
            data=[
                (f"{row[0]} (n={int(row[4])})", float(row[1]), float(row[4])) for row in cer_rows
            ],
        )
        return wandb.plot.bar(
            table,
            "Reference character length",
            "CER",
            title=title,
        )

    figure = go.Figure(
        go.Bar(
            x=[row[0] for row in cer_rows],
            y=[float(row[1]) for row in cer_rows],
            customdata=[[row[4], row[2], row[3]] for row in cer_rows],
            hovertemplate=(
                "Length: %{x}<br>"
                "CER: %{y:.4f}<br>"
                "Samples: %{customdata[0]:.0f}<br>"
                "Character errors: %{customdata[1]:.0f}<br>"
                "Reference characters: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Reference character length",
        yaxis_title="CER",
        bargap=0.2,
    )
    return wandb.Plotly(figure)


def _reference_length_histogram(rows: list[tuple[object, ...]], title: str) -> Any:
    """Sample counts by reference character length."""
    import wandb

    try:
        import plotly.graph_objects as go
    except ImportError:
        table = wandb.Table(
            columns=["Reference character length", "Samples"],
            data=[(row[0], float(row[4])) for row in rows],
        )
        return wandb.plot.bar(
            table,
            "Reference character length",
            "Samples",
            title=title,
        )

    figure = go.Figure(
        go.Bar(
            x=[row[0] for row in rows],
            y=[float(row[4]) for row in rows],
            hovertemplate="Length: %{x}<br>Samples: %{y:.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Reference character length",
        yaxis_title="Samples",
        bargap=0.2,
    )
    return wandb.Plotly(figure)


class WandbLogger:
    """Optional model-agnostic W&B run lifecycle and metric transport."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        name: str | None,
        mode: str,
        save_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self._run = None
        self._sequence_length_rows: dict[str, list[tuple[object, ...]]] = {}
        if not enabled:
            return

        import wandb

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=str(save_dir),
            config=config,
            reinit=True,
        )
        # Keep trainer step as W&B `_step`, and register epoch so charts can
        # use either axis (Step dropdown, or epoch).
        self._run.define_metric("epoch")
        self._run.define_metric("step")
        for section in (*_SECTIONS, *_LANGUAGES):
            self._run.define_metric(f"{section}/*", step_metric="epoch")

    @property
    def run(self) -> Any | None:
        """Expose run metadata needed by the training entry point."""
        return self._run

    def update_config(self, config: Mapping[str, Any]) -> None:
        """Record configuration values resolved after run initialization."""
        if self._run is not None:
            self._run.config.update(dict(config), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        """Log model metrics under W&B's training and evaluation chart sections."""
        if not self._run:
            return

        sequence_rows, sequence_keys = _sequence_length_rows(metrics)
        self._sequence_length_rows.update(sequence_rows)
        payload = {
            _wandb_metric_key(str(key)): float(value)
            for key, value in metrics.items()
            if str(key) not in sequence_keys
        }
        for scope, rows in self._sequence_length_rows.items():
            cer_chart = _cer_bar_chart(rows, f"{scope.title()} CER by reference character length")
            if cer_chart is not None:
                payload[f"length/{scope}_cer"] = cer_chart
            payload[f"length/{scope}_reference_length"] = _reference_length_histogram(
                rows, f"{scope.title()} reference character length"
            )
        if not payload:
            return
        self._run.log(payload, step=step)

    def log_test_metrics(self, metrics: Mapping[str, float], *, step: int, epoch: float) -> None:
        """Log held-out test metrics as one extra row; nothing is written to ``run.summary``."""
        if not self._run:
            return
        payload = {f"test_{key}": float(value) for key, value in metrics.items()}
        payload["step"] = float(step)
        payload["epoch"] = float(epoch)
        # W&B drops rows whose step is below the run's current step, and the best
        # checkpoint usually predates the last logged step. Write a fresh row after
        # the last one (so the final eval row keeps its own epoch); ``test/*``
        # charts use ``epoch`` as their x-axis, and ``step`` keeps the best step.
        next_step = int(getattr(self._run, "step", 0) or 0) + 1
        self.log_metrics(payload, step=max(int(step), next_step))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
