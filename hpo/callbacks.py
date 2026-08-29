from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from modules.callbacks import EpochTiming, ImmediateCSVLogger

from .runtime import optuna


class DeadlineReached(RuntimeError):
    pass


class TrialControl(tf.keras.callbacks.Callback):
    def __init__(
        self,
        trial: Any | None,
        step_offset: int,
        deadline: float,
        allow_pruning: bool,
        minimum_epoch_before_pruning: int = 4,
    ):
        super().__init__()
        self.trial = trial
        self.step_offset = step_offset
        self.deadline = deadline
        self.allow_pruning = allow_pruning
        self.minimum_epoch_before_pruning = minimum_epoch_before_pruning

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if time.time() >= self.deadline:
            raise DeadlineReached("Time budget reached")
        if self.trial is None:
            return

        values = logs or {}
        value = values.get("val_pr_auc")
        if value is None or not np.isfinite(float(value)):
            return

        step = self.step_offset + epoch
        self.trial.report(float(value), step=step)

        if (
            self.allow_pruning
            and epoch + 1 >= self.minimum_epoch_before_pruning
            and self.trial.should_prune()
        ):
            raise optuna.TrialPruned(
                f"Trial pruned at step {step}"
            )


def training_callbacks(
    stage_dir: Path,
    stage_name: str,
    patience: int,
    deadline: float,
    trial: Any | None,
    step_offset: int,
    allow_pruning: bool,
) -> list[tf.keras.callbacks.Callback]:
    stage_dir.mkdir(parents=True, exist_ok=True)

    return [
        EpochTiming(stage_name),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=max(2, patience // 2),
            min_lr=1e-7,
        ),
        ImmediateCSVLogger(stage_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
        TrialControl(
            trial=trial,
            step_offset=step_offset,
            deadline=deadline,
            allow_pruning=allow_pruning,
        ),
    ]


def best_history_epoch(history: tf.keras.callbacks.History) -> tuple[int, float]:
    values = np.asarray(history.history.get("val_pr_auc", []), dtype=float)
    if not len(values) or not np.isfinite(values).any():
        return 0, float("nan")

    index = int(np.nanargmax(values))

    return index + 1, float(values[index])
