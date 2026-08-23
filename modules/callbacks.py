from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .common import LOGGER
from .config import Config


class EpochTiming(tf.keras.callbacks.Callback):
    def __init__(self, stage_name: str):
        super().__init__()
        self.stage_name = stage_name
        self.started = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.perf_counter() - self.started
        values = logs or {}
        metrics = ", ".join(
            f"{name}={float(value):.6f}"
            for name, value in sorted(values.items())
            if np.isscalar(value)
        )
        LOGGER.info(
            "%s epoch %d finished in %.1f seconds%s",
            self.stage_name,
            epoch + 1,
            elapsed,
            f" | {metrics}" if metrics else "",
        )
        flush_python_log_handlers()


def flush_python_log_handlers() -> None:
    seen: set[int] = set()
    for logger in (logging.getLogger(), LOGGER):
        for handler in logger.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            handler.flush()
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.FileHandler) and stream is not None:
                try:
                    os.fsync(stream.fileno())
                except (AttributeError, OSError):
                    pass


class ImmediateCSVLogger(tf.keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        super().on_epoch_end(epoch, logs)
        csv_file = getattr(self, "csv_file", None)
        if csv_file is not None:
            csv_file.flush()
            try:
                os.fsync(csv_file.fileno())
            except (AttributeError, OSError):
                pass


def callbacks_for_stage(
    config: Config, stage_dir: Path, stage_name: str
) -> list[tf.keras.callbacks.Callback]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        EpochTiming(stage_name),
        tf.keras.callbacks.ModelCheckpoint(
            stage_dir / "best.weights.h5",
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=max(2, config.early_stopping_patience // 2),
            min_lr=1e-7,
        ),
        ImmediateCSVLogger(stage_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def callbacks_for_final_stage(
    stage_dir: Path, stage_name: str
) -> list[tf.keras.callbacks.Callback]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        EpochTiming(stage_name),
        ImmediateCSVLogger(stage_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def best_epoch_and_value(history: tf.keras.callbacks.History) -> tuple[int, float]:
    values = np.asarray(history.history["val_pr_auc"], dtype=float)
    best_index = int(np.nanargmax(values))
    return best_index + 1, float(values[best_index])


def report_stage_quality(
    history: tf.keras.callbacks.History, stage_name: str, validation_metadata
) -> None:
    best_epoch, best_pr_auc = best_epoch_and_value(history)
    roc_values = np.asarray(history.history["val_roc_auc"], dtype=float)
    best_roc_auc = float(np.nanmax(roc_values))
    prevalence = float(validation_metadata["label"].mean())
    LOGGER.info(
        "%s summary: best_epoch=%d, best_val_pr_auc=%.4f, "
        "best_val_roc_auc=%.4f, malignant_prevalence=%.4f",
        stage_name,
        best_epoch,
        best_pr_auc,
        best_roc_auc,
        prevalence,
    )
    if np.isfinite(best_roc_auc) and best_roc_auc < 0.55:
        LOGGER.warning(
            "%s: low validation ROC-AUC %.4f",
            stage_name,
            best_roc_auc,
        )
