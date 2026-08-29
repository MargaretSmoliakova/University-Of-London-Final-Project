from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from modules.config import Config
from modules.metrics import compute_metrics
from modules.models import balanced_class_weights
from modules.training import validate_fold_integrity

from .callbacks import best_history_epoch, training_callbacks
from .config import HPOConfig, TransferSearchSpec
from .data import make_fold_datasets
from .models import build_transfer_model, enable_fine_tuning


def predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    return np.asarray(model.predict(dataset, verbose=0), dtype=float).ravel()


@dataclass(frozen=True)
class StageResult:
    epoch: int
    pr_auc: float
    probabilities: np.ndarray
    metrics: dict[str, float | int]
    seconds: float


def _fit_training_stage(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    validation: pd.DataFrame,
    class_weights: dict[int, float],
    stage_dir: Path,
    stage_name: str,
    epochs: int,
    patience: int,
    deadline: float,
    trial: Any | None,
    step_offset: int,
    allow_pruning: bool,
) -> StageResult:
    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=training_callbacks(
            stage_dir,
            stage_name,
            patience,
            deadline,
            trial,
            step_offset=step_offset,
            allow_pruning=allow_pruning,
        ),
        verbose=2,
    )
    seconds = time.perf_counter() - started

    epoch, pr_auc = best_history_epoch(history)
    probabilities = predict_probabilities(model, validation_dataset)
    metrics = compute_metrics(validation["label"], probabilities, threshold=0.5)

    return StageResult(epoch, pr_auc, probabilities, metrics, seconds)


def _fold_result(
    fold_number: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    class_weights: dict[int, float],
    head: StageResult,
    fine: StageResult,
) -> tuple[dict[str, Any], np.ndarray]:
    if float(fine.metrics["average_precision"]) >= float(
        head.metrics["average_precision"]
    ):
        selected_stage = "fine_tuning"
        selected = fine
    else:
        selected_stage = "head_training"
        selected = head

    row: dict[str, Any] = {
        "fold": fold_number,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_patients": int(train["patient_id"].nunique()),
        "validation_patients": int(validation["patient_id"].nunique()),
        "selected_stage": selected_stage,
        "head_selected_epoch": head.epoch,
        "fine_tune_selected_epoch": fine.epoch,
        "head_keras_val_pr_auc": head.pr_auc,
        "fine_tune_keras_val_pr_auc": fine.pr_auc,
        "head_exact_average_precision": head.metrics["average_precision"],
        "fine_tune_exact_average_precision": fine.metrics["average_precision"],
        "head_seconds": float(head.seconds),
        "fine_tune_seconds": float(fine.seconds),
        "fold_seconds": float(head.seconds + fine.seconds),
        "class_weight_0": float(class_weights[0]),
        "class_weight_1": float(class_weights[1]),
        **selected.metrics,
    }

    return row, selected.probabilities


def train_transfer_fold(
    spec: TransferSearchSpec,
    development: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    experiment_config: Config,
    hpo_config: HPOConfig,
    fold_dir: Path,
    fold_number: int,
    total_folds: int,
    head_epochs: int,
    finetune_epochs: int,
    patience: int,
    deadline: float,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    trial: Any | None,
    allow_pruning: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    tf.keras.backend.clear_session()
    trial_number = int(trial.number) if trial is not None else 10_000
    training_seed = hpo_config.seed + trial_number * 100 + fold_number
    tf.keras.utils.set_random_seed(training_seed)

    train = development.iloc[train_indices].reset_index(drop=True)
    validation = development.iloc[validation_indices].reset_index(drop=True)
    validate_fold_integrity(train, validation, fold_number)
    train_dataset, validation_dataset = make_fold_datasets(
        development,
        train_indices,
        validation_indices,
        experiment_config,
        preloaded,
        training_seed,
    )

    class_weights = balanced_class_weights(train)
    model = build_transfer_model(hpo_config, spec)
    prefix = (
        f"trial {trial_number} fold {fold_number}/{total_folds}"
        if trial is not None
        else f"confirmation fold {fold_number}/{total_folds}"
    )
    stage_options = {
        "model": model,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "validation": validation,
        "class_weights": class_weights,
        "patience": patience,
        "deadline": deadline,
        "trial": trial,
        "allow_pruning": allow_pruning,
    }

    head = _fit_training_stage(
        **stage_options,
        stage_dir=fold_dir / "head_training",
        stage_name=f"{prefix} head training",
        epochs=head_epochs,
        step_offset=(fold_number - 1) * (head_epochs + finetune_epochs),
    )

    enable_fine_tuning(model, spec)
    fine = _fit_training_stage(
        **stage_options,
        stage_dir=fold_dir / "fine_tuning",
        stage_name=f"{prefix} fine-tuning",
        epochs=finetune_epochs,
        step_offset=(fold_number - 1) * (head_epochs + finetune_epochs) + head_epochs,
    )

    row, selected_probabilities = _fold_result(
        fold_number, train, validation, class_weights, head, fine
    )

    del stage_options, model, train_dataset, validation_dataset
    tf.keras.backend.clear_session()
    gc.collect()

    return row, selected_probabilities
