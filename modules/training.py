from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from .callbacks import (
    best_epoch_and_value,
    callbacks_for_final_stage,
    callbacks_for_stage,
    report_stage_quality,
)
from .common import LOGGER, write_json
from .config import Config
from .data_pipeline import make_dataset
from .metadata import split_summary
from .metrics import compute_metrics
from .models import (
    PRIMARY_REGULARIZATION,
    RegularizationSpec,
    balanced_class_weights,
    build_scratch_cnn,
    build_transfer_model,
    enable_transfer_fine_tuning,
)


def stratification_labels(metadata: pd.DataFrame) -> np.ndarray:
    return (
        metadata["lesion_type"].astype(str)
        + "_label_"
        + metadata["label"].astype(str)
    ).to_numpy()


def make_cv_splitter(config: Config) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )


def validate_fold_integrity(
    train_fold: pd.DataFrame, validation_fold: pd.DataFrame, fold_number: int
) -> None:
    overlap = set(train_fold["patient_id"]) & set(validation_fold["patient_id"])
    if overlap:
        raise RuntimeError(f"Patient overlap in fold {fold_number}")
    if train_fold["label"].nunique() != 2 or validation_fold["label"].nunique() != 2:
        raise RuntimeError(f"Missing class in fold {fold_number}")


def make_development_calibration_split(
    train_metadata: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approximate_splits = max(3, int(round(1.0 / config.calibration_fraction)))
    splitter = StratifiedGroupKFold(
        n_splits=approximate_splits,
        shuffle=True,
        random_state=config.seed + 101,
    )
    candidates = list(
        splitter.split(
            train_metadata,
            stratification_labels(train_metadata),
            groups=train_metadata["patient_id"],
        )
    )
    target_rows = config.calibration_fraction * len(train_metadata)
    development_index, calibration_index = min(
        candidates, key=lambda pair: abs(len(pair[1]) - target_rows)
    )
    development = train_metadata.iloc[development_index].reset_index(drop=True)
    calibration = train_metadata.iloc[calibration_index].reset_index(drop=True)
    validate_fold_integrity(development, calibration, fold_number=0)
    assignment = train_metadata[
        ["patient_id", "case_id", "lesion_type", "label", "image_path"]
    ].copy()
    assignment["internal_role"] = "development"
    assignment.loc[calibration_index, "internal_role"] = "calibration"
    assignment.to_csv(config.output_dir / "internal_split_assignments.csv", index=False)
    write_json(
        config.output_dir / "internal_split_summary.json",
        {
            "development": split_summary(development),
            "calibration": split_summary(calibration),
            "requested_calibration_fraction": config.calibration_fraction,
            "actual_calibration_fraction": len(calibration) / len(train_metadata),
        },
    )
    return development, calibration


def train_one_fold(
    variant: str,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    config: Config,
    fold_dir: Path,
    fold_number: int,
    regularization_spec: RegularizationSpec = PRIMARY_REGULARIZATION,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed + fold_number)
    train_dataset = make_dataset(
        train_metadata, config, training=True, seed_offset=fold_number
    )
    validation_dataset = make_dataset(validation_metadata, config, training=False)
    class_weights = balanced_class_weights(train_metadata)

    if variant in {"baseline", "regularized"}:
        regularized = variant == "regularized"
        maximum = (
            config.max_epochs_regularized if regularized else config.max_epochs_baseline
        )
        stage_name = f"{variant} fold {fold_number}/{config.folds} training"
        LOGGER.info("Starting %s (maximum %d epochs)", stage_name, maximum)
        model = build_scratch_cnn(
            config, regularized=regularized, specification=regularization_spec
        )
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=maximum,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fold_dir / "training", stage_name),
            verbose=2,
        )
        report_stage_quality(history, stage_name, validation_metadata)
        selected_epoch, selected_pr_auc = best_epoch_and_value(history)
        schedule: dict[str, Any] = {
            "selected_epoch": selected_epoch,
            "selected_stage": "single_stage",
            "validation_pr_auc": selected_pr_auc,
        }
    elif variant == "transfer":
        model = build_transfer_model(config)
        head_dir = fold_dir / "head_training"
        head_name = f"transfer fold {fold_number}/{config.folds} head training"
        LOGGER.info(
            "Starting %s (maximum %d epochs)",
            head_name,
            config.max_epochs_transfer_head,
        )
        head_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_transfer_head,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, head_dir, head_name),
            verbose=2,
        )
        report_stage_quality(head_history, head_name, validation_metadata)
        head_epoch, head_pr_auc = best_epoch_and_value(head_history)
        model.load_weights(head_dir / "best.weights.h5")

        enable_transfer_fine_tuning(model, config.fine_tune_last_layers)
        fine_dir = fold_dir / "fine_tuning"
        fine_name = f"transfer fold {fold_number}/{config.folds} fine-tuning"
        LOGGER.info(
            "Starting %s (maximum %d epochs)",
            fine_name,
            config.max_epochs_transfer_finetune,
        )
        fine_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_transfer_finetune,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fine_dir, fine_name),
            verbose=2,
        )
        report_stage_quality(fine_history, fine_name, validation_metadata)
        fine_epoch, fine_pr_auc = best_epoch_and_value(fine_history)
        if fine_pr_auc >= head_pr_auc:
            model.load_weights(fine_dir / "best.weights.h5")
            selected_stage, selected_pr_auc = "fine_tuning", fine_pr_auc
        else:
            model.load_weights(head_dir / "best.weights.h5")
            selected_stage, selected_pr_auc = "head_training", head_pr_auc
        schedule = {
            "head_epoch": head_epoch,
            "fine_tune_epoch": fine_epoch,
            "selected_stage": selected_stage,
            "validation_pr_auc": selected_pr_auc,
        }
    else:
        raise ValueError(f"Unknown model: {variant}")
    schedule["class_weights"] = class_weights
    return model, schedule


def cross_validate_variant(
    variant: str,
    train_metadata: pd.DataFrame,
    config: Config,
    regularization_spec: RegularizationSpec = PRIMARY_REGULARIZATION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | bool]]:
    variant_dir = config.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    oof_probability = np.full(len(train_metadata), np.nan, dtype=float)
    oof_fold = np.zeros(len(train_metadata), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    splitter = make_cv_splitter(config)
    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(
            train_metadata,
            stratification_labels(train_metadata),
            groups=train_metadata["patient_id"],
        ),
        start=1,
    ):
        fold_train = train_metadata.iloc[train_index].reset_index(drop=True)
        fold_validation = train_metadata.iloc[validation_index].reset_index(drop=True)
        validate_fold_integrity(fold_train, fold_validation, fold_number)
        LOGGER.info(
            "%s fold %d/%d: train=%d, validation=%d",
            variant,
            fold_number,
            config.folds,
            len(fold_train),
            len(fold_validation),
        )
        model, schedule = train_one_fold(
            variant,
            fold_train,
            fold_validation,
            config,
            variant_dir / f"fold_{fold_number}",
            fold_number,
            regularization_spec,
        )
        probabilities = model.predict(
            make_dataset(fold_validation, config, training=False), verbose=0
        ).ravel()
        oof_probability[validation_index] = probabilities
        oof_fold[validation_index] = fold_number
        fold_rows.append(
            {
                "model": variant,
                "fold": fold_number,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "train_patients": fold_train["patient_id"].nunique(),
                "validation_patients": fold_validation["patient_id"].nunique(),
                **schedule,
                **compute_metrics(fold_validation["label"], probabilities, 0.5),
            }
        )
        schedules.append(schedule)
        del model
        tf.keras.backend.clear_session()

    if np.isnan(oof_probability).any():
        raise RuntimeError(f"Missing OOF predictions: {variant}")
    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(variant_dir / "cv_fold_metrics.csv", index=False)
    oof = train_metadata[
        ["patient_id", "lesion_type", "label", "image_path", "pathology_original"]
    ].copy()
    oof["model"] = variant
    oof["fold"] = oof_fold
    oof["probability"] = oof_probability
    oof.to_csv(variant_dir / "oof_predictions.csv", index=False)

    if variant == "transfer":
        use_fine = sum(x["selected_stage"] == "fine_tuning" for x in schedules) >= math.ceil(
            len(schedules) / 2
        )
        final_schedule: dict[str, int | bool] = {
            "head_epochs": max(1, int(np.median([x["head_epoch"] for x in schedules]))),
            "fine_tune_epochs": max(
                1, int(np.median([x["fine_tune_epoch"] for x in schedules]))
            ),
            "use_fine_tuning": use_fine,
        }
    else:
        final_schedule = {
            "epochs": max(1, int(np.median([x["selected_epoch"] for x in schedules])))
        }
    write_json(variant_dir / "final_training_schedule.json", final_schedule)
    return folds_frame, oof, final_schedule


def train_final_model(
    variant: str,
    train_metadata: pd.DataFrame,
    schedule: dict[str, int | bool],
    config: Config,
    regularization_spec: RegularizationSpec = PRIMARY_REGULARIZATION,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed + 10_000)
    dataset = make_dataset(train_metadata, config, training=True, seed_offset=10_000)
    class_weights = balanced_class_weights(train_metadata)
    write_json(config.output_dir / variant / "final_class_weights.json", class_weights)

    if variant in {"baseline", "regularized"}:
        model = build_scratch_cnn(
            config,
            regularized=variant == "regularized",
            specification=regularization_spec,
        )
        stage_name = f"{variant} final training"
        model.fit(
            dataset,
            epochs=int(schedule["epochs"]),
            class_weight=class_weights,
            callbacks=callbacks_for_final_stage(
                config.output_dir / variant / "final_training", stage_name
            ),
            verbose=2,
        )
    elif variant == "transfer":
        model = build_transfer_model(config)
        model.fit(
            dataset,
            epochs=int(schedule["head_epochs"]),
            class_weight=class_weights,
            callbacks=callbacks_for_final_stage(
                config.output_dir / variant / "final_head_training",
                "transfer final head training",
            ),
            verbose=2,
        )
        if bool(schedule["use_fine_tuning"]):
            enable_transfer_fine_tuning(model, config.fine_tune_last_layers)
            model.fit(
                dataset,
                epochs=int(schedule["fine_tune_epochs"]),
                class_weight=class_weights,
                callbacks=callbacks_for_final_stage(
                    config.output_dir / variant / "final_fine_tuning",
                    "transfer final fine-tuning",
                ),
                verbose=2,
            )
    else:
        raise ValueError(f"Unknown model: {variant}")
    model.save(config.output_dir / variant / "final_model.keras")
    return model


def write_and_log_training_plan(config: Config) -> None:
    plan: list[dict[str, Any]] = []
    for variant in config.models:
        if variant == "baseline":
            description = (
                f"{config.folds} CV fits x at most {config.max_epochs_baseline} "
                "epochs, then one final fit"
            )
            calls, maximum = config.folds, config.folds * config.max_epochs_baseline
        elif variant == "regularized":
            description = (
                f"{config.folds} CV fits x at most {config.max_epochs_regularized} "
                "epochs, then one final fit"
            )
            calls, maximum = config.folds, config.folds * config.max_epochs_regularized
        else:
            description = (
                f"{config.folds} folds x (head <= {config.max_epochs_transfer_head} + "
                f"fine-tuning <= {config.max_epochs_transfer_finetune} epochs), then "
                "one final head/fine-tuning fit"
            )
            calls = config.folds * 2
            maximum = config.folds * (
                config.max_epochs_transfer_head + config.max_epochs_transfer_finetune
            )
        LOGGER.info("Training plan for %s: %s", variant, description)
        plan.append(
            {
                "model": variant,
                "description": description,
                "cv_fit_calls": calls,
                "cv_maximum_epochs_before_early_stopping": maximum,
            }
        )
    write_json(config.output_dir / "training_plan.json", plan)
