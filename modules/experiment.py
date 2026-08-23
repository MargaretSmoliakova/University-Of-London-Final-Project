from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .common import LOGGER, Thresholds, write_json
from .config import Config
from .data_pipeline import make_dataset
from .metadata import prepare_and_audit_data
from .metrics import (
    bootstrap_confidence_intervals,
    paired_auc_bootstrap,
    save_operating_points,
    select_validation_thresholds,
    subgroup_rows,
    threshold_rows,
)
from .models import PRIMARY_REGULARIZATION
from .runtime import configure_runtime
from .training import (
    cross_validate_variant,
    make_development_calibration_split,
    train_final_model,
    write_and_log_training_plan,
)


def write_model_specifications(config: Config) -> None:
    write_json(
        config.output_dir / "model_specifications.json",
        {
            "baseline": {
                "architecture": "four Conv2D-MaxPool blocks, GAP, Dense(128), sigmoid",
                "learning_rate": 1e-3,
                "augmentation": False,
                "explicit_regularization": False,
            },
            "regularized": PRIMARY_REGULARIZATION.serialisable(),
            "transfer": {
                "backbone": "ImageNet EfficientNetB0",
                "head_learning_rate": 1e-3,
                "fine_tuning_learning_rate": 1e-5,
                "fine_tune_last_layers": config.fine_tune_last_layers,
                "batch_norm_frozen": True,
                "head_dropout": [0.40, 0.25],
                "head_l2": 1e-4,
                "augmentation": True,
            },
        },
    )


def save_oof_results(
    variant: str, oof: pd.DataFrame, config: Config
) -> Thresholds:
    exploratory_thresholds = select_validation_thresholds(
        oof["label"], oof["probability"], config.target_sensitivity
    )
    write_json(
        config.output_dir / variant / "exploratory_oof_thresholds.json",
        {
            "selection_source": "development_out_of_fold_predictions",
            **asdict(exploratory_thresholds),
        },
    )
    save_operating_points(
        oof["label"],
        oof["probability"],
        config.output_dir / variant / "oof_operating_points.csv",
    )
    return exploratory_thresholds


def calibrate_final_model(
    variant: str,
    final_model: tf.keras.Model,
    calibration_metadata: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray, Thresholds]:
    calibration_probability = final_model.predict(
        make_dataset(calibration_metadata, config, training=False), verbose=0
    ).ravel()
    thresholds = select_validation_thresholds(
        calibration_metadata["label"],
        calibration_probability,
        config.target_sensitivity,
    )
    write_json(
        config.output_dir / variant / "final_calibration_thresholds.json",
        {
            "selection_source": "reserved_patient_level_calibration_subset",
            **asdict(thresholds),
        },
    )
    save_operating_points(
        calibration_metadata["label"],
        calibration_probability,
        config.output_dir / variant / "calibration_operating_points.csv",
    )
    calibration_predictions = calibration_metadata[
        [
            "patient_id",
            "case_id",
            "lesion_type",
            "label",
            "image_path",
            "pathology_original",
        ]
    ].copy()
    calibration_predictions["probability"] = calibration_probability
    calibration_predictions.to_csv(
        config.output_dir / variant / "calibration_predictions.csv", index=False
    )
    return calibration_probability, thresholds


def save_test_predictions(
    variant: str,
    test_metadata: pd.DataFrame,
    test_probability: np.ndarray,
    thresholds: Thresholds,
    config: Config,
) -> None:
    prediction_frame = test_metadata[
        [
            "patient_id",
            "case_id",
            "lesion_type",
            "label",
            "image_path",
            "pathology_original",
        ]
    ].copy()
    prediction_frame["probability"] = test_probability
    for policy, threshold in asdict(thresholds).items():
        predicted = (test_probability >= threshold).astype(int)
        prediction_frame[f"prediction_{policy}"] = predicted
        prediction_frame[f"error_{policy}"] = np.select(
            [
                (prediction_frame.label.to_numpy() == 1) & (predicted == 0),
                (prediction_frame.label.to_numpy() == 0) & (predicted == 1),
            ],
            ["false_negative", "false_positive"],
            default="correct",
        )
    prediction_frame.to_csv(
        config.output_dir / variant / "official_test_predictions.csv", index=False
    )


def run_experiments(config: Config) -> None:
    configure_runtime(config)
    write_json(config.output_dir / "config.json", config.serialisable())
    write_model_specifications(config)
    if not config.audit_only:
        write_and_log_training_plan(config)
    train_metadata, test_metadata = prepare_and_audit_data(config)
    LOGGER.info("Audit summary: train=%d, test=%d", len(train_metadata), len(test_metadata))
    if config.audit_only:
        LOGGER.info("Audit-only run complete. Inspect sample_check before training.")
        return

    development_metadata, calibration_metadata = make_development_calibration_split(
        train_metadata, config
    )
    all_cv_rows: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_subgroup_rows: list[dict[str, Any]] = []
    all_ci_rows: list[pd.DataFrame] = []
    test_predictions: dict[str, np.ndarray] = {}

    for model_index, variant in enumerate(config.models):
        LOGGER.info("Starting model: %s", variant)
        cv_frame, oof, schedule = cross_validate_variant(
            variant, development_metadata, config
        )
        all_cv_rows.append(cv_frame)
        exploratory_thresholds = save_oof_results(variant, oof, config)
        all_metric_rows.extend(
            threshold_rows(
                variant,
                "development_oof",
                development_metadata,
                oof["probability"].to_numpy(),
                exploratory_thresholds,
            )
        )

        final_model = train_final_model(variant, development_metadata, schedule, config)
        calibration_probability, thresholds = calibrate_final_model(
            variant, final_model, calibration_metadata, config
        )
        all_metric_rows.extend(
            threshold_rows(
                variant,
                "reserved_calibration",
                calibration_metadata,
                calibration_probability,
                thresholds,
            )
        )

        test_probability = final_model.predict(
            make_dataset(test_metadata, config, training=False), verbose=0
        ).ravel()
        test_predictions[variant] = test_probability
        save_test_predictions(
            variant, test_metadata, test_probability, thresholds, config
        )
        all_metric_rows.extend(
            threshold_rows(
                variant, "official_test", test_metadata, test_probability, thresholds
            )
        )
        all_subgroup_rows.extend(
            subgroup_rows(variant, test_metadata, test_probability, thresholds)
        )
        ci_frame = bootstrap_confidence_intervals(
            test_metadata,
            test_probability,
            thresholds,
            repeats=config.bootstrap_repeats,
            seed=config.seed + 20_000 + model_index,
        )
        ci_frame.insert(0, "model", variant)
        all_ci_rows.append(ci_frame)

        del final_model
        tf.keras.backend.clear_session()

    pd.concat(all_cv_rows, ignore_index=True).to_csv(
        config.output_dir / "all_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame(all_metric_rows).to_csv(
        config.output_dir / "all_operating_point_metrics.csv", index=False
    )
    pd.DataFrame(all_subgroup_rows).to_csv(
        config.output_dir / "official_test_subgroup_metrics.csv", index=False
    )
    pd.concat(all_ci_rows, ignore_index=True).to_csv(
        config.output_dir / "official_test_cluster_bootstrap_cis.csv", index=False
    )
    paired_auc_bootstrap(
        test_metadata,
        test_predictions,
        repeats=config.bootstrap_repeats,
        seed=config.seed + 30_000,
    ).to_csv(config.output_dir / "official_test_paired_auc_differences.csv", index=False)
    LOGGER.info("All requested experiments completed")
