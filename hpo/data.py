from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from modules.common import LOGGER, write_json
from modules.config import Config
from modules.data_pipeline import make_dataset
from modules.metadata import prepare_and_audit_data
from modules.training import make_development_calibration_split, stratification_labels

from .config import HPOConfig, SEARCH_SPACE


def preload_images(
    development: pd.DataFrame,
    experiment_config: Config,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    started = time.perf_counter()
    LOGGER.info(
        "Preloading %d development images",
        len(development),
    )

    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    dataset = make_dataset(development, experiment_config, training=False)

    for batch_images, batch_labels in dataset:
        arrays.append(np.asarray(batch_images, dtype=np.float32))
        labels.append(np.asarray(batch_labels, dtype=np.float32))

    images = np.concatenate(arrays, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    elapsed = time.perf_counter() - started

    if len(images) != len(development):
        raise RuntimeError("Incomplete image preload")

    payload = {
        "rows": int(len(images)),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "memory_gib": float(images.nbytes / (1024**3)),
        "elapsed_seconds": float(elapsed),
    }

    write_json(output_dir / "preloaded_image_cache_summary.json", payload)
    LOGGER.info(
        "Preloaded images in %.1f seconds; RAM array uses %.2f GiB",
        elapsed,
        payload["memory_gib"],
    )

    return images, all_labels


def make_array_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images[indices], labels[indices]))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    if training:
        dataset = dataset.shuffle(
            len(indices), seed=seed, reshuffle_each_iteration=True
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_fold_datasets(
    development: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    experiment_config: Config,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    if preloaded is not None:
        images, labels = preloaded

        return (
            make_array_dataset(
                images,
                labels,
                train_indices,
                experiment_config.batch_size,
                training=True,
                seed=seed,
            ),
            make_array_dataset(
                images,
                labels,
                validation_indices,
                experiment_config.batch_size,
                training=False,
                seed=seed,
            ),
        )

    train = development.iloc[train_indices].reset_index(drop=True)
    validation = development.iloc[validation_indices].reset_index(drop=True)

    return (
        make_dataset(train, experiment_config, training=True, seed_offset=seed),
        make_dataset(validation, experiment_config, training=False),
    )


def make_splits(
    development: pd.DataFrame, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    return list(
        splitter.split(
            development,
            stratification_labels(development),
            groups=development["patient_id"],
        )
    )


def save_fold_assignments(
    development: pd.DataFrame,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    rows: list[pd.DataFrame] = []
    base_columns = ["patient_id", "case_id", "lesion_type", "label", "image_path"]

    for fold_number, (_, validation_indices) in enumerate(splits, start=1):
        part = development.iloc[validation_indices][base_columns].copy()
        part.insert(0, "validation_fold", fold_number)
        rows.append(part)

    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def development_fingerprint(development: pd.DataFrame) -> str:
    columns = ["patient_id", "case_id", "lesion_type", "label", "image_path"]
    ordered = development[columns].astype(str).sort_values(columns).reset_index(drop=True)

    digest = hashlib.sha256()

    for row in ordered.itertuples(index=False, name=None):
        digest.update("\t".join(row).encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def ensure_study_contract(
    hpo_config: HPOConfig,
    development: pd.DataFrame,
    screen_splits: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    path = hpo_config.output_dir / "study_contract.json"
    contract = {
        "study_type": "exploratory transfer-learning hyperparameter optimisation",
        "model_family": "ImageNet EfficientNetB0",
        "primary_objective": "mean exact Average Precision across screen folds",
        "official_test_used_for_prediction_or_selection": False,
        "calibration_subset_used_for_prediction_or_selection": False,
        "development_rows": int(len(development)),
        "development_patients": int(development["patient_id"].nunique()),
        "development_fingerprint": development_fingerprint(development),
        "screen_folds": hpo_config.screen_folds,
        "screen_validation_row_counts": [int(len(indices)) for _, indices in screen_splits],
        "screen_head_epochs": hpo_config.screen_head_epochs,
        "screen_finetune_epochs": hpo_config.screen_finetune_epochs,
        "screen_patience": hpo_config.screen_patience,
        "confirmation_folds": hpo_config.confirmation_folds,
        "confirmation_top_k": hpo_config.confirmation_top_k,
        "confirmation_head_epochs": hpo_config.confirmation_head_epochs,
        "confirmation_finetune_epochs": hpo_config.confirmation_finetune_epochs,
        "confirmation_patience": hpo_config.confirmation_patience,
        "image_size": hpo_config.image_size,
        "batch_size": hpo_config.batch_size,
        "seed": hpo_config.seed,
        "calibration_fraction": hpo_config.calibration_fraction,
        "lesion_type": hpo_config.lesion_type,
        "combined_patient_overlap_policy": hpo_config.combined_patient_overlap_policy,
        "include_benign_without_callback": hpo_config.include_benign_without_callback,
        "search_space": SEARCH_SPACE,
    }

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Study configuration mismatch")
    else:
        write_json(path, contract)


def prepare_hpo_data(
    hpo_config: HPOConfig,
    experiment_config: Config,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    LOGGER.info("Preparing and auditing dataset before HPO")
    adjusted_train, _ = prepare_and_audit_data(experiment_config)

    development, calibration_not_used = make_development_calibration_split(
        adjusted_train, experiment_config
    )

    write_json(
        hpo_config.output_dir / "hpo_data_roles.json",
        {
            "development_rows_used_for_hpo": int(len(development)),
            "development_patients": int(development["patient_id"].nunique()),
            "calibration_rows_not_used_for_hpo": int(len(calibration_not_used)),
            "calibration_patients": int(calibration_not_used["patient_id"].nunique()),
            "official_test_used_for_audit_only": True,
        },
    )

    screen_splits = make_splits(development, hpo_config.screen_folds, hpo_config.seed)
    confirmation_splits = make_splits(
        development, hpo_config.confirmation_folds, hpo_config.seed + 1001
    )

    save_fold_assignments(
        development, screen_splits, hpo_config.output_dir / "screen_fold_assignments.csv"
    )

    save_fold_assignments(
        development,
        confirmation_splits,
        hpo_config.output_dir / "confirmation_fold_assignments.csv",
    )

    ensure_study_contract(hpo_config, development, screen_splits)

    return development, screen_splits, confirmation_splits
