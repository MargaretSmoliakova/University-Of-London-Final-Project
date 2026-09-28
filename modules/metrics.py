"""
Measure how well models distinguish benign and malignant lesions.

These helpers choose decision thresholds, summarise prediction quality, estimate
uncertainty, and compare models and lesion types.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from .common import Thresholds


def safe_ratio(numerator: int, denominator: int) -> float:
    """
    Calculate a ratio, returning an unavailable value when its denominator is zero.
    """
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(
    true_labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    """
    Summarise prediction quality using malignancy scores and a chosen decision cutoff.

    Include overall scores and counts of correct predictions, missed cancers,
    and false alarms.
    """
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    has_two_classes = np.unique(y_true).size == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": safe_ratio(int(tp), int(tp + fn)),
        "specificity": safe_ratio(int(tn), int(tn + fp)),
        "precision_ppv": float(precision_score(y_true, predicted, zero_division=0)),
        "negative_predictive_value": safe_ratio(int(tn), int(tn + fn)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_probability))
            if has_two_classes
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y_true, y_probability))
            if has_two_classes
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_validation_thresholds(
    true_labels: Sequence[int], probabilities: Sequence[float], target_sensitivity: float
) -> Thresholds:
    """
    Choose decision cutoffs from validation predictions for later evaluation.

    Keep the default cutoff and select alternatives that balance sensitivity
    and specificity or meet the requested sensitivity.
    """
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probability, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    youden = tpr - fpr
    best = np.flatnonzero(youden == np.max(youden))
    youden_index = int(best[np.argmax(thresholds[best])])
    eligible = np.flatnonzero(tpr >= target_sensitivity)
    if not len(eligible):
        raise RuntimeError(f"Sensitivity target unmet: {target_sensitivity:.3f}")
    best_fpr = np.min(fpr[eligible])
    ties = eligible[fpr[eligible] == best_fpr]
    sensitivity_index = int(ties[np.argmax(thresholds[ties])])
    return Thresholds(
        default=0.5,
        youden_validation=float(thresholds[youden_index]),
        sensitivity_target_validation=float(thresholds[sensitivity_index]),
    )


def save_operating_points(
    true_labels: Sequence[int], probabilities: Sequence[float], output_path: Path
) -> None:
    """Save how sensitivity and specificity change across possible decision cutoffs."""
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probability, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    pd.DataFrame(
        {
            "threshold": thresholds[finite],
            "sensitivity": tpr[finite],
            "specificity": 1.0 - fpr[finite],
            "youden_j": tpr[finite] - fpr[finite],
        }
    ).to_csv(output_path, index=False)


def threshold_rows(
    model_name: str,
    split_name: str,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    """
    Build performance-report rows for each decision rule on a named model and data
    split.
    """
    return [
        {
            "model": model_name,
            "split": split_name,
            "threshold_policy": policy,
            **compute_metrics(metadata["label"], probabilities, threshold),
        }
        for policy, threshold in asdict(thresholds).items()
    ]


def bootstrap_confidence_intervals(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Thresholds,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Estimate uncertainty in performance by repeatedly resampling whole patients.

    Keeping each patient's images together preserves their grouping when calculating
    confidence intervals.
    """
    groups = metadata["patient_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    y_true = metadata["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    policies = asdict(thresholds)
    metric_names = (
        "roc_auc",
        "average_precision",
        "brier_score",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision_ppv",
        "negative_predictive_value",
        "f1",
    )
    draws: dict[tuple[str, str], list[float]] = {
        (policy, metric): [] for policy in policies for metric in metric_names
    }
    for _ in range(repeats):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_labels = y_true[sampled_indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_probabilities = probabilities[sampled_indices]
        for policy, threshold in policies.items():
            metrics = compute_metrics(sampled_labels, sampled_probabilities, threshold)
            for metric in metric_names:
                value = float(metrics[metric])
                if np.isfinite(value):
                    draws[(policy, metric)].append(value)
    rows: list[dict[str, Any]] = []
    for (policy, metric), values in draws.items():
        if values:
            rows.append(
                {
                    "threshold_policy": policy,
                    "metric": metric,
                    "ci_lower_95": float(np.percentile(values, 2.5)),
                    "ci_upper_95": float(np.percentile(values, 97.5)),
                    "valid_bootstrap_repeats": len(values),
                }
            )
    return pd.DataFrame(rows)


def subgroup_rows(
    model_name: str,
    test_metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    """Report prediction quality separately for each lesion type and decision rule."""
    rows: list[dict[str, Any]] = []
    for lesion_type, indices in test_metadata.groupby("lesion_type").groups.items():
        subset = test_metadata.loc[indices]
        subset_probabilities = probabilities[np.asarray(indices, dtype=int)]
        for policy, threshold in asdict(thresholds).items():
            rows.append(
                {
                    "model": model_name,
                    "lesion_type": lesion_type,
                    "threshold_policy": policy,
                    **compute_metrics(subset["label"], subset_probabilities, threshold),
                }
            )
    return rows


def paired_auc_bootstrap(
    test_metadata: pd.DataFrame,
    prediction_columns: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compare model scores using the same resampled patients for each pair.

    Estimate the difference in class-separation performance and its uncertainty.
    """
    if len(prediction_columns) < 2:
        return pd.DataFrame()
    groups = test_metadata["patient_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    y_true = test_metadata["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    model_names = list(prediction_columns)
    rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(model_names):
        for second in model_names[first_index + 1 :]:
            observed = roc_auc_score(y_true, prediction_columns[second]) - roc_auc_score(
                y_true, prediction_columns[first]
            )
            differences: list[float] = []
            for _ in range(repeats):
                sampled_groups = rng.choice(
                    unique_groups, size=len(unique_groups), replace=True
                )
                indices = np.concatenate([group_indices[group] for group in sampled_groups])
                labels = y_true[indices]
                if np.unique(labels).size < 2:
                    continue
                difference = roc_auc_score(
                    labels, prediction_columns[second][indices]
                ) - roc_auc_score(labels, prediction_columns[first][indices])
                differences.append(float(difference))
            rows.append(
                {
                    "reference_model": first,
                    "comparison_model": second,
                    "test_auc_difference_comparison_minus_reference": float(observed),
                    "ci_lower_95": float(np.percentile(differences, 2.5)),
                    "ci_upper_95": float(np.percentile(differences, 97.5)),
                    "valid_bootstrap_repeats": len(differences),
                }
            )
    return pd.DataFrame(rows)
