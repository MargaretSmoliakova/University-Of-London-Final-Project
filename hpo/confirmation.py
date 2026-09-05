from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.common import LOGGER, write_json
from modules.config import Config
from modules.metrics import compute_metrics

from .callbacks import DeadlineReached
from .config import REFERENCE_SPEC, spec_from_mapping
from .regularized_config import SearchConfig, SearchSpec
from .results import regularized_fold_summary
from .training import get_fold_trainer


def confirmation_candidates(
    study: Any, top_k: int, reference_spec: SearchSpec = REFERENCE_SPEC
) -> list[tuple[str, SearchSpec]]:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    ranked = sorted(
        completed,
        key=lambda item: (
            -(float(item.value) if item.value is not None else -np.inf),
            -float(item.user_attrs.get("mean_fold_roc_auc", -np.inf)),
            float(item.user_attrs.get("mean_fold_brier", np.inf)),
        ),
    )

    candidates: list[tuple[str, SearchSpec]] = [("reference", reference_spec)]
    seen = {json.dumps(reference_spec.serialisable(), sort_keys=True)}

    for trial in ranked:
        spec = spec_from_mapping(trial.params, type(reference_spec))
        key = json.dumps(spec.serialisable(), sort_keys=True)
        if key in seen:
            continue

        candidates.append((f"screen_trial_{trial.number:04d}", spec))
        seen.add(key)

        if len(candidates) >= top_k + 1:
            break

    return candidates


def _run_confirmation_folds(
    candidate_name: str,
    spec: SearchSpec,
    development: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment_config: Config,
    hpo_config: SearchConfig,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    total_deadline: float,
    candidate_dir: Path,
    confirmation_dir: Path,
    fold_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, bool]:
    probabilities = np.full(len(development), np.nan, dtype=float)
    assignments = np.zeros(len(development), dtype=int)
    complete = True
    train_fold = get_fold_trainer(spec)

    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        try:
            row, fold_probabilities = train_fold(
                spec=spec,
                development=development,
                train_indices=train_indices,
                validation_indices=validation_indices,
                experiment_config=experiment_config,
                hpo_config=hpo_config,
                fold_dir=candidate_dir / f"fold_{fold_number}",
                fold_number=fold_number,
                total_folds=len(splits),
                **hpo_config.training_epochs(confirmation=True),
                patience=hpo_config.confirmation_patience,
                deadline=total_deadline,
                preloaded=preloaded,
                trial=None,
                allow_pruning=False,
            )
        except DeadlineReached:
            LOGGER.warning(
                "Total time budget reached during confirmation of %s fold %d",
                candidate_name,
                fold_number,
            )
            complete = False
            break

        row = {"candidate": candidate_name, **row}
        fold_rows.append(row)
        probabilities[validation_indices] = fold_probabilities
        assignments[validation_indices] = fold_number
        pd.DataFrame(fold_rows).to_csv(
            confirmation_dir / "confirmation_fold_metrics.csv", index=False
        )

    return probabilities, assignments, complete


def run_confirmation(
    candidates: list[tuple[str, SearchSpec]],
    development: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment_config: Config,
    hpo_config: SearchConfig,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    total_deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmation_dir = hpo_config.output_dir / "confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        confirmation_dir / "candidate_parameters.json",
        {name: spec.serialisable() for name, spec in candidates},
    )

    fold_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    for candidate_name, spec in candidates:
        if time.time() >= total_deadline:
            LOGGER.warning("Total time budget reached before candidate %s", candidate_name)
            break

        LOGGER.info("Confirming candidate %s", candidate_name)
        candidate_dir = confirmation_dir / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "parameters.json", spec.serialisable())

        probabilities, assignments, complete = _run_confirmation_folds(
            candidate_name,
            spec,
            development,
            splits,
            experiment_config,
            hpo_config,
            preloaded,
            total_deadline,
            candidate_dir,
            confirmation_dir,
            fold_rows,
        )

        if complete and not np.isnan(probabilities).any():
            oof = development[
                ["patient_id", "case_id", "lesion_type", "label", "image_path"]
            ].copy()
            oof["candidate"] = candidate_name
            oof["fold"] = assignments
            oof["probability"] = probabilities
            oof_frames.append(oof)
            oof.to_csv(candidate_dir / "oof_predictions.csv", index=False)

    folds = pd.DataFrame(fold_rows)
    oof_all = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()

    folds.to_csv(confirmation_dir / "confirmation_fold_metrics.csv", index=False)
    oof_all.to_csv(confirmation_dir / "confirmation_oof_predictions.csv", index=False)

    return folds, oof_all


def _rank_confirmation_candidates(
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    confirmation_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if len(oof):
        complete_names = set(oof["candidate"].unique())

        for candidate in sorted(complete_names):
            candidate_folds = folds[folds["candidate"] == candidate]
            candidate_oof = oof[oof["candidate"] == candidate]
            metrics = compute_metrics(
                candidate_oof["label"], candidate_oof["probability"], threshold=0.5
            )

            rows.append(
                {
                    "candidate": candidate,
                    "mean_fold_average_precision": float(
                        candidate_folds["average_precision"].mean()
                    ),
                    "sd_fold_average_precision": float(
                        candidate_folds["average_precision"].std(ddof=1)
                    ),
                    "mean_fold_roc_auc": float(candidate_folds["roc_auc"].mean()),
                    "sd_fold_roc_auc": float(candidate_folds["roc_auc"].std(ddof=1)),
                    "mean_fold_brier": float(candidate_folds["brier_score"].mean()),
                    **regularized_fold_summary(candidate_folds),
                    "oof_average_precision": metrics["average_precision"],
                    "oof_roc_auc": metrics["roc_auc"],
                    "oof_brier": metrics["brier_score"],
                }
            )

    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(
            ["mean_fold_average_precision", "mean_fold_roc_auc", "mean_fold_brier"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    summary.to_csv(confirmation_dir / "confirmation_summary.csv", index=False)

    return summary


def _compare_with_reference(
    folds: pd.DataFrame, confirmation_dir: Path
) -> pd.DataFrame:
    comparison_rows: list[dict[str, Any]] = []
    reference = folds[folds["candidate"] == "reference"]

    if len(reference):
        reference = reference[["fold", "average_precision", "roc_auc", "brier_score"]].rename(
            columns={
                "average_precision": "reference_average_precision",
                "roc_auc": "reference_roc_auc",
                "brier_score": "reference_brier_score",
            }
        )

        for candidate in sorted(set(folds["candidate"]) - {"reference"}):
            challenger = folds[folds["candidate"] == candidate]
            merged = challenger.merge(reference, on="fold", validate="one_to_one")

            if len(merged):
                comparison_rows.append(
                    {
                        "candidate": candidate,
                        "paired_folds": len(merged),
                        "mean_ap_difference_vs_reference": float(
                            (
                                merged["average_precision"]
                                - merged["reference_average_precision"]
                            ).mean()
                        ),
                        "folds_with_better_ap_than_reference": int(
                            (
                                merged["average_precision"]
                                > merged["reference_average_precision"]
                            ).sum()
                        ),
                        "mean_roc_auc_difference_vs_reference": float(
                            (merged["roc_auc"] - merged["reference_roc_auc"]).mean()
                        ),
                        "mean_brier_difference_vs_reference": float(
                            (merged["brier_score"] - merged["reference_brier_score"]).mean()
                        ),
                    }
                )

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        confirmation_dir / "paired_differences_vs_reference.csv", index=False
    )

    return comparisons


def _save_selected_parameters(
    summary: pd.DataFrame,
    spec_map: dict[str, SearchSpec],
    confirmation_dir: Path,
) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None

    if len(summary):
        winner = summary.iloc[0]
        candidate = str(winner["candidate"])
        selected_parameters = spec_map[candidate].serialisable()
        reference_parameters = spec_map["reference"].serialisable()

        parameter_changes = [
            {
                "parameter": name,
                "reference_value": reference_parameters[name],
                "selected_value": selected_parameters[name],
                "changed_from_reference": reference_parameters[name]
                != selected_parameters[name],
            }
            for name in sorted(reference_parameters)
        ]

        pd.DataFrame(parameter_changes).to_csv(
            confirmation_dir / "selected_vs_reference_parameters.csv", index=False
        )

        selected = {
            "candidate": candidate,
            "parameters": selected_parameters,
            "parameters_changed_from_reference_count": int(
                sum(row["changed_from_reference"] for row in parameter_changes)
            ),
            "parameter_changes_vs_reference": parameter_changes,
            "confirmation_metrics": winner.to_dict(),
            "official_test_used_for_selection": False,
            "calibration_subset_used_for_selection": False,
        }

        write_json(confirmation_dir / "selected_parameters.json", selected)

    return selected


def summarise_confirmation(
    folds: pd.DataFrame,
    oof: pd.DataFrame,
    candidates: list[tuple[str, SearchSpec]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    confirmation_dir = output_dir / "confirmation"
    spec_map = {name: spec for name, spec in candidates}

    summary = _rank_confirmation_candidates(folds, oof, confirmation_dir)

    comparisons = _compare_with_reference(folds, confirmation_dir)

    selected = _save_selected_parameters(summary, spec_map, confirmation_dir)

    return summary, comparisons, selected
