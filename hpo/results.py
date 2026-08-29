from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.common import LOGGER, write_json

from .config import SEARCH_SPACE
from .plots import create_screen_plots
from .runtime import optuna


def trial_table(study: Any) -> pd.DataFrame:
    parameter_names = sorted(SEARCH_SPACE)

    rows: list[dict[str, Any]] = []

    for trial in study.get_trials(deepcopy=False):
        duration = trial.duration.total_seconds() if trial.duration is not None else np.nan
        row: dict[str, Any] = {
            "trial": int(trial.number),
            "state": trial.state.name,
            "objective_mean_fold_average_precision": trial.value,
            "duration_seconds": duration,
            "completed_folds": trial.user_attrs.get("completed_folds", 0),
            "mean_fold_roc_auc": trial.user_attrs.get("mean_fold_roc_auc", np.nan),
            "mean_fold_brier": trial.user_attrs.get("mean_fold_brier", np.nan),
            "sd_fold_average_precision": trial.user_attrs.get(
                "sd_fold_average_precision", np.nan
            ),
            "oof_average_precision": trial.user_attrs.get(
                "oof_average_precision", np.nan
            ),
            "oof_roc_auc": trial.user_attrs.get("oof_roc_auc", np.nan),
            "oof_brier": trial.user_attrs.get("oof_brier", np.nan),
            "failure": trial.user_attrs.get("failure", ""),
        }

        for name in parameter_names:
            row[f"param_{name}"] = trial.params.get(name, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def save_live_trials(study: Any, output_dir: Path) -> None:
    table = trial_table(study)
    table.to_csv(output_dir / "all_trials.csv", index=False)

    if len(table):
        table[table["state"] == "COMPLETE"].sort_values(
            "objective_mean_fold_average_precision", ascending=False
        ).to_csv(output_dir / "completed_trials_ranked.csv", index=False)
        table[table["state"] == "PRUNED"].to_csv(
            output_dir / "pruned_trials.csv", index=False
        )
        table[table["state"] == "FAIL"].to_csv(
            output_dir / "failed_trials.csv", index=False
        )


def calculate_parameter_effects(
    completed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    objective = "objective_mean_fold_average_precision"

    for name in sorted(SEARCH_SPACE):
        column = f"param_{name}"
        subset = completed[[column, objective]].dropna().copy()
        if subset.empty:
            continue

        numeric = pd.to_numeric(subset[column], errors="coerce")
        is_numeric = bool(numeric.notna().all())
        unique_count = int(subset[column].nunique())

        if is_numeric:
            subset["numeric_value"] = numeric.astype(float)
            correlation = subset["numeric_value"].corr(subset[objective], method="spearman")
            correlation_rows.append(
                {
                    "parameter": name,
                    "spearman_rank_correlation_with_mean_ap": correlation,
                    "completed_trials": len(subset),
                }
            )

        if is_numeric and unique_count > 8:
            number_of_bins = min(4, unique_count)
            try:
                subset["region"] = pd.qcut(
                    subset["numeric_value"], q=number_of_bins, duplicates="drop"
                ).astype(str)
            except ValueError:
                subset["region"] = subset["numeric_value"].map(lambda value: f"{value:.6g}")
        else:
            subset["region"] = subset[column].astype(str)

        for region, group in subset.groupby("region", sort=False):
            effect_rows.append(
                {
                    "parameter": name,
                    "observed_value_or_region": region,
                    "completed_trials": int(len(group)),
                    "mean_ap": float(group[objective].mean()),
                    "median_ap": float(group[objective].median()),
                    "best_ap": float(group[objective].max()),
                    "worst_ap": float(group[objective].min()),
                }
            )

    return pd.DataFrame(effect_rows), pd.DataFrame(correlation_rows)


def summarise_parameter_effects(
    effects: pd.DataFrame, importances: pd.DataFrame
) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()

    importance_map = (
        importances.set_index("parameter")["importance"].to_dict()
        if len(importances)
        else {}
    )

    rows: list[dict[str, Any]] = []

    for parameter, group in effects.groupby("parameter"):
        eligible_regions = group[group["completed_trials"] >= 2]
        regions = eligible_regions if len(eligible_regions) else group
        best_region = regions.sort_values("mean_ap", ascending=False).iloc[0]
        worst_region = regions.sort_values("mean_ap", ascending=True).iloc[0]
        mean_ap_difference = float(best_region["mean_ap"] - worst_region["mean_ap"])

        rows.append(
            {
                "parameter": parameter,
                "importance": importance_map.get(parameter, np.nan),
                "best_observed_value_or_region": best_region["observed_value_or_region"],
                "best_region_mean_ap": best_region["mean_ap"],
                "worst_observed_value_or_region": worst_region["observed_value_or_region"],
                "worst_region_mean_ap": worst_region["mean_ap"],
                "observed_mean_ap_difference": mean_ap_difference,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["importance", "observed_mean_ap_difference"], ascending=[False, False]
    )


def parameter_importance_table(study: Any) -> pd.DataFrame:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if len(completed) < 3:
        return pd.DataFrame(columns=["parameter", "importance"])

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        LOGGER.warning("Parameter importance failed: %s", type(exc).__name__)

        return pd.DataFrame(columns=["parameter", "importance"])

    return pd.DataFrame(
        [{"parameter": name, "importance": value} for name, value in importances.items()]
    )


def save_screen_analysis(study: Any, output_dir: Path) -> dict[str, pd.DataFrame]:
    save_live_trials(study, output_dir)
    trials = trial_table(study)
    completed = trials[trials["state"] == "COMPLETE"].copy()

    importances = parameter_importance_table(study)
    importances.to_csv(output_dir / "parameter_importance.csv", index=False)

    effects, correlations = calculate_parameter_effects(completed)
    effects.to_csv(output_dir / "parameter_effect_summary.csv", index=False)
    correlations.to_csv(output_dir / "parameter_rank_correlations.csv", index=False)

    parameter_summary = summarise_parameter_effects(effects, importances)
    parameter_summary.to_csv(
        output_dir / "automatic_parameter_interpretation.csv", index=False
    )

    create_screen_plots(trials, importances, output_dir)

    if len(completed):
        best = completed.sort_values(
            [
                "objective_mean_fold_average_precision",
                "mean_fold_roc_auc",
                "mean_fold_brier",
            ],
            ascending=[False, False, True],
        ).iloc[0]

        trial_number = int(best["trial"])
        trial = next(item for item in study.trials if item.number == trial_number)

        write_json(
            output_dir / "best_screen_trial.json",
            {
                "trial": trial_number,
                "parameters": trial.params,
                "mean_fold_average_precision": trial.value,
                "mean_fold_roc_auc": trial.user_attrs.get("mean_fold_roc_auc"),
                "mean_fold_brier": trial.user_attrs.get("mean_fold_brier"),
            },
        )

    return {
        "trials": trials,
        "completed": completed,
        "importances": importances,
        "effects": effects,
        "correlations": correlations,
        "interpretation": parameter_summary,
    }
