from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .config import SEARCH_SPACE


def _plot_optimization_history(completed: pd.DataFrame, plot_dir: Path, plt: ModuleType) -> None:
    trial_numbers = completed["trial"].to_numpy(dtype=int)
    mean_ap_values = completed["objective_mean_fold_average_precision"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.scatter(trial_numbers, mean_ap_values, color="#2b6cb0", label="Completed trial", zorder=3)
    axis.plot(
        trial_numbers,
        np.maximum.accumulate(mean_ap_values),
        color="#c53030",
        label="Best so far",
    )

    axis.set_xlabel("Trial number")
    axis.set_ylabel("Mean screen-fold Average Precision")
    axis.set_title("Hyperparameter optimisation history")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(plot_dir / "optimization_history.png", dpi=180)
    plt.close(figure)


def _plot_top_trials(completed: pd.DataFrame, plot_dir: Path, plt: ModuleType) -> None:
    top = completed.nlargest(min(12, len(completed)), "objective_mean_fold_average_precision")
    top = top.sort_values("objective_mean_fold_average_precision")

    figure, axis = plt.subplots(figsize=(10, max(5, 0.42 * len(top))))
    errors = top["sd_fold_average_precision"].fillna(0).to_numpy(dtype=float)
    axis.errorbar(
        top["objective_mean_fold_average_precision"],
        [f"Trial {value}" for value in top["trial"]],
        xerr=errors,
        fmt="o",
        color="#2f855a",
        ecolor="#9ae6b4",
        capsize=4,
    )

    axis.set_xlabel("Mean Average Precision (error bar = fold SD)")
    axis.set_title("Highest-ranked completed screening trials")
    axis.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(plot_dir / "top_trials_with_fold_variability.png", dpi=180)
    plt.close(figure)


def _plot_parameter_importance(importances: pd.DataFrame, plot_dir: Path, plt: ModuleType) -> None:
    if len(importances):
        shown = importances.sort_values("importance").tail(14)
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.barh(shown["parameter"], shown["importance"], color="#805ad5")

        axis.set_xlabel("Relative importance")
        axis.set_title("Optuna parameter importance within completed trials")
        axis.grid(axis="x", alpha=0.25)

        figure.tight_layout()
        figure.savefig(plot_dir / "parameter_importance.png", dpi=180)
        plt.close(figure)


def _plot_parameter_effects(
    completed: pd.DataFrame,
    plot_dir: Path,
    plt: ModuleType,
    search_space: dict[str, Any],
) -> None:
    parameter_names = sorted(search_space)
    columns = 3
    rows = math.ceil(len(parameter_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.1 * rows))
    flat_axes = np.asarray(axes).reshape(-1)
    objective = "objective_mean_fold_average_precision"

    for axis, name in zip(flat_axes, parameter_names):
        series = completed[f"param_{name}"]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            axis.scatter(numeric, completed[objective], alpha=0.75, color="#2b6cb0")
            if "learning_rate" in name or name == "l2_strength":
                axis.set_xscale("log")
        else:
            categories = list(dict.fromkeys(series.astype(str)))
            positions = {category: index for index, category in enumerate(categories)}
            axis.scatter(
                [positions[value] for value in series.astype(str)],
                completed[objective],
                alpha=0.75,
                color="#2b6cb0",
            )
            axis.set_xticks(range(len(categories)), categories, rotation=30, ha="right")

        axis.set_title(name)
        axis.set_ylabel("Mean AP")
        axis.grid(alpha=0.2)

    for axis in flat_axes[len(parameter_names) :]:
        axis.axis("off")

    figure.suptitle(
        "Parameter values and screening performance",
        fontsize=14,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(plot_dir / "all_parameter_effects.png", dpi=180)
    plt.close(figure)


def create_screen_plots(
    trials: pd.DataFrame,
    importances: pd.DataFrame,
    output_dir: Path,
    search_space: dict[str, Any] = SEARCH_SPACE,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    completed = trials[trials["state"] == "COMPLETE"].sort_values("trial")
    if completed.empty:
        return

    _plot_optimization_history(completed, plot_dir, plt)

    _plot_top_trials(completed, plot_dir, plt)

    _plot_parameter_importance(importances, plot_dir, plt)

    _plot_parameter_effects(completed, plot_dir, plt, search_space)


def create_confirmation_plot(
    summary: pd.DataFrame, folds: pd.DataFrame, output_dir: Path
) -> None:
    if summary.empty:
        return

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("mean_fold_average_precision")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].errorbar(
        ordered["mean_fold_average_precision"],
        ordered["candidate"],
        xerr=ordered["sd_fold_average_precision"],
        fmt="o",
        color="#2f855a",
        ecolor="#9ae6b4",
        capsize=5,
    )

    axes[0].set_xlabel("Mean confirmation-fold AP (error bar = fold SD)")
    axes[0].set_title("Confirmation ranking")
    axes[0].grid(axis="x", alpha=0.25)

    for candidate, group in folds.groupby("candidate"):
        if candidate not in set(summary["candidate"]):
            continue
        axes[1].plot(
            group["fold"],
            group["average_precision"],
            marker="o",
            label=candidate,
        )

    axes[1].set_xlabel("Confirmation fold")
    axes[1].set_ylabel("Average Precision")
    axes[1].set_title("Fold-to-fold stability")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(plot_dir / "confirmation_comparison.png", dpi=180)
    plt.close(figure)
