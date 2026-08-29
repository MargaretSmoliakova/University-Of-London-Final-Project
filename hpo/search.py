from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.common import LOGGER, write_json
from modules.config import Config
from modules.metrics import compute_metrics

from .callbacks import DeadlineReached
from .config import HPOConfig, REFERENCE_SPEC, TransferSearchSpec, suggest_spec
from .runtime import optuna
from .training import train_transfer_fold


def _report_trial_progress(
    trial: Any,
    fold_rows: list[dict[str, Any]],
    fold_number: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    hpo_config: HPOConfig,
    spec: TransferSearchSpec,
) -> None:
    trial.set_user_attr("completed_folds", len(fold_rows))

    running_ap = float(np.mean([item["average_precision"] for item in fold_rows]))

    LOGGER.info(
        "Trial %d fold %d/%d complete; running mean AP=%.5f",
        trial.number,
        fold_number,
        len(splits),
        running_ap,
    )

    if fold_number >= 2:
        trial.report(
            running_ap,
            step=len(splits) * (
                hpo_config.screen_head_epochs
                + hpo_config.screen_finetune_epochs
            )
            + fold_number,
        )
        if spec != REFERENCE_SPEC and trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned after fold {fold_number}"
            )


def _save_trial_results(
    trial: Any,
    trial_dir: Path,
    development: pd.DataFrame,
    fold_rows: list[dict[str, Any]],
    oof_probability: np.ndarray,
    oof_fold: np.ndarray,
) -> float:
    if np.isnan(oof_probability).any():
        raise RuntimeError(f"Incomplete OOF predictions: trial {trial.number}")

    folds = pd.DataFrame(fold_rows)
    oof = development[
        ["patient_id", "case_id", "lesion_type", "label", "image_path"]
    ].copy()
    oof["trial"] = trial.number
    oof["fold"] = oof_fold
    oof["probability"] = oof_probability
    oof.to_csv(trial_dir / "oof_predictions.csv", index=False)

    oof_metrics = compute_metrics(oof["label"], oof_probability, threshold=0.5)
    mean_ap = float(folds["average_precision"].mean())

    trial.set_user_attr("completed_folds", len(folds))
    trial.set_user_attr("mean_fold_roc_auc", float(folds["roc_auc"].mean()))
    trial.set_user_attr("mean_fold_brier", float(folds["brier_score"].mean()))
    trial.set_user_attr(
        "sd_fold_average_precision",
        float(folds["average_precision"].std(ddof=1)) if len(folds) > 1 else 0.0,
    )
    trial.set_user_attr("oof_average_precision", oof_metrics["average_precision"])
    trial.set_user_attr("oof_roc_auc", oof_metrics["roc_auc"])
    trial.set_user_attr("oof_brier", oof_metrics["brier_score"])

    write_json(
        trial_dir / "result.json",
        {
            "objective_mean_fold_average_precision": mean_ap,
            "mean_fold_roc_auc": float(folds["roc_auc"].mean()),
            "mean_fold_brier": float(folds["brier_score"].mean()),
            "oof_metrics": oof_metrics,
        },
    )
    LOGGER.info("Trial %d completed with mean fold AP=%.5f", trial.number, mean_ap)

    return mean_ap


def make_objective(
    development: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment_config: Config,
    hpo_config: HPOConfig,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    search_deadline: float,
):
    def objective(trial: Any) -> float:
        if time.time() >= search_deadline:
            raise optuna.TrialPruned("Search time budget reached")

        trial_dir = hpo_config.output_dir / "screen_trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        spec = suggest_spec(trial)
        write_json(trial_dir / "parameters.json", spec.serialisable())
        LOGGER.info("Starting trial %d with parameters %s", trial.number, spec.serialisable())

        fold_rows: list[dict[str, Any]] = []
        oof_probability = np.full(len(development), np.nan, dtype=float)
        oof_fold = np.zeros(len(development), dtype=int)

        try:
            for fold_number, (train_indices, validation_indices) in enumerate(
                splits, start=1
            ):
                row, probabilities = train_transfer_fold(
                    spec=spec,
                    development=development,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    experiment_config=experiment_config,
                    hpo_config=hpo_config,
                    fold_dir=trial_dir / f"fold_{fold_number}",
                    fold_number=fold_number,
                    total_folds=len(splits),
                    head_epochs=hpo_config.screen_head_epochs,
                    finetune_epochs=hpo_config.screen_finetune_epochs,
                    patience=hpo_config.screen_patience,
                    deadline=search_deadline,
                    preloaded=preloaded,
                    trial=trial,
                    allow_pruning=spec != REFERENCE_SPEC,
                )

                row = {"trial": int(trial.number), **row}
                fold_rows.append(row)
                oof_probability[validation_indices] = probabilities
                oof_fold[validation_indices] = fold_number
                pd.DataFrame(fold_rows).to_csv(trial_dir / "fold_metrics.csv", index=False)

                _report_trial_progress(
                    trial, fold_rows, fold_number, splits, hpo_config, spec
                )
        except DeadlineReached as exc:
            trial.set_user_attr("failure", str(exc))
            raise optuna.TrialPruned(str(exc)) from exc
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            (trial_dir / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise

        return _save_trial_results(
            trial, trial_dir, development, fold_rows, oof_probability, oof_fold
        )

    return objective


def load_or_create_study(hpo_config: HPOConfig) -> Any:
    database_path = (hpo_config.output_dir / "study.db").resolve()
    storage_url = f"sqlite:///{database_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(
        seed=hpo_config.seed,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=hpo_config.startup_trials,
        n_warmup_steps=4,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name="cbis_ddsm_efficientnetb0_cpu_hpo",
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    reference_parameters = REFERENCE_SPEC.serialisable()
    reference_is_present = any(
        trial.params == reference_parameters
        and trial.state.name in {"COMPLETE", "RUNNING", "WAITING"}
        for trial in study.trials
    )
    if not reference_is_present:
        study.enqueue_trial(REFERENCE_SPEC.serialisable())

    return study
