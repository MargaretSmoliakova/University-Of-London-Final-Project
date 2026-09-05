from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from modules.common import LOGGER, write_json
from modules.config import Config
from modules.runtime import configure_runtime

from .regularized_config import SearchConfig
from .confirmation import (
    confirmation_candidates,
    run_confirmation,
    summarise_confirmation,
)
from .data import preload_images, prepare_hpo_data
from .plots import create_confirmation_plot
from .results import save_live_trials, save_screen_analysis
from .runtime import configure_cpu_threads, environment_summary, optuna
from .search import load_or_create_study, make_objective


def _run_screening(
    study: Any,
    development: pd.DataFrame,
    screen_splits: list[tuple[np.ndarray, np.ndarray]],
    experiment_config: Config,
    hpo_config: SearchConfig,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    search_deadline: float,
    remaining_search_seconds: float,
) -> None:
    remaining_trials = max(0, hpo_config.max_trials - len(study.trials))

    def after_trial(study_object: Any, frozen_trial: Any) -> None:
        save_live_trials(
            study_object,
            hpo_config.output_dir,
            hpo_config.search_space,
            hpo_config.trial_attributes,
        )
        completed = [
            trial for trial in study_object.trials if trial.state.name == "COMPLETE"
        ]

        best = max((trial.value for trial in completed), default=float("nan"))
        LOGGER.info(
            "Study progress: total=%d, completed=%d, latest_state=%s, best_mean_AP=%s",
            len(study_object.trials),
            len(completed),
            frozen_trial.state.name,
            f"{best:.5f}" if np.isfinite(best) else "not available",
        )

    if remaining_trials > 0 and remaining_search_seconds > 1:
        objective = make_objective(
            development,
            screen_splits,
            experiment_config,
            hpo_config,
            preloaded,
            search_deadline,
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_search_seconds,
            n_jobs=1,
            callbacks=[after_trial],
            catch=(RuntimeError, ValueError, MemoryError),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    else:
        LOGGER.info("No additional screening trials scheduled; using saved study")


def _confirm_candidates(
    study: Any,
    screen_analysis: dict[str, pd.DataFrame],
    development: pd.DataFrame,
    confirmation_splits: list[tuple[np.ndarray, np.ndarray]],
    experiment_config: Config,
    hpo_config: SearchConfig,
    preloaded: tuple[np.ndarray, np.ndarray] | None,
    total_deadline: float,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    candidates = confirmation_candidates(
        study, hpo_config.confirmation_top_k, hpo_config.reference_spec
    )
    confirmation_summary = pd.DataFrame()
    selected = None

    if hpo_config.skip_confirmation:
        LOGGER.warning("Confirmation was explicitly skipped")
    elif not screen_analysis["completed"].empty and time.time() < total_deadline:
        confirmation_folds, confirmation_oof = run_confirmation(
            candidates,
            development,
            confirmation_splits,
            experiment_config,
            hpo_config,
            preloaded,
            total_deadline,
        )
        confirmation_summary, _, selected = summarise_confirmation(
            confirmation_folds,
            confirmation_oof,
            candidates,
            hpo_config.output_dir,
        )
        if selected is not None and hpo_config.selection_filename is not None:
            write_json(hpo_config.output_dir / hpo_config.selection_filename, selected)

        create_confirmation_plot(
            confirmation_summary, confirmation_folds, hpo_config.output_dir
        )
    else:
        LOGGER.warning("Confirmation unavailable: no completed trials or time remaining")

    return confirmation_summary, selected


def _save_run_status(
    hpo_config: SearchConfig,
    screen_analysis: dict[str, pd.DataFrame],
    confirmation_summary: pd.DataFrame,
    selected: dict[str, Any] | None,
    elapsed: float,
) -> None:
    selection_status = {}
    if hpo_config.selection_filename is not None:
        selection_status = {
            "status": "completed" if selected is not None else "completed_without_selection",
            "selection_ready_for_full_run": selected is not None,
        }

    write_json(
        hpo_config.output_dir / "run_status.json",
        {
            "status": "completed",
            **selection_status,
            "elapsed_seconds": elapsed,
            "screen_trials_total": int(len(screen_analysis["trials"])),
            "screen_trials_complete": int(len(screen_analysis["completed"])),
            "confirmation_candidates_complete": int(len(confirmation_summary)),
            "selected_candidate": selected["candidate"] if selected else None,
            "official_test_evaluated": False,
            "calibration_subset_used_for_selection": False,
        },
    )


def run_hpo(hpo_config: SearchConfig) -> None:
    run_started_wall = time.time()
    run_started_perf = time.perf_counter()
    experiment_config = hpo_config.experiment_config()

    configure_runtime(experiment_config)

    if optuna is None:
        raise RuntimeError("Optuna is not installed")

    configure_cpu_threads(hpo_config.cpu_threads)
    write_json(hpo_config.output_dir / "hpo_config.json", hpo_config.serialisable())
    if hpo_config.selection_filename is not None:
        write_json(hpo_config.output_dir / "search_space.json", hpo_config.search_space)
    write_json(hpo_config.output_dir / "environment.json", environment_summary())

    development, screen_splits, confirmation_splits = prepare_hpo_data(
        hpo_config, experiment_config
    )

    preloaded = (
        preload_images(development, experiment_config, hpo_config.output_dir)
        if hpo_config.preload_images
        else None
    )

    total_seconds = hpo_config.time_budget_hours * 3600.0
    search_deadline = run_started_wall + total_seconds * hpo_config.search_budget_fraction
    total_deadline = run_started_wall + total_seconds
    remaining_search_seconds = max(1.0, search_deadline - time.time())

    LOGGER.info(
        "Time budget: %.2f total hours; %.2f hours currently remain for screening",
        hpo_config.time_budget_hours,
        remaining_search_seconds / 3600.0,
    )

    study = load_or_create_study(hpo_config)

    _run_screening(
        study,
        development,
        screen_splits,
        experiment_config,
        hpo_config,
        preloaded,
        search_deadline,
        remaining_search_seconds,
    )

    screen_analysis = save_screen_analysis(
        study,
        hpo_config.output_dir,
        hpo_config.search_space,
        hpo_config.trial_attributes,
    )

    confirmation_summary, selected = _confirm_candidates(
        study,
        screen_analysis,
        development,
        confirmation_splits,
        experiment_config,
        hpo_config,
        preloaded,
        total_deadline,
    )

    elapsed = time.perf_counter() - run_started_perf
    _save_run_status(
        hpo_config, screen_analysis, confirmation_summary, selected, elapsed
    )

    LOGGER.info("HPO finished in %.2f hours", elapsed / 3600.0)
