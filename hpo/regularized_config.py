from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from modules.config import Config

from .config import HPOConfig, TransferSearchSpec


SEARCH_SPACE = {
    "learning_rate": {"type": "float_log", "low": 3e-5, "high": 3e-3},
    "initial_filters": {"type": "categorical", "choices": [16, 24, 32]},
    "block_count": {"type": "categorical", "choices": [3, 4, 5]},
    "dense_units": {"type": "categorical", "choices": [64, 128, 256]},
    "spatial_dropout": {"type": "categorical", "choices": [0.0, 0.05, 0.10, 0.20, 0.30]},
    "head_dropout": {"type": "float_step", "low": 0.10, "high": 0.60, "step": 0.05},
    "l2_strength": {"type": "float_log", "low": 1e-6, "high": 1e-3},
    "use_batch_norm": {"type": "categorical", "choices": [True, False]},
    "rotation": {"type": "categorical", "choices": [0.0, 0.015, 0.025, 0.04]},
    "translation": {"type": "categorical", "choices": [0.0, 0.02, 0.04, 0.06]},
    "zoom": {"type": "categorical", "choices": [0.0, 0.04, 0.08, 0.12]},
    "contrast": {"type": "categorical", "choices": [0.0, 0.05, 0.10, 0.15]},
    "label_smoothing": {"type": "categorical", "choices": [0.0, 0.02, 0.05, 0.10]},
    "activation": {"type": "categorical", "choices": ["relu", "swish"]},
    "pooling": {"type": "categorical", "choices": ["average", "maximum"]},
}


@dataclass(frozen=True)
class RegularizedSearchSpec:
    learning_rate: float = 3e-4
    l2_strength: float = 1e-4
    spatial_dropout: float = 0.10
    head_dropout: float = 0.40
    use_batch_norm: bool = True
    initial_filters: int = 32
    block_count: int = 4
    dense_units: int = 128
    activation: str = "relu"
    pooling: str = "average"
    rotation: float = 0.025
    translation: float = 0.04
    zoom: float = 0.08
    contrast: float = 0.10
    label_smoothing: float = 0.0

    def serialisable(self) -> dict[str, Any]:
        return asdict(self)


REFERENCE_SPEC = RegularizedSearchSpec()


@dataclass(frozen=True)
class RegularizedHPOConfig:
    model_variant: ClassVar[str] = "regularized"
    search_space: ClassVar[dict[str, Any]] = SEARCH_SPACE
    reference_spec: ClassVar[RegularizedSearchSpec] = REFERENCE_SPEC
    study_name: ClassVar[str] = "cbis_ddsm_regularized_cnn_cpu_hpo"
    study_type: ClassVar[str] = "exploratory regularized scratch-CNN hyperparameter optimisation"
    model_family: ClassVar[str] = "regularized scratch CNN"
    trial_attributes: ClassVar[tuple[str, ...]] = (
        "median_selected_epoch",
        "mean_fold_seconds",
        "trainable_parameter_count",
    )
    selection_filename: ClassVar[str | None] = "selected_regularized_parameters.json"
    epoch_fields: ClassVar[tuple[str, ...]] = ("screen_epochs", "confirmation_epochs")

    dataset_root: Path
    output_dir: Path
    program_name: str
    image_size: int = 224
    batch_size: int = 16
    seed: int = 42
    time_budget_hours: float = 46.0
    search_budget_fraction: float = 0.65
    max_trials: int = 70
    startup_trials: int = 8
    screen_folds: int = 2
    confirmation_folds: int = 5
    confirmation_top_k: int = 2
    screen_epochs: int = 20
    confirmation_epochs: int = 40
    screen_patience: int = 4
    confirmation_patience: int = 7
    calibration_fraction: float = 0.15
    cpu_threads: int = 0
    preload_images: bool = True
    lesion_type: str = "both"
    combined_patient_overlap_policy: str = "exclude-train"
    include_benign_without_callback: bool = False
    skip_confirmation: bool = False

    def serialisable(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        result["output_dir"] = str(self.output_dir)

        return result

    def training_epochs(self, confirmation: bool = False) -> dict[str, int]:
        epochs = self.confirmation_epochs if confirmation else self.screen_epochs

        return {"epochs": epochs}

    def experiment_config(self) -> Config:
        return Config(
            dataset_root=self.dataset_root,
            output_dir=self.output_dir,
            program_name=self.program_name,
            models=("regularized",),
            lesion_type=self.lesion_type,
            image_size=self.image_size,
            batch_size=self.batch_size,
            folds=self.screen_folds,
            seed=self.seed,
            max_epochs_regularized=self.screen_epochs,
            early_stopping_patience=self.screen_patience,
            calibration_fraction=self.calibration_fraction,
            include_benign_without_callback=self.include_benign_without_callback,
            combined_patient_overlap_policy=self.combined_patient_overlap_policy,
            dataset_cache="none" if self.preload_images else "memory",
            bootstrap_repeats=200,
        )


SearchConfig = HPOConfig | RegularizedHPOConfig
SearchSpec = TransferSearchSpec | RegularizedSearchSpec
