from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from modules.config import Config


SEARCH_SPACE = {
    "head_learning_rate": {"type": "float_log", "low": 1e-4, "high": 3e-3},
    "fine_tune_learning_rate": {"type": "float_log", "low": 1e-6, "high": 5e-5},
    "dense_units": {"type": "categorical", "choices": [64, 128, 256]},
    "head_dropout_1": {
        "type": "float_step",
        "low": 0.15,
        "high": 0.55,
        "step": 0.05,
    },
    "head_dropout_2": {
        "type": "float_step",
        "low": 0.05,
        "high": 0.40,
        "step": 0.05,
    },
    "l2_strength": {"type": "float_log", "low": 1e-6, "high": 1e-3},
    "fine_tune_last_layers": {
        "type": "categorical",
        "choices": [20, 40, 60, 80, 120],
    },
    "rotation": {"type": "categorical", "choices": [0.0, 0.015, 0.025, 0.04]},
    "translation": {"type": "categorical", "choices": [0.0, 0.02, 0.04, 0.06]},
    "zoom": {"type": "categorical", "choices": [0.0, 0.04, 0.08, 0.12]},
    "contrast": {"type": "categorical", "choices": [0.0, 0.05, 0.10, 0.15]},
    "label_smoothing": {"type": "categorical", "choices": [0.0, 0.02, 0.05, 0.10]},
    "head_activation": {"type": "categorical", "choices": ["relu", "swish"]},
    "pooling": {"type": "categorical", "choices": ["average", "maximum"]},
}


@dataclass(frozen=True)
class TransferSearchSpec:
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    dense_units: int = 128
    head_dropout_1: float = 0.40
    head_dropout_2: float = 0.25
    l2_strength: float = 1e-4
    fine_tune_last_layers: int = 40
    rotation: float = 0.025
    translation: float = 0.04
    zoom: float = 0.08
    contrast: float = 0.10
    label_smoothing: float = 0.0
    head_activation: str = "swish"
    pooling: str = "average"

    def serialisable(self) -> dict[str, Any]:
        return asdict(self)


REFERENCE_SPEC = TransferSearchSpec()


@dataclass(frozen=True)
class HPOConfig:
    search_space: ClassVar[dict[str, Any]] = SEARCH_SPACE
    reference_spec: ClassVar[TransferSearchSpec] = REFERENCE_SPEC
    study_name: ClassVar[str] = "cbis_ddsm_efficientnetb0_cpu_hpo"
    study_type: ClassVar[str] = "exploratory transfer-learning hyperparameter optimisation"
    model_family: ClassVar[str] = "ImageNet EfficientNetB0"
    trial_attributes: ClassVar[tuple[str, ...]] = ()
    selection_filename: ClassVar[str | None] = None
    epoch_fields: ClassVar[tuple[str, ...]] = (
        "screen_head_epochs",
        "screen_finetune_epochs",
        "confirmation_head_epochs",
        "confirmation_finetune_epochs",
    )

    dataset_root: Path
    output_dir: Path
    program_name: str
    image_size: int = 224
    batch_size: int = 16
    seed: int = 42
    time_budget_hours: float = 46.0
    search_budget_fraction: float = 0.65
    max_trials: int = 60
    startup_trials: int = 8
    screen_folds: int = 2
    confirmation_folds: int = 3
    confirmation_top_k: int = 2
    screen_head_epochs: int = 6
    screen_finetune_epochs: int = 8
    confirmation_head_epochs: int = 12
    confirmation_finetune_epochs: int = 18
    screen_patience: int = 3
    confirmation_patience: int = 5
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
        if confirmation:
            return {
                "head_epochs": self.confirmation_head_epochs,
                "finetune_epochs": self.confirmation_finetune_epochs,
            }

        return {
            "head_epochs": self.screen_head_epochs,
            "finetune_epochs": self.screen_finetune_epochs,
        }

    def experiment_config(self) -> Config:
        return Config(
            dataset_root=self.dataset_root,
            output_dir=self.output_dir,
            program_name=self.program_name,
            models=("transfer",),
            lesion_type=self.lesion_type,
            image_size=self.image_size,
            batch_size=self.batch_size,
            folds=self.screen_folds,
            seed=self.seed,
            max_epochs_transfer_head=self.screen_head_epochs,
            max_epochs_transfer_finetune=self.screen_finetune_epochs,
            early_stopping_patience=self.screen_patience,
            fine_tune_last_layers=REFERENCE_SPEC.fine_tune_last_layers,
            calibration_fraction=self.calibration_fraction,
            include_benign_without_callback=self.include_benign_without_callback,
            combined_patient_overlap_policy=self.combined_patient_overlap_policy,
            dataset_cache="none" if self.preload_images else "memory",
            bootstrap_repeats=200,
        )


def _build_parser(config_type: type = HPOConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--time-budget-hours", type=float, default=46.0)
    parser.add_argument("--search-budget-fraction", type=float, default=0.65)
    parser.add_argument("--max-trials", type=int, default=config_type.max_trials)
    parser.add_argument("--startup-trials", type=int, default=8)

    parser.add_argument("--screen-folds", type=int, default=2)
    parser.add_argument("--confirmation-folds", type=int, default=config_type.confirmation_folds)
    parser.add_argument("--confirmation-top-k", type=int, default=2)

    for name in config_type.epoch_fields:
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(config_type, name)
        )

    parser.add_argument("--screen-patience", type=int, default=config_type.screen_patience)
    parser.add_argument("--confirmation-patience", type=int, default=config_type.confirmation_patience)

    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lesion-type", choices=("both", "mass", "calcification"), default="both"
    )
    parser.add_argument(
        "--combined-patient-overlap-policy",
        choices=("error", "exclude-train"),
        default="exclude-train",
    )

    parser.add_argument("--include-benign-without-callback", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")

    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_type: type = HPOConfig,
) -> None:
    if args.image_size < 128:
        parser.error("--image-size < 128")
    if args.batch_size < 1:
        parser.error("--batch-size < 1")

    if args.time_budget_hours < 1:
        parser.error("--time-budget-hours < 1")
    if not 0.40 <= args.search_budget_fraction <= 0.85:
        parser.error("--search-budget-fraction out of range")
    if args.max_trials < 3:
        parser.error("--max-trials < 3")
    if not 2 <= args.startup_trials <= args.max_trials:
        parser.error("--startup-trials out of range")

    if args.screen_folds < 2 or args.confirmation_folds < 2:
        parser.error("Fold count < 2")
    if not 1 <= args.confirmation_top_k <= 3:
        parser.error("--confirmation-top-k out of range")

    epoch_values = (
        *(getattr(args, name) for name in config_type.epoch_fields),
        args.screen_patience,
        args.confirmation_patience,
    )
    if any(value < 1 for value in epoch_values):
        parser.error("Epochs or patience < 1")

    if not 0.05 <= args.calibration_fraction <= 0.30:
        parser.error("--calibration-fraction out of range")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads < 0")

    if args.combined_patient_overlap_policy == "exclude-train" and args.lesion_type != "both":
        parser.error("Invalid overlap policy for lesion type")


def _config_from_args(
    args: argparse.Namespace,
    program_name: str,
    config_type: type = HPOConfig,
) -> HPOConfig:
    output_root = args.output_dir.resolve()
    output_dir = output_root if output_root.name == program_name else output_root / program_name

    values = vars(args).copy()
    values.update(
        dataset_root=args.dataset_root.resolve(),
        output_dir=output_dir,
        program_name=program_name,
    )

    return config_type(**values)


def parse_hpo_args(program_name: str, config_type: type = HPOConfig) -> HPOConfig:
    parser = _build_parser(config_type)
    args = parser.parse_args()

    _validate_args(parser, args, config_type)

    return _config_from_args(args, program_name, config_type)


def suggest_spec(
    trial: Any,
    reference_spec: Any = REFERENCE_SPEC,
    search_space: dict[str, Any] = SEARCH_SPACE,
) -> Any:
    parameters: dict[str, Any] = {}

    for name in type(reference_spec).__dataclass_fields__:
        settings = search_space[name]
        parameter_type = settings["type"]
        if parameter_type == "categorical":
            value = trial.suggest_categorical(name, settings["choices"])
        else:
            sampling_options = (
                {"log": True}
                if parameter_type == "float_log"
                else {"step": settings["step"]}
            )
            value = trial.suggest_float(
                name, settings["low"], settings["high"], **sampling_options
            )

        parameters[name] = value

    return type(reference_spec)(**parameters)


def spec_from_mapping(
    values: dict[str, Any], spec_type: type = TransferSearchSpec
) -> Any:
    expected = set(spec_type.__dataclass_fields__)
    missing = expected.difference(values)
    if missing:
        raise ValueError(f"Missing parameters: {sorted(missing)}")

    return spec_type(**{name: values[name] for name in expected})
