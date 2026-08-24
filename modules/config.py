from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    output_dir: Path
    program_name: str
    models: tuple[str, ...] = ("baseline", "regularized", "transfer")
    lesion_type: str = "both"
    image_size: int = 224
    batch_size: int = 16
    folds: int = 5
    seed: int = 42
    max_epochs_baseline: int = 40
    max_epochs_regularized: int = 50
    max_epochs_transfer_head: int = 15
    max_epochs_transfer_finetune: int = 20
    early_stopping_patience: int = 7
    fine_tune_last_layers: int = 40
    target_sensitivity: float = 0.90
    bootstrap_repeats: int = 2000
    calibration_fraction: float = 0.15
    include_benign_without_callback: bool = False
    combined_patient_overlap_policy: str = "error"
    dataset_cache: str = "none"
    audit_only: bool = False

    def serialisable(self) -> dict[str, object]:
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        result["output_dir"] = str(self.output_dir)
        return result


def parse_experiment_args(program_name: str) -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("baseline", "regularized", "transfer"),
        default=("baseline", "regularized", "transfer"),
    )
    parser.add_argument(
        "--lesion-type", choices=("both", "mass", "calcification"), default="both"
    )
    parser.add_argument(
        "--combined-patient-overlap-policy",
        choices=("error", "exclude-train"),
        default="error",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs-baseline", type=int, default=40)
    parser.add_argument("--max-epochs-regularized", type=int, default=50)
    parser.add_argument("--max-epochs-transfer-head", type=int, default=15)
    parser.add_argument("--max-epochs-transfer-finetune", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--fine-tune-last-layers", type=int, default=40)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--include-benign-without-callback", action="store_true")
    parser.add_argument(
        "--dataset-cache",
        choices=("none", "memory"),
        default="none",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.target_sensitivity < 1.0:
        parser.error("--target-sensitivity out of range")
    if args.folds < 2:
        parser.error("--folds < 2")
    if args.bootstrap_repeats < 200:
        parser.error("--bootstrap-repeats < 200")
    if not 0.05 <= args.calibration_fraction <= 0.30:
        parser.error("--calibration-fraction out of range")
    if args.image_size < 64:
        parser.error("--image-size < 64")
    if args.batch_size < 1:
        parser.error("--batch-size < 1")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience < 1")
    if args.fine_tune_last_layers < 1:
        parser.error("--fine-tune-last-layers < 1")
    if (
        args.combined_patient_overlap_policy == "exclude-train"
        and args.lesion_type != "both"
    ):
        parser.error("Invalid overlap policy for lesion type")

    output_root = args.output_dir.resolve()
    versioned_output = (
        output_root if output_root.name == program_name else output_root / program_name
    )
    return Config(
        dataset_root=args.dataset_root.resolve(),
        output_dir=versioned_output,
        program_name=program_name,
        models=tuple(args.models),
        lesion_type=args.lesion_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        folds=args.folds,
        seed=args.seed,
        max_epochs_baseline=args.max_epochs_baseline,
        max_epochs_regularized=args.max_epochs_regularized,
        max_epochs_transfer_head=args.max_epochs_transfer_head,
        max_epochs_transfer_finetune=args.max_epochs_transfer_finetune,
        early_stopping_patience=args.early_stopping_patience,
        fine_tune_last_layers=args.fine_tune_last_layers,
        target_sensitivity=args.target_sensitivity,
        bootstrap_repeats=args.bootstrap_repeats,
        calibration_fraction=args.calibration_fraction,
        include_benign_without_callback=args.include_benign_without_callback,
        combined_patient_overlap_policy=args.combined_patient_overlap_policy,
        dataset_cache=args.dataset_cache,
        audit_only=args.audit_only,
    )
