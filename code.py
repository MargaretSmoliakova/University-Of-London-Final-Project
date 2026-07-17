from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from PIL import Image

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from sklearn.model_selection import StratifiedGroupKFold


LOGGER = logging.getLogger("cbis_ddsm")

PATHOLOGY_TO_LABEL = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
METADATA_FILES = {
    "mass": (
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
    ),
    "calcification": (
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ),
}
CASE_ID_PATTERN = re.compile(
    r"(Mass|Calc)-(Training|Test)_P_\d+_(LEFT|RIGHT)_(CC|MLO)_\d+",
    re.IGNORECASE,
)
UID_COMPONENT_PATTERN = re.compile(r"^\d+(?:\.\d+){3,}$")


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    output_dir: Path
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
    include_benign_without_callback: bool = False
    combined_patient_overlap_policy: str = "error"
    dataset_cache: str = "none"
    audit_only: bool = False


@dataclass(frozen=True)
class DicomInfo:
    path: str
    rows: int
    columns: int
    bits_stored: int
    sop_instance_uid: str
    series_instance_uid: str
    mask_like: bool

    @property
    def area(self) -> int:
        return self.rows * self.columns


def parse_args() -> Config:
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
    parser.add_argument("--include-benign-without-callback", action="store_true")
    parser.add_argument(
        "--dataset-cache",
        choices=("none", "memory"),
        default="none",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if (
        args.combined_patient_overlap_policy == "exclude-train"
        and args.lesion_type != "both"
    ):
        parser.error(
            "--combined-patient-overlap-policy exclude-train is only valid with "
            "--lesion-type both"
        )

    output_root = args.output_dir.resolve()
    script_output_folder = Path(__file__).stem
    versioned_output_dir = (
        output_root
        if output_root.name == script_output_folder
        else output_root / script_output_folder
    )

    return Config(
        dataset_root=args.dataset_root.resolve(),
        output_dir=versioned_output_dir,
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
        include_benign_without_callback=args.include_benign_without_callback,
        combined_patient_overlap_policy=args.combined_patient_overlap_policy,
        dataset_cache=args.dataset_cache,
        audit_only=args.audit_only,
    )


def configure_runtime(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.output_dir / "experiment.log", encoding="utf-8"),
        ],
    )
    LOGGER.info(
        "All logs and experiment artifacts for %s are stored under %s",
        Path(__file__).name,
        config.output_dir,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    try:
        tf.config.experimental.enable_op_determinism()
        LOGGER.info("TensorFlow deterministic operations enabled")
    except Exception as exc:
        LOGGER.warning("Could not enable complete TensorFlow determinism: %s", exc)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pixel_array(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if path.suffix.lower() == ".dcm":
        dicom = pydicom.dcmread(str(path))
        pixels = dicom.pixel_array

        pixels = apply_modality_lut(pixels, dicom)
        pixels = apply_voi_lut(pixels, dicom).astype(np.float32)
        if str(dicom.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = float(np.nanmax(pixels)) - pixels
    else:
        pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got {pixels.shape} for {path}"
        )
    if not np.isfinite(pixels).all():
        raise ValueError(f"Non-finite pixels found in {path}")
    return pixels.astype(np.float32, copy=False)


def percentile_window_to_255(pixels: np.ndarray) -> np.ndarray:
    foreground = pixels[pixels > np.min(pixels)]
    source = foreground if foreground.size >= 100 else pixels.ravel()
    low, high = np.percentile(source, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(pixels)), float(np.max(pixels))
    if high <= low:
        raise ValueError("Image has no usable intensity range")
    clipped = np.clip(pixels, low, high)
    return ((clipped - low) * (255.0 / (high - low))).astype(np.float32)


def sampled_unique_count(pixels: np.ndarray, maximum_samples: int = 200_000) -> int:
    flattened = pixels.ravel()
    if flattened.size > maximum_samples:
        stride = math.ceil(flattened.size / maximum_samples)
        flattened = flattened[::stride]
    return int(np.unique(flattened).size)


@lru_cache(maxsize=50_000)
def inspect_dicom(path_string: str) -> DicomInfo:
    path = Path(path_string)
    header = pydicom.dcmread(str(path), stop_before_pixels=True)
    pixels = read_pixel_array(path)
    unique_count = sampled_unique_count(pixels)

    mask_like = unique_count <= 32
    return DicomInfo(
        path=str(path.resolve()),
        rows=int(getattr(header, "Rows", pixels.shape[0])),
        columns=int(getattr(header, "Columns", pixels.shape[1])),
        bits_stored=int(getattr(header, "BitsStored", 0)),
        sop_instance_uid=str(getattr(header, "SOPInstanceUID", "")),
        series_instance_uid=str(getattr(header, "SeriesInstanceUID", "")),
        mask_like=mask_like,
    )


class ImageResolver:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.by_case_id: dict[str, list[Path]] = {}
        dicom_paths = sorted(dataset_root.rglob("*.dcm"))
        if not dicom_paths:
            raise FileNotFoundError(f"No DICOM files found under {dataset_root}")
        for path in dicom_paths:
            match = CASE_ID_PATTERN.search(path.as_posix())
            if match:
                self.by_case_id.setdefault(match.group(0).lower(), []).append(
                    path.resolve()
                )
        LOGGER.info(
            "Indexed %d DICOM files across %d case identifiers",
            len(dicom_paths),
            len(self.by_case_id),
        )

    @staticmethod
    def _uid_components(reference: str) -> set[str]:
        return {
            component
            for component in reference.replace("\\", "/").split("/")
            if UID_COMPONENT_PATTERN.match(component)
        }

    def _direct_candidates(self, reference: str) -> list[Path]:
        normalised = reference.replace("\\", "/").lstrip("/")
        candidates: list[Path] = []
        direct = self.dataset_root / normalised
        if direct.is_file():
            candidates.append(direct.resolve())
        elif direct.is_dir():
            candidates.extend(sorted(path.resolve() for path in direct.rglob("*.dcm")))

        match = CASE_ID_PATTERN.search(normalised)
        if match:
            suffix_candidate = self.dataset_root / normalised[match.start() :]
            if suffix_candidate.is_file():
                candidates.append(suffix_candidate.resolve())
            elif suffix_candidate.is_dir():
                candidates.extend(
                    sorted(path.resolve() for path in suffix_candidate.rglob("*.dcm"))
                )
        return list(dict.fromkeys(candidates))

    def resolve_crop(self, reference: Any) -> dict[str, Any]:
        if pd.isna(reference):
            return {"image_path": None, "resolution_method": "missing_crop_reference"}
        reference_string = str(reference).strip()
        match = CASE_ID_PATTERN.search(reference_string.replace("\\", "/"))
        case_id = match.group(0).lower() if match else ""
        uid_components = self._uid_components(reference_string)

        direct_candidates = self._direct_candidates(reference_string)
        case_candidates = self.by_case_id.get(case_id, []) if case_id else []
        candidates = list(dict.fromkeys(direct_candidates + case_candidates))
        if not candidates:
            return {
                "image_path": None,
                "resolution_method": "unresolved",
                "case_id": case_id,
            }

        inspected: list[DicomInfo] = []
        errors: list[str] = []
        for candidate in candidates:
            try:
                inspected.append(inspect_dicom(str(candidate)))
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        tissue = [info for info in inspected if not info.mask_like]
        if not tissue:
            return {
                "image_path": None,
                "resolution_method": "only_mask_or_unreadable_candidates",
                "case_id": case_id,
                "resolution_errors": " | ".join(errors[:3]),
            }

        direct_set = {str(path) for path in direct_candidates}

        def rank(info: DicomInfo) -> tuple[int, int, int, str]:
            path_parts = set(Path(info.path).parts)
            uid_match = bool(uid_components.intersection(path_parts))
            return (
                info.area,
                0 if info.path in direct_set else 1,
                0 if uid_match else 1,
                info.path,
            )

        selected = min(tissue, key=rank)
        method = (
            "exact_or_direct_crop"
            if selected.path in direct_set
            else "case_id_ranked_crop"
        )
        return {
            "image_path": selected.path,
            "resolution_method": method,
            "case_id": case_id,
            "rows": selected.rows,
            "columns": selected.columns,
            "bits_stored": selected.bits_stored,
            "sop_instance_uid": selected.sop_instance_uid,
            "series_instance_uid": selected.series_instance_uid,
            "mask_like": selected.mask_like,
            "resolution_errors": " | ".join(errors[:3]),
        }


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def load_one_metadata(
    dataset_root: Path,
    filename: str,
    lesion_type: str,
    official_split: str,
    include_benign_without_callback: bool,
) -> pd.DataFrame:
    path = dataset_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = normalise_columns(pd.read_csv(path))
    required = {"patient_id", "pathology", "cropped image file path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")

    metadata["pathology_original"] = (
        metadata["pathology"].astype(str).str.upper().str.strip()
    )
    metadata["label"] = metadata["pathology_original"].map(PATHOLOGY_TO_LABEL)
    metadata = metadata.dropna(subset=["label", "patient_id"]).copy()
    if not include_benign_without_callback:
        metadata = metadata[
            metadata["pathology_original"] != "BENIGN_WITHOUT_CALLBACK"
        ].copy()
    metadata["label"] = metadata["label"].astype(int)
    metadata["patient_id"] = metadata["patient_id"].astype(str).str.upper().str.strip()
    metadata["lesion_type"] = lesion_type
    metadata["official_split"] = official_split
    metadata["source_csv"] = filename
    metadata["source_row"] = metadata.index.astype(int)
    return metadata.reset_index(drop=True)


def load_metadata(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_types = (
        tuple(METADATA_FILES)
        if config.lesion_type == "both"
        else (config.lesion_type,)
    )
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for lesion_type in lesion_types:
        train_file, test_file = METADATA_FILES[lesion_type]
        train_parts.append(
            load_one_metadata(
                config.dataset_root,
                train_file,
                lesion_type,
                "train",
                config.include_benign_without_callback,
            )
        )
        test_parts.append(
            load_one_metadata(
                config.dataset_root,
                test_file,
                lesion_type,
                "test",
                config.include_benign_without_callback,
            )
        )
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def resolve_metadata_images(
    metadata: pd.DataFrame, resolver: ImageResolver
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_records = [
        resolver.resolve_crop(reference)
        for reference in metadata["cropped image file path"]
    ]
    resolution = pd.DataFrame(resolved_records, index=metadata.index)
    combined = pd.concat([metadata, resolution], axis=1)
    unresolved = combined[combined["image_path"].isna()].copy()
    resolved = combined.dropna(subset=["image_path"]).copy().reset_index(drop=True)
    return resolved, unresolved.reset_index(drop=True)


def split_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(metadata)),
        "patients": int(metadata["patient_id"].nunique()),
        "class_counts": metadata["label"].value_counts().sort_index().to_dict(),
        "lesion_type_counts": metadata["lesion_type"].value_counts().to_dict(),
        "pathology_counts": metadata["pathology_original"].value_counts().to_dict(),
        "resolution_methods": metadata["resolution_method"].value_counts().to_dict(),
    }


def exclude_cross_lesion_patient_overlap_from_train(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    unresolved_train: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_patients = set(train_metadata["patient_id"])
    test_patients = set(test_metadata["patient_id"])
    patient_overlap = sorted(train_patients.intersection(test_patients))

    patient_details: list[dict[str, Any]] = []
    same_lesion_overlap: list[str] = []
    for patient_id in patient_overlap:
        train_lesion_types = sorted(
            train_metadata.loc[
                train_metadata["patient_id"] == patient_id, "lesion_type"
            ].unique()
        )
        test_lesion_types = sorted(
            test_metadata.loc[
                test_metadata["patient_id"] == patient_id, "lesion_type"
            ].unique()
        )
        common_lesion_types = sorted(
            set(train_lesion_types).intersection(test_lesion_types)
        )
        if common_lesion_types:
            same_lesion_overlap.append(patient_id)
        patient_details.append(
            {
                "patient_id": patient_id,
                "train_lesion_types": train_lesion_types,
                "test_lesion_types": test_lesion_types,
                "same_lesion_types_across_splits": common_lesion_types,
            }
        )

    excluded_resolved = train_metadata[
        train_metadata["patient_id"].isin(patient_overlap)
    ].copy()
    excluded_unresolved = unresolved_train[
        unresolved_train["patient_id"].isin(patient_overlap)
    ].copy()

    excluded_parts: list[pd.DataFrame] = []
    for resolution_status, frame in (
        ("resolved", excluded_resolved),
        ("unresolved", excluded_unresolved),
    ):
        part = frame.copy()
        part.insert(0, "resolution_status", resolution_status)
        excluded_parts.append(part)
    excluded_records = pd.concat(excluded_parts, ignore_index=True, sort=False)
    excluded_records.to_csv(
        output_dir / "excluded_train_patient_overlap.csv", index=False
    )

    report: dict[str, Any] = {
        "policy": "official-test-priority: exclude overlapping patients from train",
        "reason": (
            "The official mass and calcification partitions are separate; their "
            "combination is not globally patient-disjoint."
        ),
        "official_test_changed": False,
        "overlapping_patient_count": len(patient_overlap),
        "overlapping_patient_ids": patient_overlap,
        "patient_lesion_type_details": patient_details,
        "same_lesion_overlap_count": len(same_lesion_overlap),
        "same_lesion_overlap_patient_ids": same_lesion_overlap,
        "train_before": split_summary(train_metadata),
        "excluded_train_resolved_rows": int(len(excluded_resolved)),
        "excluded_train_unresolved_rows": int(len(excluded_unresolved)),
        "excluded_train_class_counts": (
            excluded_resolved["label"].value_counts().sort_index().to_dict()
        ),
        "excluded_train_lesion_type_counts": (
            excluded_resolved["lesion_type"].value_counts().to_dict()
        ),
    }

    if same_lesion_overlap:
        report["status"] = "failed: same-lesion official split overlap detected"
        write_json(output_dir / "combined_patient_overlap_resolution.json", report)
        raise RuntimeError(
            "The exclude-train policy only permits cross-lesion overlap caused by "
            "combining the separate mass and calcification partitions. Same-lesion "
            f"patient overlap remains for {len(same_lesion_overlap)} patients."
        )

    filtered_train = train_metadata[
        ~train_metadata["patient_id"].isin(patient_overlap)
    ].copy().reset_index(drop=True)
    filtered_unresolved_train = unresolved_train[
        ~unresolved_train["patient_id"].isin(patient_overlap)
    ].copy().reset_index(drop=True)

    remaining_overlap = set(filtered_train["patient_id"]).intersection(test_patients)
    if remaining_overlap:
        raise AssertionError("Patient overlap remained after train-only exclusion")

    report["train_after"] = split_summary(filtered_train)
    report["status"] = "applied"
    write_json(output_dir / "combined_patient_overlap_resolution.json", report)
    LOGGER.warning(
        "Excluded %d resolved and %d unresolved train rows from %d patients to "
        "keep the combined official test set patient-disjoint",
        len(excluded_resolved),
        len(excluded_unresolved),
        len(patient_overlap),
    )
    return filtered_train, filtered_unresolved_train, report


def audit_dataset(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    unresolved_train: pd.DataFrame,
    unresolved_test: pd.DataFrame,
    output_dir: Path,
    official_split_adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_patients = set(train_metadata["patient_id"])
    test_patients = set(test_metadata["patient_id"])
    patient_overlap = sorted(train_patients.intersection(test_patients))
    train_paths = set(train_metadata["image_path"])
    test_paths = set(test_metadata["image_path"])
    path_overlap = sorted(train_paths.intersection(test_paths))
    train_sops = {uid for uid in train_metadata["sop_instance_uid"] if uid}
    test_sops = {uid for uid in test_metadata["sop_instance_uid"] if uid}
    sop_overlap = sorted(train_sops.intersection(test_sops))

    combined = pd.concat([train_metadata, test_metadata], ignore_index=True)
    label_conflicts = (
        combined.groupby("image_path")["label"].nunique().loc[lambda values: values > 1]
    )
    duplicate_path_counts = (
        combined.groupby(["official_split", "image_path"])
        .size()
        .loc[lambda values: values > 1]
    )
    duplicate_rows = int((duplicate_path_counts - 1).sum())


    train_hashes = {file_sha256(path): path for path in sorted(train_paths)}
    test_hashes = {file_sha256(path): path for path in sorted(test_paths)}
    hash_overlap = sorted(set(train_hashes).intersection(test_hashes))

    audit = {
        "train": split_summary(train_metadata),
        "test": split_summary(test_metadata),
        "unresolved_train_rows": int(len(unresolved_train)),
        "unresolved_test_rows": int(len(unresolved_test)),
        "patient_overlap_count": len(patient_overlap),
        "patient_overlap_examples": patient_overlap[:20],
        "path_overlap_count": len(path_overlap),
        "path_overlap_examples": path_overlap[:10],
        "sop_uid_overlap_count": len(sop_overlap),
        "sop_uid_overlap_examples": sop_overlap[:10],
        "byte_identical_cross_split_count": len(hash_overlap),
        "byte_identical_cross_split_hash_examples": hash_overlap[:10],
        "within_split_duplicate_path_rows": duplicate_rows,
        "within_split_duplicate_path_examples": [
            {"split": split, "image_path": path, "row_count": int(count)}
            for (split, path), count in duplicate_path_counts.head(10).items()
        ],
        "image_path_label_conflict_count": int(len(label_conflicts)),
        "image_path_label_conflicts": label_conflicts.index.astype(str).tolist()[:10],
        "official_split_adjustment": official_split_adjustment,
    }
    write_json(output_dir / "data_audit.json", audit)
    unresolved_train.to_csv(output_dir / "unresolved_train.csv", index=False)
    unresolved_test.to_csv(output_dir / "unresolved_test.csv", index=False)
    train_metadata.to_csv(output_dir / "resolved_train_manifest.csv", index=False)
    test_metadata.to_csv(output_dir / "resolved_test_manifest.csv", index=False)

    leakage_failures = {
        "patient overlap": patient_overlap,
        "path overlap": path_overlap,
        "SOP Instance UID overlap": sop_overlap,
        "byte-identical files": hash_overlap,
    }
    present = {name: values for name, values in leakage_failures.items() if values}
    if present:
        raise RuntimeError(
            "Potential official train/test leakage detected; training stopped. "
            + "; ".join(f"{name}={len(values)}" for name, values in present.items())
        )
    if len(label_conflicts):
        raise RuntimeError(
            "Conflicting labels were assigned to the same resolved image"
        )
    if duplicate_rows:
        raise RuntimeError(
            "The same resolved ROI is used by multiple rows inside a split; inspect "
            "data_audit.json instead of allowing duplicated samples to "
            "reweight training"
        )
    if train_metadata["label"].nunique() != 2 or test_metadata["label"].nunique() != 2:
        raise RuntimeError(
            "Both official splits must contain benign and malignant samples"
        )
    return audit


def prepare_experiment_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_train, raw_test = load_metadata(config)
    resolver = ImageResolver(config.dataset_root)
    train_metadata, unresolved_train = resolve_metadata_images(raw_train, resolver)
    test_metadata, unresolved_test = resolve_metadata_images(raw_test, resolver)
    official_split_adjustment = None
    if config.combined_patient_overlap_policy == "exclude-train":
        (
            train_metadata,
            unresolved_train,
            official_split_adjustment,
        ) = exclude_cross_lesion_patient_overlap_from_train(
            train_metadata,
            test_metadata,
            unresolved_train,
            config.output_dir,
        )
    audit = audit_dataset(
        train_metadata,
        test_metadata,
        unresolved_train,
        unresolved_test,
        config.output_dir,
        official_split_adjustment,
    )
    LOGGER.info("Data audit passed: %s", json.dumps(audit, default=json_default))
    return train_metadata, test_metadata


def stratification_labels(metadata: pd.DataFrame) -> np.ndarray:
    return (
        metadata["lesion_type"].astype(str)
        + "_label_"
        + metadata["label"].astype(str)
    ).to_numpy()


def validate_fold_integrity(
    train_fold: pd.DataFrame, validation_fold: pd.DataFrame, fold_number: int
) -> None:
    overlap = set(train_fold["patient_id"]).intersection(validation_fold["patient_id"])
    if overlap:
        raise RuntimeError(
            f"Patient leakage in CV fold {fold_number}: {sorted(overlap)[:10]}"
        )
    if train_fold["label"].nunique() != 2 or validation_fold["label"].nunique() != 2:
        raise RuntimeError(f"Both classes are required in CV fold {fold_number}")


def make_cv_splitter(config: Config) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )


def _numpy_load_image(path_bytes: bytes) -> np.ndarray:
    path = path_bytes.decode("utf-8")
    return percentile_window_to_255(read_pixel_array(path))[..., None]


def load_image_tensor(
    image_path: tf.Tensor, label: tf.Tensor, image_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.numpy_function(_numpy_load_image, [image_path], tf.float32)
    image.set_shape((None, None, 1))
    image = tf.image.resize_with_pad(
        image, image_size, image_size, method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.image.grayscale_to_rgb(image)
    image = tf.ensure_shape(image, (image_size, image_size, 3))
    return image, tf.cast(label, tf.float32)


def make_dataset(
    metadata: pd.DataFrame,
    config: Config,
    training: bool,
    seed_offset: int = 0,
) -> tf.data.Dataset:
    paths = metadata["image_path"].astype(str).to_numpy()
    labels = metadata["label"].astype(np.float32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda path, label: load_image_tensor(path, label, config.image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )


    if config.dataset_cache == "memory":
        estimated_gib = (
            len(metadata) * config.image_size * config.image_size * 3 * 4 / (1024**3)
        )
        LOGGER.info(
            "In-memory image cache enabled for %d rows "
            "(approximately %.2f GiB once full)",
            len(metadata),
            estimated_gib,
        )
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(metadata),
            seed=config.seed + seed_offset,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def augmentation_layers() -> tf.keras.Sequential:

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.025, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.08, 0.08, fill_mode="reflect"),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_only_augmentation",
    )


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )


def build_scratch_cnn(config: Config, regularized: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(
        (config.image_size, config.image_size, 3), name="image_0_255"
    )
    x = inputs
    if regularized:
        x = augmentation_layers()(x)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="cnn_rescale_0_1")(x)
    l2 = tf.keras.regularizers.l2(1e-4) if regularized else None

    for block_index, filters in enumerate((32, 64, 128, 256), start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not regularized,
            kernel_initializer="he_normal",
            kernel_regularizer=l2,
            name=f"block{block_index}_conv",
        )(x)
        if regularized:
            x = tf.keras.layers.BatchNormalization(name=f"block{block_index}_bn")(x)
        x = tf.keras.layers.Activation("relu", name=f"block{block_index}_relu")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"block{block_index}_pool")(x)
        if regularized and block_index >= 3:
            x = tf.keras.layers.SpatialDropout2D(
                0.10, name=f"block{block_index}_drop"
            )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=l2, name="classification_features"
    )(x)
    if regularized:
        x = tf.keras.layers.Dropout(0.40, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignancy_probability"
    )(x)
    model = tf.keras.Model(
        inputs,
        outputs,
        name="regularized_cnn" if regularized else "baseline_cnn",
    )
    compile_binary_model(model, learning_rate=3e-4 if regularized else 1e-3)
    return model


def build_transfer_model(config: Config) -> tf.keras.Model:
    inputs = tf.keras.Input(
        (config.image_size, config.image_size, 3), name="image_0_255"
    )
    x = augmentation_layers()(inputs)


    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.image_size, config.image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(0.40, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(
        128,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="classification_features",
    )(x)
    x = tf.keras.layers.Dropout(0.25, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="malignancy_probability"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    compile_binary_model(model, learning_rate=1e-3)
    return model


def enable_transfer_fine_tuning(model: tf.keras.Model, last_layers: int) -> None:
    nested_models = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()
    ]
    if len(nested_models) != 1:
        raise RuntimeError(
            "Expected one EfficientNet backbone, found "
            f"{[layer.name for layer in nested_models]}"
        )
    backbone = nested_models[0]
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - last_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    compile_binary_model(model, learning_rate=1e-5)


def balanced_class_weights(metadata: pd.DataFrame) -> dict[int, float]:
    counts = metadata["label"].value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError(
            f"Training fold does not contain both classes: {counts.to_dict()}"
        )
    total = float(counts.sum())
    return {int(label): total / (2.0 * float(count)) for label, count in counts.items()}


def flush_python_log_handlers() -> None:
    seen: set[int] = set()
    for logger in (LOGGER, logging.getLogger()):
        for handler in logger.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            handler.flush()
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.FileHandler) and stream is not None:
                try:
                    os.fsync(stream.fileno())
                except (AttributeError, OSError):
                    pass


class EpochTiming(tf.keras.callbacks.Callback):
    def __init__(self, stage_name: str):
        super().__init__()
        self.stage_name = stage_name
        self.started_at = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.started_at = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.perf_counter() - self.started_at
        if logs is not None:
            logs["epoch_seconds"] = elapsed
        metric_text = ", ".join(
            f"{name}={float(value):.6f}"
            for name, value in (logs or {}).items()
            if name != "epoch_seconds" and isinstance(value, (int, float, np.number))
        )
        LOGGER.info(
            "%s epoch %d finished in %.1f seconds | %s",
            self.stage_name,
            epoch + 1,
            elapsed,
            metric_text,
        )
        flush_python_log_handlers()


class ImmediateCSVLogger(tf.keras.callbacks.CSVLogger):
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        super().on_epoch_end(epoch, logs)
        csv_file = getattr(self, "csv_file", None)
        if csv_file is not None:
            csv_file.flush()
            try:
                os.fsync(csv_file.fileno())
            except (AttributeError, OSError):
                pass


def callbacks_for_stage(
    config: Config, stage_dir: Path, stage_name: str
) -> list[tf.keras.callbacks.Callback]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        EpochTiming(stage_name),
        tf.keras.callbacks.ModelCheckpoint(
            stage_dir / "best.weights.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=max(2, config.early_stopping_patience // 2),
            min_lr=1e-7,
        ),
        ImmediateCSVLogger(stage_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def callbacks_for_final_stage(
    stage_dir: Path, stage_name: str
) -> list[tf.keras.callbacks.Callback]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        EpochTiming(stage_name),
        ImmediateCSVLogger(stage_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def best_epoch_and_value(history: tf.keras.callbacks.History) -> tuple[int, float]:
    values = np.asarray(history.history["val_loss"], dtype=float)
    best_index = int(np.nanargmin(values))
    return best_index + 1, float(values[best_index])


def train_one_fold(
    variant: str,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    config: Config,
    fold_dir: Path,
    fold_number: int,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed + fold_number)
    train_dataset = make_dataset(
        train_metadata, config, training=True, seed_offset=fold_number
    )
    validation_dataset = make_dataset(validation_metadata, config, training=False)
    class_weights = balanced_class_weights(train_metadata)

    if variant == "baseline":
        stage_name = f"baseline fold {fold_number}/{config.folds} training"
        LOGGER.info(
            "Starting %s (maximum %d epochs)", stage_name, config.max_epochs_baseline
        )
        model = build_scratch_cnn(config, regularized=False)
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_baseline,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fold_dir / "training", stage_name),
            verbose=2,
        )
        selected_epoch, selected_loss = best_epoch_and_value(history)
        schedule = {
            "selected_epoch": selected_epoch,
            "selected_stage": "single_stage",
            "validation_loss": selected_loss,
        }
    elif variant == "regularized":
        stage_name = f"regularized fold {fold_number}/{config.folds} training"
        LOGGER.info(
            "Starting %s (maximum %d epochs)",
            stage_name,
            config.max_epochs_regularized,
        )
        model = build_scratch_cnn(config, regularized=True)
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_regularized,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fold_dir / "training", stage_name),
            verbose=2,
        )
        selected_epoch, selected_loss = best_epoch_and_value(history)
        schedule = {
            "selected_epoch": selected_epoch,
            "selected_stage": "single_stage",
            "validation_loss": selected_loss,
        }
    elif variant == "transfer":
        model = build_transfer_model(config)
        head_dir = fold_dir / "head_training"
        head_stage_name = f"transfer fold {fold_number}/{config.folds} head training"
        LOGGER.info(
            "Starting %s (maximum %d epochs)",
            head_stage_name,
            config.max_epochs_transfer_head,
        )
        head_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_transfer_head,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, head_dir, head_stage_name),
            verbose=2,
        )
        head_epoch, head_loss = best_epoch_and_value(head_history)
        model.load_weights(head_dir / "best.weights.h5")

        enable_transfer_fine_tuning(model, config.fine_tune_last_layers)
        fine_dir = fold_dir / "fine_tuning"
        fine_stage_name = f"transfer fold {fold_number}/{config.folds} fine-tuning"
        LOGGER.info(
            "Starting %s (maximum %d epochs)",
            fine_stage_name,
            config.max_epochs_transfer_finetune,
        )
        fine_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_transfer_finetune,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fine_dir, fine_stage_name),
            verbose=2,
        )
        fine_epoch, fine_loss = best_epoch_and_value(fine_history)
        if fine_loss <= head_loss:
            model.load_weights(fine_dir / "best.weights.h5")
            selected_stage = "fine_tuning"
            selected_loss = fine_loss
        else:
            model.load_weights(head_dir / "best.weights.h5")
            selected_stage = "head_training"
            selected_loss = head_loss
        schedule = {
            "head_epoch": head_epoch,
            "fine_tune_epoch": fine_epoch,
            "selected_stage": selected_stage,
            "validation_loss": selected_loss,
        }
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    schedule["class_weights"] = class_weights
    return model, schedule


def train_final_model(
    variant: str,
    train_metadata: pd.DataFrame,
    schedule: dict[str, int | bool],
    config: Config,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed + 10_000)
    dataset = make_dataset(train_metadata, config, training=True, seed_offset=10_000)
    class_weights = balanced_class_weights(train_metadata)
    write_json(config.output_dir / variant / "final_class_weights.json", class_weights)

    if variant == "baseline":
        model = build_scratch_cnn(config, regularized=False)
        stage_name = "baseline final training"
        model.fit(
            dataset,
            epochs=schedule["epochs"],
            class_weight=class_weights,
            callbacks=callbacks_for_final_stage(
                config.output_dir / variant / "final_training", stage_name
            ),
            verbose=2,
        )
    elif variant == "regularized":
        model = build_scratch_cnn(config, regularized=True)
        stage_name = "regularized final training"
        model.fit(
            dataset,
            epochs=schedule["epochs"],
            class_weight=class_weights,
            callbacks=callbacks_for_final_stage(
                config.output_dir / variant / "final_training", stage_name
            ),
            verbose=2,
        )
    elif variant == "transfer":
        model = build_transfer_model(config)
        head_stage_name = "transfer final head training"
        model.fit(
            dataset,
            epochs=schedule["head_epochs"],
            class_weight=class_weights,
            callbacks=callbacks_for_final_stage(
                config.output_dir / variant / "final_head_training",
                head_stage_name,
            ),
            verbose=2,
        )
        if schedule["use_fine_tuning"]:
            enable_transfer_fine_tuning(model, config.fine_tune_last_layers)
            fine_stage_name = "transfer final fine-tuning"
            model.fit(
                dataset,
                epochs=schedule["fine_tune_epochs"],
                class_weight=class_weights,
                callbacks=callbacks_for_final_stage(
                    config.output_dir / variant / "final_fine_tuning",
                    fine_stage_name,
                ),
                verbose=2,
            )
    else:
        raise ValueError(variant)

    model_path = config.output_dir / variant / "final_model.keras"
    model.save(model_path)
    return model


def cross_validate_variant(
    variant: str,
    train_metadata: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    variant_dir = config.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    splitter = make_cv_splitter(config)
    strata = stratification_labels(train_metadata)

    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(train_metadata, strata, groups=train_metadata["patient_id"]),
        start=1,
    ):
        fold_train = train_metadata.iloc[train_index].reset_index(drop=True)
        fold_validation = train_metadata.iloc[validation_index].reset_index(drop=True)
        validate_fold_integrity(fold_train, fold_validation, fold_number)
        LOGGER.info(
            "%s fold %d/%d: train=%d, validation=%d",
            variant,
            fold_number,
            config.folds,
            len(fold_train),
            len(fold_validation),
        )
        model, schedule = train_one_fold(
            variant,
            fold_train,
            fold_validation,
            config,
            variant_dir / f"fold_{fold_number}",
            fold_number,
        )
        fold_rows.append(
            {
                "model": variant,
                "fold": fold_number,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "train_patients": fold_train["patient_id"].nunique(),
                "validation_patients": fold_validation["patient_id"].nunique(),
                **schedule,
            }
        )
        schedules.append(schedule)
        del model
        tf.keras.backend.clear_session()

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(variant_dir / "cv_training_summary.csv", index=False)

    if variant == "transfer":
        fine_tuning_selected = sum(
            item["selected_stage"] == "fine_tuning" for item in schedules
        ) >= math.ceil(len(schedules) / 2)
        final_schedule = {
            "head_epochs": max(
                1, int(np.median([item["head_epoch"] for item in schedules]))
            ),
            "fine_tune_epochs": max(
                1, int(np.median([item["fine_tune_epoch"] for item in schedules]))
            ),
            "use_fine_tuning": fine_tuning_selected,
        }
    else:
        final_schedule = {
            "epochs": max(
                1, int(np.median([item["selected_epoch"] for item in schedules]))
            )
        }
    write_json(variant_dir / "final_training_schedule.json", final_schedule)
    return folds_frame, final_schedule


def write_and_log_training_plan(config: Config) -> None:
    plan: list[dict[str, Any]] = []
    for variant in config.models:
        if variant == "baseline":
            cv_fits = config.folds
            cv_max_epochs = config.folds * config.max_epochs_baseline
            description = (
                f"{config.folds} CV fits x at most "
                f"{config.max_epochs_baseline} epochs, then one final fit"
            )
        elif variant == "regularized":
            cv_fits = config.folds
            cv_max_epochs = config.folds * config.max_epochs_regularized
            description = (
                f"{config.folds} CV fits x at most "
                f"{config.max_epochs_regularized} epochs, then one final fit"
            )
        else:
            cv_fits = config.folds * 2
            cv_max_epochs = config.folds * (
                config.max_epochs_transfer_head + config.max_epochs_transfer_finetune
            )
            description = (
                f"{config.folds} folds x (head <= {config.max_epochs_transfer_head} + "
                f"fine-tuning <= {config.max_epochs_transfer_finetune} epochs), "
                "then one final head/fine-tuning fit"
            )
        plan.append(
            {
                "model": variant,
                "description": description,
                "cv_fit_calls": cv_fits,
                "cv_maximum_epochs_before_early_stopping": cv_max_epochs,
            }
        )
        LOGGER.info("Training plan for %s: %s", variant, description)
    write_json(config.output_dir / "training_plan.json", plan)


def run_experiments(config: Config) -> None:
    configure_runtime(config)
    write_json(config.output_dir / "config.json", asdict(config))
    if not config.audit_only:
        write_and_log_training_plan(config)
    train_metadata, _ = prepare_experiment_data(config)
    if config.audit_only:
        LOGGER.info("Audit-only run complete.")
        return

    all_cv_rows: list[pd.DataFrame] = []
    for variant in config.models:
        LOGGER.info("Starting model: %s", variant)
        cv_frame, schedule = cross_validate_variant(variant, train_metadata, config)
        all_cv_rows.append(cv_frame)
        final_model = train_final_model(variant, train_metadata, schedule, config)
        del final_model
        tf.keras.backend.clear_session()

    pd.concat(all_cv_rows, ignore_index=True).to_csv(
        config.output_dir / "all_cv_training_summary.csv", index=False
    )
    LOGGER.info("All requested experiments completed")


run_experiments(parse_args())
