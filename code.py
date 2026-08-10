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
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from PIL import Image

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
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
from sklearn.model_selection import StratifiedGroupKFold


LOGGER = logging.getLogger("cbis_ddsm")

PATHOLOGY_TO_LABEL = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
METADATA_FILES = {
    "mass": ("mass_case_description_train_set.csv", "mass_case_description_test_set.csv"),
    "calcification": ("calc_case_description_train_set.csv", "calc_case_description_test_set.csv"),
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
    target_sensitivity: float = 0.90
    bootstrap_repeats: int = 2000
    calibration_fraction: float = 0.15
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


@dataclass(frozen=True)
class Thresholds:
    default: float
    youden_validation: float
    sensitivity_target_validation: float


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
    if (
        args.combined_patient_overlap_policy == "exclude-train"
        and args.lesion_type != "both"
    ):
        parser.error("Invalid overlap policy for lesion type")

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
        target_sensitivity=args.target_sensitivity,
        bootstrap_repeats=args.bootstrap_repeats,
        calibration_fraction=args.calibration_fraction,
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
        LOGGER.warning("Determinism failed: %s", type(exc).__name__)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


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
        raise ValueError(f"Invalid image shape: {pixels.shape}")
    if not np.isfinite(pixels).all():
        raise ValueError(f"Non-finite pixels: {path.name}")
    return pixels.astype(np.float32, copy=False)


def percentile_window_to_255(pixels: np.ndarray) -> np.ndarray:
    foreground = pixels[pixels > np.min(pixels)]
    source = foreground if foreground.size >= 100 else pixels.ravel()
    low, high = np.percentile(source, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(pixels)), float(np.max(pixels))
    if high <= low:
        raise ValueError("Invalid intensity range")
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageResolver:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.by_case_id: dict[str, list[Path]] = {}
        self.case_ids_by_patient: dict[str, set[str]] = {}
        self.by_series_uid: dict[str, list[Path]] = {}
        dicom_paths = sorted(dataset_root.rglob("*.dcm"))
        if not dicom_paths:
            raise FileNotFoundError("No DICOM files found")
        for path in dicom_paths:
            resolved_path = path.resolve()
            match = CASE_ID_PATTERN.search(path.as_posix())
            if match:
                case_id = match.group(0).lower()
                self.by_case_id.setdefault(case_id, []).append(resolved_path)
                patient_match = re.search(r"p_\d+", case_id, re.IGNORECASE)
                if patient_match:
                    patient_id = patient_match.group(0).lower()
                    self.case_ids_by_patient.setdefault(patient_id, set()).add(case_id)

            uid_path_components = [
                component
                for component in path.parts
                if UID_COMPONENT_PATTERN.fullmatch(component)
            ]
            if uid_path_components:


                series_uid = uid_path_components[-1]
                self.by_series_uid.setdefault(series_uid, []).append(resolved_path)
        LOGGER.info(
            "Indexed %d DICOM files across %d case identifiers and %d series UID directories",
            len(dicom_paths),
            len(self.by_case_id),
            len(self.by_series_uid),
        )

    @staticmethod
    def _uid_components(reference: str) -> set[str]:
        return {
            component
            for component in reference.replace("\\", "/").split("/")
            if UID_COMPONENT_PATTERN.fullmatch(component)
        }

    @staticmethod
    def _series_uid(reference: str) -> str:
        uid_components = [
            component
            for component in reference.replace("\\", "/").split("/")
            if UID_COMPONENT_PATTERN.fullmatch(component)
        ]
        return uid_components[-1] if uid_components else ""

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
                candidates.extend(sorted(path.resolve() for path in suffix_candidate.rglob("*.dcm")))
        return list(dict.fromkeys(candidates))

    def resolve_crop(self, reference: Any) -> dict[str, Any]:
        if pd.isna(reference):
            return {"image_path": None, "resolution_method": "missing_crop_reference"}
        reference_string = str(reference).strip()
        match = CASE_ID_PATTERN.search(reference_string.replace("\\", "/"))
        case_id = match.group(0).lower() if match else ""
        uid_components = self._uid_components(reference_string)
        requested_series_uid = self._series_uid(reference_string)

        direct_candidates = self._direct_candidates(reference_string)
        case_candidates = self.by_case_id.get(case_id, []) if case_id else []
        primary_candidates = list(dict.fromkeys(direct_candidates + case_candidates))
        uid_candidates = (
            self.by_series_uid.get(requested_series_uid, [])
            if requested_series_uid
            else []
        )
        used_series_uid_fallback = not primary_candidates and bool(uid_candidates)
        candidates = (
            list(dict.fromkeys(uid_candidates))
            if used_series_uid_fallback
            else primary_candidates
        )
        if not candidates:
            patient_match = re.search(r"p_\d+", case_id, re.IGNORECASE)
            patient_id = patient_match.group(0).lower() if patient_match else ""
            available_for_patient = sorted(
                self.case_ids_by_patient.get(patient_id, set())
            )
            same_view_prefix = re.sub(r"_\d+$", "", case_id)
            same_view_candidates = [
                available_case_id
                for available_case_id in available_for_patient
                if available_case_id.startswith(same_view_prefix)
            ]
            return {
                "image_path": None,
                "resolution_method": "unresolved",
                "case_id": case_id,
                "requested_series_instance_uid": requested_series_uid,
                "available_case_ids_for_patient": " | ".join(available_for_patient),
                "same_view_case_id_candidates": " | ".join(same_view_candidates),
            }

        inspected: list[DicomInfo] = []
        errors: list[str] = []
        for candidate in candidates:
            try:
                inspected.append(inspect_dicom(str(candidate)))
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        if used_series_uid_fallback:
            inspected = [
                info
                for info in inspected
                if info.series_instance_uid == requested_series_uid
            ]

        tissue = [info for info in inspected if not info.mask_like]
        if not tissue:
            return {
                "image_path": None,
                "resolution_method": (
                    "series_uid_header_mismatch_or_no_tissue"
                    if used_series_uid_fallback
                    else "only_mask_or_unreadable_candidates"
                ),
                "case_id": case_id,
                "requested_series_instance_uid": requested_series_uid,
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
        if used_series_uid_fallback:
            method = "series_uid_ranked_crop"
        else:
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


def load_one_metadata(
    dataset_root: Path,
    filename: str,
    lesion_type: str,
    official_split: str,
    include_benign_without_callback: bool,
) -> pd.DataFrame:
    path = dataset_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {filename}")
    metadata = normalise_columns(pd.read_csv(path))
    required = {"patient_id", "pathology", "cropped image file path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    metadata["pathology_original"] = metadata["pathology"].astype(str).str.upper().str.strip()
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
        report["status"] = "failed: same-lesion overlap"
        write_json(output_dir / "combined_patient_overlap_resolution.json", report)
        raise RuntimeError(f"Same-lesion overlap: {len(same_lesion_overlap)} patients")

    filtered_train = train_metadata[
        ~train_metadata["patient_id"].isin(patient_overlap)
    ].copy().reset_index(drop=True)
    filtered_unresolved_train = unresolved_train[
        ~unresolved_train["patient_id"].isin(patient_overlap)
    ].copy().reset_index(drop=True)

    remaining_overlap = set(filtered_train["patient_id"]).intersection(test_patients)
    if remaining_overlap:
        raise AssertionError("Patient overlap remains")

    report["train_after"] = split_summary(filtered_train)
    report["status"] = "applied"
    write_json(output_dir / "combined_patient_overlap_resolution.json", report)
    LOGGER.warning(
        "Excluded overlapping rows: %d resolved, %d unresolved; %d patients",
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
        combined.groupby(["official_split", "image_path"]).size().loc[lambda values: values > 1]
    )
    duplicate_rows = int((duplicate_path_counts - 1).sum())


    train_hashes = {file_sha256(path): path for path in sorted(train_paths)}
    test_hashes = {file_sha256(path): path for path in sorted(test_paths)}
    hash_overlap = sorted(set(train_hashes).intersection(test_hashes))

    unresolved_summary = {
        "train_rows": int(len(unresolved_train)),
        "test_rows": int(len(unresolved_test)),
        "train_by_lesion_type": (
            unresolved_train["lesion_type"].value_counts().to_dict()
            if len(unresolved_train)
            else {}
        ),
        "test_by_lesion_type": (
            unresolved_test["lesion_type"].value_counts().to_dict()
            if len(unresolved_test)
            else {}
        ),
        "train_by_label": (
            unresolved_train["label"].value_counts().sort_index().to_dict()
            if len(unresolved_train)
            else {}
        ),
        "test_by_label": (
            unresolved_test["label"].value_counts().sort_index().to_dict()
            if len(unresolved_test)
            else {}
        ),
        "train_unique_patients": int(unresolved_train["patient_id"].nunique()),
        "test_unique_patients": int(unresolved_test["patient_id"].nunique()),
        "test_case_id_examples": (
            unresolved_test["case_id"].dropna().astype(str).head(20).tolist()
        ),
    }

    audit = {
        "train": split_summary(train_metadata),
        "test": split_summary(test_metadata),
        "unresolved_train_rows": int(len(unresolved_train)),
        "unresolved_test_rows": int(len(unresolved_test)),
        "unresolved_summary": unresolved_summary,
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

    if len(unresolved_train) or len(unresolved_test):
        raise RuntimeError(f"Unresolved images: train={len(unresolved_train)}, test={len(unresolved_test)}")

    leakage_failures = {
        "patient overlap": patient_overlap,
        "path overlap": path_overlap,
        "SOP Instance UID overlap": sop_overlap,
        "byte-identical files": hash_overlap,
    }
    present = {name: values for name, values in leakage_failures.items() if values}
    if present:
        raise RuntimeError("Train/test leakage")
    if len(label_conflicts):
        raise RuntimeError("Conflicting image labels")
    if duplicate_rows:
        raise RuntimeError("Duplicate image rows")
    if train_metadata["label"].nunique() != 2 or test_metadata["label"].nunique() != 2:
        raise RuntimeError("Class missing from split")
    return audit


def save_visual_samples(metadata: pd.DataFrame, output_dir: Path, count: int = 24) -> None:
    sample_dir = output_dir / "sample_check"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if metadata.empty:
        return
    sample_count = min(count, len(metadata))
    sampled_parts: list[pd.DataFrame] = []
    for _, group in metadata.groupby(["lesion_type", "label"]):
        sampled_parts.append(
            group.sample(n=min(max(1, sample_count // 4), len(group)), random_state=17)
        )
    sampled = pd.concat(sampled_parts, ignore_index=True).head(sample_count)
    for index, row in enumerate(sampled.itertuples(index=False)):
        pixels = percentile_window_to_255(read_pixel_array(row.image_path))
        image = Image.fromarray(pixels.astype(np.uint8)).resize(
            (384, 384), resample=Image.Resampling.BILINEAR
        )
        image.save(
            sample_dir
            / f"{index:02d}_{row.lesion_type}_label-{row.label}_{row.patient_id}.png"
        )
    LOGGER.info("Saved %d images for mandatory manual ROI inspection", len(sampled))


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
            "In-memory image cache enabled for %d rows (approximately %.2f GiB once full)",
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
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def build_scratch_cnn(config: Config, regularized: bool) -> tf.keras.Model:
    inputs = tf.keras.Input((config.image_size, config.image_size, 3), name="image_0_255")
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
            x = tf.keras.layers.SpatialDropout2D(0.10, name=f"block{block_index}_drop")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(
        128, activation="relu", kernel_regularizer=l2, name="classification_features"
    )(x)
    if regularized:
        x = tf.keras.layers.Dropout(0.40, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy_probability")(x)
    model = tf.keras.Model(
        inputs,
        outputs,
        name="regularized_cnn" if regularized else "baseline_cnn",
    )
    compile_binary_model(model, learning_rate=3e-4 if regularized else 1e-3)
    return model


def build_transfer_model(config: Config) -> tf.keras.Model:
    inputs = tf.keras.Input((config.image_size, config.image_size, 3), name="image_0_255")
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
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy_probability")(x)
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
        raise RuntimeError(f"Invalid backbone count: {len(nested_models)}")
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
        raise ValueError("Missing training class")
    total = float(counts.sum())
    return {int(label): total / (2.0 * float(count)) for label, count in counts.items()}


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
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
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
    values = np.asarray(history.history["val_pr_auc"], dtype=float)
    best_index = int(np.nanargmax(values))
    return best_index + 1, float(values[best_index])


def report_stage_quality(
    history: tf.keras.callbacks.History,
    stage_name: str,
    validation_metadata: pd.DataFrame,
) -> None:
    best_epoch, best_pr_auc = best_epoch_and_value(history)
    roc_values = np.asarray(history.history.get("val_roc_auc", []), dtype=float)
    best_roc_auc = float(np.nanmax(roc_values)) if roc_values.size else float("nan")
    prevalence = float(validation_metadata["label"].mean())
    LOGGER.info(
        "%s summary: best_epoch=%d, best_val_pr_auc=%.4f, "
        "best_val_roc_auc=%.4f, malignant_prevalence=%.4f",
        stage_name,
        best_epoch,
        best_pr_auc,
        best_roc_auc,
        prevalence,
    )
    if np.isfinite(best_roc_auc) and best_roc_auc < 0.55:
        LOGGER.warning(
            "%s: low validation ROC-AUC %.4f",
            stage_name,
            best_roc_auc,
        )


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
    train_dataset = make_dataset(train_metadata, config, training=True, seed_offset=fold_number)
    validation_dataset = make_dataset(validation_metadata, config, training=False)
    class_weights = balanced_class_weights(train_metadata)

    if variant == "baseline":
        stage_name = f"baseline fold {fold_number}/{config.folds} training"
        LOGGER.info("Starting %s (maximum %d epochs)", stage_name, config.max_epochs_baseline)
        model = build_scratch_cnn(config, regularized=False)
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=config.max_epochs_baseline,
            class_weight=class_weights,
            callbacks=callbacks_for_stage(config, fold_dir / "training", stage_name),
            verbose=2,
        )
        report_stage_quality(history, stage_name, validation_metadata)
        selected_epoch, selected_pr_auc = best_epoch_and_value(history)
        schedule = {
            "selected_epoch": selected_epoch,
            "selected_stage": "single_stage",
            "validation_pr_auc": selected_pr_auc,
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
        report_stage_quality(history, stage_name, validation_metadata)
        selected_epoch, selected_pr_auc = best_epoch_and_value(history)
        schedule = {
            "selected_epoch": selected_epoch,
            "selected_stage": "single_stage",
            "validation_pr_auc": selected_pr_auc,
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
        report_stage_quality(head_history, head_stage_name, validation_metadata)
        head_epoch, head_pr_auc = best_epoch_and_value(head_history)
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
        report_stage_quality(fine_history, fine_stage_name, validation_metadata)
        fine_epoch, fine_pr_auc = best_epoch_and_value(fine_history)
        if fine_pr_auc >= head_pr_auc:
            model.load_weights(fine_dir / "best.weights.h5")
            selected_stage = "fine_tuning"
            selected_pr_auc = fine_pr_auc
        else:
            model.load_weights(head_dir / "best.weights.h5")
            selected_stage = "head_training"
            selected_pr_auc = head_pr_auc
        schedule = {
            "head_epoch": head_epoch,
            "fine_tune_epoch": fine_epoch,
            "selected_stage": selected_stage,
            "validation_pr_auc": selected_pr_auc,
        }
    else:
        raise ValueError(f"Unknown model: {variant}")

    schedule["class_weights"] = class_weights
    return model, schedule


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(
    true_labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    y_predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    has_two_classes = np.unique(y_true).size == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_predicted)),
        "sensitivity": safe_ratio(int(tp), int(tp + fn)),
        "specificity": safe_ratio(int(tn), int(tn + fp)),
        "precision_ppv": float(precision_score(y_true, y_predicted, zero_division=0)),
        "negative_predictive_value": safe_ratio(int(tn), int(tn + fn)),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)) if has_two_classes else float("nan"),
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
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probability, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]

    youden = tpr - fpr
    best_youden = np.flatnonzero(youden == np.max(youden))

    youden_index = int(best_youden[np.argmax(thresholds[best_youden])])

    eligible = np.flatnonzero(tpr >= target_sensitivity)
    if not len(eligible):
        raise RuntimeError(f"Sensitivity target unmet: {target_sensitivity:.3f}")
    best_specificity = np.min(fpr[eligible])
    specificity_ties = eligible[fpr[eligible] == best_specificity]
    sensitivity_index = int(specificity_ties[np.argmax(thresholds[specificity_ties])])
    return Thresholds(
        default=0.5,
        youden_validation=float(thresholds[youden_index]),
        sensitivity_target_validation=float(thresholds[sensitivity_index]),
    )


def save_operating_points(
    true_labels: Sequence[int], probabilities: Sequence[float], output_path: Path
) -> None:
    y_true = np.asarray(true_labels, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probability, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    table = pd.DataFrame(
        {
            "threshold": thresholds[finite],
            "sensitivity": tpr[finite],
            "specificity": 1.0 - fpr[finite],
            "youden_j": tpr[finite] - fpr[finite],
        }
    )
    table.to_csv(output_path, index=False)


def make_cv_splitter(config: Config) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.seed,
    )


def stratification_labels(metadata: pd.DataFrame) -> np.ndarray:
    return (
        metadata["lesion_type"].astype(str)
        + "_label_"
        + metadata["label"].astype(str)
    ).to_numpy()


def make_development_calibration_split(
    train_metadata: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approximate_splits = max(3, int(round(1.0 / config.calibration_fraction)))
    splitter = StratifiedGroupKFold(
        n_splits=approximate_splits,
        shuffle=True,
        random_state=config.seed + 101,
    )
    strata = stratification_labels(train_metadata)
    candidates = list(
        splitter.split(train_metadata, strata, groups=train_metadata["patient_id"])
    )
    target_rows = config.calibration_fraction * len(train_metadata)
    development_index, calibration_index = min(
        candidates, key=lambda pair: abs(len(pair[1]) - target_rows)
    )
    development = train_metadata.iloc[development_index].reset_index(drop=True)
    calibration = train_metadata.iloc[calibration_index].reset_index(drop=True)
    validate_fold_integrity(development, calibration, fold_number=0)

    assignment = train_metadata[
        ["patient_id", "case_id", "lesion_type", "label", "image_path"]
    ].copy()
    assignment["internal_role"] = "development"
    assignment.loc[calibration_index, "internal_role"] = "calibration"
    assignment.to_csv(config.output_dir / "internal_split_assignments.csv", index=False)
    write_json(
        config.output_dir / "internal_split_summary.json",
        {
            "development": split_summary(development),
            "calibration": split_summary(calibration),
            "requested_calibration_fraction": config.calibration_fraction,
            "actual_calibration_fraction": len(calibration) / len(train_metadata),
        },
    )
    return development, calibration


def validate_fold_integrity(
    train_fold: pd.DataFrame, validation_fold: pd.DataFrame, fold_number: int
) -> None:
    overlap = set(train_fold["patient_id"]).intersection(validation_fold["patient_id"])
    if overlap:
        raise RuntimeError(f"Patient overlap in fold {fold_number}")
    if train_fold["label"].nunique() != 2 or validation_fold["label"].nunique() != 2:
        raise RuntimeError(f"Missing class in fold {fold_number}")


def cross_validate_variant(
    variant: str,
    train_metadata: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    variant_dir = config.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    oof_probability = np.full(len(train_metadata), np.nan, dtype=float)
    oof_fold = np.full(len(train_metadata), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    splitter = make_cv_splitter(config)
    strata = stratification_labels(train_metadata)

    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(train_metadata, strata, groups=train_metadata["patient_id"]), start=1
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
        probabilities = model.predict(
            make_dataset(fold_validation, config, training=False), verbose=0
        ).ravel()
        oof_probability[validation_index] = probabilities
        oof_fold[validation_index] = fold_number
        fold_metrics = compute_metrics(fold_validation["label"], probabilities, 0.5)
        fold_rows.append(
            {
                "model": variant,
                "fold": fold_number,
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "train_patients": fold_train["patient_id"].nunique(),
                "validation_patients": fold_validation["patient_id"].nunique(),
                **schedule,
                **fold_metrics,
            }
        )
        schedules.append(schedule)
        del model
        tf.keras.backend.clear_session()

    if np.isnan(oof_probability).any():
        raise RuntimeError(f"Missing OOF predictions: {variant}")
    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(variant_dir / "cv_fold_metrics.csv", index=False)
    oof = train_metadata[
        ["patient_id", "lesion_type", "label", "image_path", "pathology_original"]
    ].copy()
    oof["model"] = variant
    oof["fold"] = oof_fold
    oof["probability"] = oof_probability
    oof.to_csv(variant_dir / "oof_predictions.csv", index=False)

    if variant == "transfer":
        fine_tuning_selected = sum(
            item["selected_stage"] == "fine_tuning" for item in schedules
        ) >= math.ceil(len(schedules) / 2)
        final_schedule = {
            "head_epochs": max(1, int(np.median([item["head_epoch"] for item in schedules]))),
            "fine_tune_epochs": max(
                1, int(np.median([item["fine_tune_epoch"] for item in schedules]))
            ),
            "use_fine_tuning": fine_tuning_selected,
        }
    else:
        final_schedule = {
            "epochs": max(1, int(np.median([item["selected_epoch"] for item in schedules])))
        }
    write_json(variant_dir / "final_training_schedule.json", final_schedule)
    return folds_frame, oof, final_schedule


def train_final_model(
    variant: str,
    train_metadata: pd.DataFrame,
    schedule: dict[str, int],
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
        raise ValueError(f"Unknown model: {variant}")

    model_path = config.output_dir / variant / "final_model.keras"
    model.save(model_path)
    return model


def threshold_rows(
    model_name: str,
    split_name: str,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, threshold in asdict(thresholds).items():
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "threshold_policy": policy,
                **compute_metrics(metadata["label"], probabilities, threshold),
            }
        )
    return rows


def bootstrap_confidence_intervals(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: Thresholds,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
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
        if not values:
            continue
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
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
                sampled_labels = y_true[sampled_indices]
                if np.unique(sampled_labels).size < 2:
                    continue
                difference = roc_auc_score(
                    sampled_labels, prediction_columns[second][sampled_indices]
                ) - roc_auc_score(sampled_labels, prediction_columns[first][sampled_indices])
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
    save_visual_samples(train_metadata, config.output_dir)
    if config.audit_only:
        LOGGER.info("Audit-only run complete. Inspect sample_check before training.")
        return
    development_metadata, calibration_metadata = make_development_calibration_split(
        train_metadata, config
    )

    all_cv_rows: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_subgroup_rows: list[dict[str, Any]] = []
    all_ci_rows: list[pd.DataFrame] = []
    test_predictions: dict[str, np.ndarray] = {}

    for model_index, variant in enumerate(config.models):
        LOGGER.info("Starting model: %s", variant)
        cv_frame, oof, schedule = cross_validate_variant(
            variant, development_metadata, config
        )
        all_cv_rows.append(cv_frame)
        exploratory_oof_thresholds = select_validation_thresholds(
            oof["label"], oof["probability"], config.target_sensitivity
        )
        write_json(
            config.output_dir / variant / "exploratory_oof_thresholds.json",
            {
                "selection_source": "development_out_of_fold_predictions",
                **asdict(exploratory_oof_thresholds),
            },
        )
        save_operating_points(
            oof["label"],
            oof["probability"],
            config.output_dir / variant / "oof_operating_points.csv",
        )
        all_metric_rows.extend(
            threshold_rows(
                variant,
                "development_oof",
                development_metadata,
                oof["probability"].to_numpy(),
                exploratory_oof_thresholds,
            )
        )

        final_model = train_final_model(variant, development_metadata, schedule, config)
        calibration_probability = final_model.predict(
            make_dataset(calibration_metadata, config, training=False), verbose=0
        ).ravel()
        thresholds = select_validation_thresholds(
            calibration_metadata["label"],
            calibration_probability,
            config.target_sensitivity,
        )
        write_json(
            config.output_dir / variant / "final_calibration_thresholds.json",
            {
                "selection_source": "reserved_patient_level_calibration_subset",
                **asdict(thresholds),
            },
        )
        save_operating_points(
            calibration_metadata["label"],
            calibration_probability,
            config.output_dir / variant / "calibration_operating_points.csv",
        )
        calibration_predictions = calibration_metadata[
            [
                "patient_id",
                "case_id",
                "lesion_type",
                "label",
                "image_path",
                "pathology_original",
            ]
        ].copy()
        calibration_predictions["probability"] = calibration_probability
        calibration_predictions.to_csv(
            config.output_dir / variant / "calibration_predictions.csv", index=False
        )
        all_metric_rows.extend(
            threshold_rows(
                variant,
                "reserved_calibration",
                calibration_metadata,
                calibration_probability,
                thresholds,
            )
        )

        test_probability = final_model.predict(
            make_dataset(test_metadata, config, training=False), verbose=0
        ).ravel()
        test_predictions[variant] = test_probability
        prediction_frame = test_metadata[
            [
                "patient_id",
                "case_id",
                "lesion_type",
                "label",
                "image_path",
                "pathology_original",
            ]
        ].copy()
        prediction_frame["probability"] = test_probability
        for policy, threshold in asdict(thresholds).items():
            predicted = (test_probability >= threshold).astype(int)
            prediction_frame[f"prediction_{policy}"] = predicted
            prediction_frame[f"error_{policy}"] = np.select(
                [
                    (prediction_frame["label"].to_numpy() == 1) & (predicted == 0),
                    (prediction_frame["label"].to_numpy() == 0) & (predicted == 1),
                ],
                ["false_negative", "false_positive"],
                default="correct",
            )
        prediction_frame.to_csv(
            config.output_dir / variant / "official_test_predictions.csv", index=False
        )
        all_metric_rows.extend(
            threshold_rows(variant, "official_test", test_metadata, test_probability, thresholds)
        )
        all_subgroup_rows.extend(
            subgroup_rows(variant, test_metadata, test_probability, thresholds)
        )
        ci_frame = bootstrap_confidence_intervals(
            test_metadata,
            test_probability,
            thresholds,
            repeats=config.bootstrap_repeats,
            seed=config.seed + 20_000 + model_index,
        )
        ci_frame.insert(0, "model", variant)
        all_ci_rows.append(ci_frame)
        del final_model
        tf.keras.backend.clear_session()

    pd.concat(all_cv_rows, ignore_index=True).to_csv(
        config.output_dir / "all_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame(all_metric_rows).to_csv(
        config.output_dir / "all_operating_point_metrics.csv", index=False
    )
    pd.DataFrame(all_subgroup_rows).to_csv(
        config.output_dir / "official_test_subgroup_metrics.csv", index=False
    )
    pd.concat(all_ci_rows, ignore_index=True).to_csv(
        config.output_dir / "official_test_cluster_bootstrap_cis.csv", index=False
    )
    paired_auc_bootstrap(
        test_metadata,
        test_predictions,
        repeats=config.bootstrap_repeats,
        seed=config.seed + 30_000,
    ).to_csv(config.output_dir / "official_test_paired_auc_differences.csv", index=False)
    LOGGER.info("All requested experiments completed")


run_experiments(parse_args())
