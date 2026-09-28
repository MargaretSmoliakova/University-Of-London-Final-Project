"""
Connect dataset records to mammogram images and check that they are usable.

This module cleans metadata, locates lesion crops, checks for missing images and
overlap between training and test data, and saves samples for visual review.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .common import (
    CASE_ID_PATTERN,
    LOGGER,
    METADATA_FILES,
    PATHOLOGY_TO_LABEL,
    UID_COMPONENT_PATTERN,
    DicomInfo,
    write_json,
)
from .config import Config
from .dicom_io import file_sha256, inspect_dicom, percentile_window_to_255, read_pixel_array


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names so metadata files can be read consistently."""
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


class ImageResolver:
    """Match metadata image references to usable lesion crops in the dataset."""
    def __init__(self, dataset_root: Path):
        """
        Index the available DICOM files so image references can be resolved efficiently.
        """
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
            uid_components = [
                component
                for component in path.parts
                if UID_COMPONENT_PATTERN.fullmatch(component)
            ]
            if uid_components:
                self.by_series_uid.setdefault(uid_components[-1], []).append(resolved_path)
        LOGGER.info(
            "Indexed %d DICOM files across %d case identifiers and %d series UID directories",
            len(dicom_paths),
            len(self.by_case_id),
            len(self.by_series_uid),
        )

    @staticmethod
    def _uid_components(reference: str) -> set[str]:
        """
        Extract image identifiers from a reference path to help match it to local files.
        """
        return {
            component
            for component in reference.replace("\\", "/").split("/")
            if UID_COMPONENT_PATTERN.fullmatch(component)
        }

    @staticmethod
    def _series_uid(reference: str) -> str:
        """Find the series identifier in an image reference for a fallback lookup."""
        components = [
            component
            for component in reference.replace("\\", "/").split("/")
            if UID_COMPONENT_PATTERN.fullmatch(component)
        ]
        return components[-1] if components else ""

    def _direct_candidates(self, reference: str) -> list[Path]:
        """
        Find local files that match the supplied image path or its case-specific
        portion.
        """
        normalised = reference.replace("\\", "/").lstrip("/")
        candidates: list[Path] = []
        direct = self.dataset_root / normalised
        if direct.is_file():
            candidates.append(direct.resolve())
        elif direct.is_dir():
            candidates.extend(sorted(path.resolve() for path in direct.rglob("*.dcm")))
        match = CASE_ID_PATTERN.search(normalised)
        if match:
            suffix = self.dataset_root / normalised[match.start() :]
            if suffix.is_file():
                candidates.append(suffix.resolve())
            elif suffix.is_dir():
                candidates.extend(sorted(path.resolve() for path in suffix.rglob("*.dcm")))
        return list(dict.fromkeys(candidates))

    def resolve_crop(self, reference: Any) -> dict[str, Any]:
        """
        Choose a likely tissue crop for a metadata image reference.

        Check matching files, exclude apparent masks, and return either the selected
        image's details or information explaining why no image could be resolved.
        """
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
            self.by_series_uid.get(requested_series_uid, []) if requested_series_uid else []
        )
        using_uid_fallback = not primary_candidates and bool(uid_candidates)
        candidates = (
            list(dict.fromkeys(uid_candidates))
            if using_uid_fallback
            else primary_candidates
        )
        if not candidates:
            patient_match = re.search(r"p_\d+", case_id, re.IGNORECASE)
            patient_id = patient_match.group(0).lower() if patient_match else ""
            available = sorted(self.case_ids_by_patient.get(patient_id, set()))
            same_view_prefix = re.sub(r"_\d+$", "", case_id)
            same_view = [item for item in available if item.startswith(same_view_prefix)]
            return {
                "image_path": None,
                "resolution_method": "unresolved",
                "case_id": case_id,
                "requested_series_instance_uid": requested_series_uid,
                "available_case_ids_for_patient": " | ".join(available),
                "same_view_case_id_candidates": " | ".join(same_view),
            }

        inspected: list[DicomInfo] = []
        errors: list[str] = []
        for candidate in candidates:
            try:
                inspected.append(inspect_dicom(str(candidate)))
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        if using_uid_fallback:
            inspected = [
                info for info in inspected if info.series_instance_uid == requested_series_uid
            ]
        tissue = [info for info in inspected if not info.mask_like]
        if not tissue:
            return {
                "image_path": None,
                "resolution_method": (
                    "series_uid_header_mismatch_or_no_tissue"
                    if using_uid_fallback
                    else "only_mask_or_unreadable_candidates"
                ),
                "case_id": case_id,
                "requested_series_instance_uid": requested_series_uid,
                "resolution_errors": " | ".join(errors[:3]),
            }

        direct_set = {str(path) for path in direct_candidates}

        def rank(info: DicomInfo) -> tuple[int, int, int, str]:
            """
            Prefer smaller tissue images, then direct path and identifier matches, when
            choosing a crop.
            """
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
            "series_uid_ranked_crop"
            if using_uid_fallback
            else (
                "exact_or_direct_crop"
                if selected.path in direct_set
                else "case_id_ranked_crop"
            )
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
    """
    Read one metadata table and keep usable, labelled cases under the chosen inclusion
    policy.
    """
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
    """Load the official training and test records for the requested lesion types."""
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
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def resolve_metadata_images(
    metadata: pd.DataFrame, resolver: ImageResolver
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach local image details to metadata and separate records whose images cannot be
    found.
    """
    resolution = pd.DataFrame(
        [
            resolver.resolve_crop(reference)
            for reference in metadata["cropped image file path"]
        ],
        index=metadata.index,
    )
    combined = pd.concat([metadata, resolution], axis=1)
    unresolved = combined[combined["image_path"].isna()].copy()
    resolved = combined.dropna(subset=["image_path"]).copy().reset_index(drop=True)
    return resolved, unresolved.reset_index(drop=True)


def split_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise a data split's size, patients, labels, and image-resolution results for
    reporting.
    """
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
    """
    Remove training cases for patients also present in the other lesion type's test
    data.

    Record the exclusions and reject overlap within the same lesion type so the
    combined dataset can keep its official test cases separate.
    """
    overlap = sorted(set(train_metadata["patient_id"]) & set(test_metadata["patient_id"]))
    patient_details: list[dict[str, Any]] = []
    same_lesion_overlap: list[str] = []
    for patient_id in overlap:
        train_types = sorted(
            train_metadata.loc[
                train_metadata["patient_id"] == patient_id, "lesion_type"
            ].unique()
        )
        test_types = sorted(
            test_metadata.loc[
                test_metadata["patient_id"] == patient_id, "lesion_type"
            ].unique()
        )
        common = sorted(set(train_types) & set(test_types))
        if common:
            same_lesion_overlap.append(patient_id)
        patient_details.append(
            {
                "patient_id": patient_id,
                "train_lesion_types": train_types,
                "test_lesion_types": test_types,
                "same_lesion_types_across_splits": common,
            }
        )

    excluded_resolved = train_metadata[train_metadata["patient_id"].isin(overlap)].copy()
    excluded_unresolved = unresolved_train[unresolved_train["patient_id"].isin(overlap)].copy()
    parts: list[pd.DataFrame] = []
    for status, frame in (
        ("resolved", excluded_resolved),
        ("unresolved", excluded_unresolved),
    ):
        part = frame.copy()
        part.insert(0, "resolution_status", status)
        parts.append(part)
    pd.concat(parts, ignore_index=True, sort=False).to_csv(
        output_dir / "excluded_train_patient_overlap.csv", index=False
    )

    report: dict[str, Any] = {
        "policy": "official-test-priority: exclude overlapping patients from train",
        "reason": (
            "The official mass and calcification partitions are separate; their "
            "combination is not globally patient-disjoint."
        ),
        "official_test_changed": False,
        "overlapping_patient_count": len(overlap),
        "overlapping_patient_ids": overlap,
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
        raise RuntimeError(f"Same-lesion overlap: {len(same_lesion_overlap)} patients")
    filtered_train = (
        train_metadata[~train_metadata["patient_id"].isin(overlap)]
        .copy()
        .reset_index(drop=True)
    )
    filtered_unresolved = unresolved_train[
        ~unresolved_train["patient_id"].isin(overlap)
    ].copy().reset_index(drop=True)
    if set(filtered_train["patient_id"]) & set(test_metadata["patient_id"]):
        raise AssertionError("Patient overlap remains")
    report["train_after"] = split_summary(filtered_train)
    report["status"] = "applied"
    write_json(output_dir / "combined_patient_overlap_resolution.json", report)
    LOGGER.warning(
        "Excluded overlapping rows: %d resolved, %d unresolved; %d patients",
        len(excluded_resolved),
        len(excluded_unresolved),
        len(overlap),
    )
    return filtered_train, filtered_unresolved, report


def audit_dataset(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    unresolved_train: pd.DataFrame,
    unresolved_test: pd.DataFrame,
    output_dir: Path,
    official_split_adjustment: dict[str, Any] | None = None,
) -> None:
    """
    Check that training and test data are complete, consistent, and separate.

    Save audit reports and stop on missing images, shared patients or images,
    conflicting labels, duplicate rows, or absent classes.
    """
    train_patients, test_patients = (
        set(train_metadata.patient_id),
        set(test_metadata.patient_id),
    )
    patient_overlap = sorted(train_patients & test_patients)
    train_paths, test_paths = set(train_metadata.image_path), set(test_metadata.image_path)
    path_overlap = sorted(train_paths & test_paths)
    train_sops = {uid for uid in train_metadata.sop_instance_uid if uid}
    test_sops = {uid for uid in test_metadata.sop_instance_uid if uid}
    sop_overlap = sorted(train_sops & test_sops)

    combined = pd.concat([train_metadata, test_metadata], ignore_index=True)
    label_conflicts = combined.groupby("image_path")["label"].nunique().loc[lambda x: x > 1]
    duplicate_counts = (
        combined.groupby(["official_split", "image_path"]).size().loc[lambda x: x > 1]
    )
    duplicate_rows = int((duplicate_counts - 1).sum())
    train_hashes = {file_sha256(path): path for path in sorted(train_paths)}
    test_hashes = {file_sha256(path): path for path in sorted(test_paths)}
    hash_overlap = sorted(set(train_hashes) & set(test_hashes))

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
            for (split, path), count in duplicate_counts.head(10).items()
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
        raise RuntimeError(
            f"Unresolved images: train={len(unresolved_train)}, "
            f"test={len(unresolved_test)}"
        )
    failures = {
        "patient overlap": patient_overlap,
        "path overlap": path_overlap,
        "SOP Instance UID overlap": sop_overlap,
        "byte-identical files": hash_overlap,
    }
    present = {name: values for name, values in failures.items() if values}
    if present:
        raise RuntimeError("Train/test leakage")
    if len(label_conflicts):
        raise RuntimeError("Conflicting image labels")
    if duplicate_rows:
        raise RuntimeError("Duplicate image rows")
    if train_metadata.label.nunique() != 2 or test_metadata.label.nunique() != 2:
        raise RuntimeError("Class missing from split")


def save_visual_samples(metadata: pd.DataFrame, output_dir: Path, count: int = 24) -> None:
    """
    Save a selection of lesion images so their contents can be checked visually before
    training.
    """
    sample_dir = output_dir / "sample_check"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if metadata.empty:
        return
    sample_count = min(count, len(metadata))
    parts = [
        group.sample(n=min(max(1, sample_count // 4), len(group)), random_state=17)
        for _, group in metadata.groupby(["lesion_type", "label"])
    ]
    sampled = pd.concat(parts, ignore_index=True).head(sample_count)
    for index, row in enumerate(sampled.itertuples(index=False)):
        pixels = percentile_window_to_255(read_pixel_array(row.image_path))
        image = Image.fromarray(pixels.astype("uint8")).resize(
            (384, 384), resample=Image.Resampling.BILINEAR
        )
        image.save(
            sample_dir
            / f"{index:02d}_{row.lesion_type}_label-{row.label}_{row.patient_id}.png"
        )
    LOGGER.info("Saved %d images for mandatory manual ROI inspection", len(sampled))


def prepare_and_audit_data(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset, locate its images, apply the overlap policy, and run the data
    checks.

    Return the checked training and test records and save example images for review.
    """
    train_raw, test_raw = load_metadata(config)
    resolver = ImageResolver(config.dataset_root)
    train, unresolved_train = resolve_metadata_images(train_raw, resolver)
    test, unresolved_test = resolve_metadata_images(test_raw, resolver)
    adjustment: dict[str, Any] | None = None
    if config.combined_patient_overlap_policy == "exclude-train":
        train, unresolved_train, adjustment = exclude_cross_lesion_patient_overlap_from_train(
            train, test, unresolved_train, config.output_dir
        )
    audit_dataset(
        train,
        test,
        unresolved_train,
        unresolved_test,
        config.output_dir,
        adjustment,
    )
    LOGGER.info("Data audit passed")
    save_visual_samples(train, config.output_dir)
    return train, test
