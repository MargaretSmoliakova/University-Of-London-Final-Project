from __future__ import annotations

import argparse
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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
    lesion_type: str = "both"
    image_size: int = 224
    batch_size: int = 16
    seed: int = 42
    max_epochs_baseline: int = 30
    include_benign_without_callback: bool = False

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
    parser = argparse.ArgumentParser(description="Refactored CBIS-DDSM baseline experiment")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lesion-type", choices=("both", "mass", "calcification"), default="both")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs-baseline", type=int, default=30)
    parser.add_argument("--include-benign-without-callback", action="store_true")
    args = parser.parse_args()
    return Config(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        lesion_type=args.lesion_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_epochs_baseline=args.max_epochs_baseline,
        include_benign_without_callback=args.include_benign_without_callback,
    )


def configure_runtime(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)

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

def audit_dataset(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    unresolved_train: pd.DataFrame,
    unresolved_test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    train_patients = set(train_metadata["patient_id"])
    test_patients = set(test_metadata["patient_id"])
    patient_overlap = sorted(train_patients.intersection(test_patients))
    path_overlap = sorted(
        set(train_metadata["image_path"]).intersection(test_metadata["image_path"])
    )
    train_sops = {uid for uid in train_metadata["sop_instance_uid"] if uid}
    test_sops = {uid for uid in test_metadata["sop_instance_uid"] if uid}
    sop_overlap = sorted(train_sops.intersection(test_sops))

    audit = {
        "train": split_summary(train_metadata),
        "test": split_summary(test_metadata),
        "unresolved_train_rows": int(len(unresolved_train)),
        "unresolved_test_rows": int(len(unresolved_test)),
        "patient_overlap_count": len(patient_overlap),
        "path_overlap_count": len(path_overlap),
        "sop_uid_overlap_count": len(sop_overlap),
    }
    write_json(output_dir / "data_audit.json", audit)
    unresolved_train.to_csv(output_dir / "unresolved_train.csv", index=False)
    unresolved_test.to_csv(output_dir / "unresolved_test.csv", index=False)
    train_metadata.to_csv(output_dir / "resolved_train_manifest.csv", index=False)
    test_metadata.to_csv(output_dir / "resolved_test_manifest.csv", index=False)

    if patient_overlap or path_overlap or sop_overlap:
        raise RuntimeError(
            "Potential train/test leakage detected; inspect data_audit.json before training"
        )
    return audit


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
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(metadata),
            seed=config.seed + seed_offset,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )


def build_baseline_cnn(config: Config) -> tf.keras.Model:
    inputs = tf.keras.Input((config.image_size, config.image_size, 3), name="image_0_255")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    compile_binary_model(model, 1e-3)
    return model

def balanced_class_weights(metadata: pd.DataFrame) -> dict[int, float]:
    counts = metadata["label"].value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError(
            f"Training fold does not contain both classes: {counts.to_dict()}"
        )
    total = float(counts.sum())
    return {int(label): total / (2.0 * float(count)) for label, count in counts.items()}

def run_experiment(config: Config) -> None:
    configure_runtime(config)
    write_json(config.output_dir / "config.json", asdict(config))

    raw_train, raw_test = load_metadata(config)
    resolver = ImageResolver(config.dataset_root)
    train_metadata, unresolved_train = resolve_metadata_images(raw_train, resolver)
    test_metadata, unresolved_test = resolve_metadata_images(raw_test, resolver)
    audit_dataset(
        train_metadata,
        test_metadata,
        unresolved_train,
        unresolved_test,
        config.output_dir,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.seed)
    train_index, validation_index = next(
        splitter.split(train_metadata, groups=train_metadata["patient_id"])
    )
    fold_train = train_metadata.iloc[train_index].reset_index(drop=True)
    fold_validation = train_metadata.iloc[validation_index].reset_index(drop=True)

    model = build_baseline_cnn(config)
    model.fit(
        make_dataset(fold_train, config, training=True),
        validation_data=make_dataset(fold_validation, config, training=False),
        epochs=config.max_epochs_baseline,
        class_weight=balanced_class_weights(fold_train),
        verbose=2,
    )
    model.save(config.output_dir / "baseline_model.keras")


if __name__ == "__main__":
    run_experiment(parse_args())
