"""
Manage the selected model settings shared by searches and full experiments.

This module checks saved selections, verifies compatible experiment settings,
and loads or saves the files used to reproduce the chosen models.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any


SELECTION_FILENAMES = {
    "regularized": "selected_regularized_parameters.json",
    "transfer": "selected_transfer_parameters.json",
}
HPO_SETTING_NAMES = (
    "image_size", "batch_size", "seed", "calibration_fraction", "lesion_type",
    "combined_patient_overlap_policy", "include_benign_without_callback",
)
PARAMETER_TYPES = {
    "regularized": {
        "learning_rate": float, "l2_strength": float, "spatial_dropout": float,
        "head_dropout": float, "use_batch_norm": bool, "initial_filters": int,
        "block_count": int, "dense_units": int, "activation": str, "pooling": str,
        "rotation": float, "translation": float, "zoom": float, "contrast": float,
        "label_smoothing": float,
    },
    "transfer": {
        "head_learning_rate": float, "fine_tune_learning_rate": float,
        "dense_units": int, "head_dropout_1": float, "head_dropout_2": float,
        "l2_strength": float, "fine_tune_last_layers": int, "rotation": float,
        "translation": float, "zoom": float, "contrast": float,
        "label_smoothing": float, "head_activation": str, "pooling": str,
    },
}


def project_directory() -> Path:
    """Locate the project folder where selected model settings are normally stored."""
    return Path(__file__).resolve().parents[1]


def selection_path(model: str, directory: Path | None = None) -> Path:
    """Find the selected-parameters file for a model in the chosen folder."""
    return (directory if directory is not None else project_directory()) / SELECTION_FILENAMES[model]


def validate_selection(payload: Any, model: str) -> dict[str, Any]:
    """
    Check that saved model settings are complete, valid, and marked as selected without
    reserved data.
    """
    if not isinstance(payload, dict):
        raise ValueError("The selection JSON must contain an object")
    if payload.get("model_variant", model) != model:
        raise ValueError(f"The selection belongs to a different model, not {model}")
    if not isinstance(payload.get("candidate"), str) or not payload["candidate"]:
        raise ValueError("The selection JSON must identify its confirmed candidate")
    for field in ("official_test_used_for_selection", "calibration_subset_used_for_selection"):
        if payload.get(field) is not False:
            raise ValueError(f"The selection JSON must confirm {field}=false")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("The selection JSON must contain a 'parameters' object")
    expected = PARAMETER_TYPES[model]
    missing, extra = set(expected) - set(parameters), set(parameters) - set(expected)
    if missing or extra:
        raise ValueError(f"{model} parameter schema mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name, kind in expected.items():
        value = parameters[name]
        if kind is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is kind
        if not valid:
            raise ValueError(f"Invalid {model} parameter {name}: expected {kind.__name__}")
        if kind is int and value < 1:
            raise ValueError(f"{name} must be positive")
        if "learning_rate" in name and value <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "l2_strength" and value < 0:
            raise ValueError("l2_strength cannot be negative")
        if "dropout" in name and not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
        if name in ("rotation", "translation", "zoom", "contrast") and not 0 <= value <= 0.5:
            raise ValueError(f"{name} must be in [0, 0.5]")
        if name == "label_smoothing" and not 0 <= value < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if name in ("activation", "head_activation") and value not in ("relu", "swish"):
            raise ValueError(f"Unsupported {name}: {value}")
        if name == "pooling" and value not in ("average", "maximum"):
            raise ValueError(f"Unsupported pooling: {value}")
        if name == "block_count" and not 2 <= value <= 6:
            raise ValueError("block_count must be between 2 and 6")
        if name == "initial_filters" and value < 4:
            raise ValueError("initial_filters must be at least 4")
    if "hpo_settings" in payload and not isinstance(payload["hpo_settings"], dict):
        raise ValueError("hpo_settings must be an object")
    return payload


def load_selection(model: str, directory: Path | None = None) -> dict[str, Any]:
    """
    Read and validate a model's selected settings, explaining missing or unusable files.
    """
    path = selection_path(model, directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing HPO selection: {path}. Run main_{model}_hpo.py, or copy that "
            f"run's confirmation/selected_parameters.json here as {path.name}."
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read HPO selection {path}: {exc}") from exc
    try:
        return validate_selection(payload, model)
    except ValueError as exc:
        raise ValueError(f"Invalid HPO selection {path}: {exc}") from exc


def validate_hpo_settings(payload: dict[str, Any], settings: Any) -> None:
    """
    Check that the full experiment matches the recorded data and setup choices used
    during the search.
    """
    recorded = payload.get("hpo_settings", {})
    differences = [
        f"{name}: HPO={recorded[name]!r}, full run={getattr(settings, name)!r}"
        for name in HPO_SETTING_NAMES
        if name in recorded and recorded[name] != getattr(settings, name)
    ]
    if differences:
        raise ValueError("HPO settings do not match the full experiment: " + "; ".join(differences))


def save_selection(model: str, payload: dict[str, Any], directory: Path | None = None) -> Path:
    """
    Validate and save selected settings, replacing the destination only after the new
    file is written.
    """
    from .common import write_json

    validate_selection(payload, model)
    path = selection_path(model, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
