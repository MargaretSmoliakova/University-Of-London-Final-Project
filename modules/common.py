"""
Provide shared definitions and small helpers used throughout the project.

This module keeps common dataset labels, image details, decision thresholds,
logging, and JSON output handling in one place.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger("cbis_ddsm")

PATHOLOGY_TO_LABEL = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}
METADATA_FILES = {
    "mass": ("mass_case_description_train_set.csv", "mass_case_description_test_set.csv"),
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
class DicomInfo:
    """
    Keep the image details needed to identify a DICOM file and assess its suitability.
    """
    path: str
    rows: int
    columns: int
    bits_stored: int
    sop_instance_uid: str
    series_instance_uid: str
    mask_like: bool

    @property
    def area(self) -> int:
        """Return the image's pixel count so candidate crops can be compared by size."""
        return self.rows * self.columns


@dataclass(frozen=True)
class Thresholds:
    """Keep the decision cutoffs used to turn malignancy scores into predictions."""
    default: float
    youden_validation: float
    sensitivity_target_validation: float


def json_default(value: Any) -> Any:
    """
    Convert paths and numerical values into forms that can be saved in JSON reports.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    """Save a readable JSON report, creating its destination folder when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )
