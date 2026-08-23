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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )
