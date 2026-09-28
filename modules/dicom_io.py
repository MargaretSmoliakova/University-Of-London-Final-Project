"""
Read mammogram images and inspect the DICOM files that store them.

These helpers prepare pixel values, collect image details, flag possible masks,
and identify identical files during dataset checks.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

from .common import DicomInfo


def read_pixel_array(image_path: str | Path) -> np.ndarray:
    """
    Load a mammogram as a grayscale image with usable pixel values.

    Apply the DICOM display adjustments when present and reject images with an
    unsupported shape or invalid pixels.
    """
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
    """
    Bring image brightness into a common range while limiting extreme values.

    This makes images with different original intensity ranges suitable for the
    same training and visualisation steps.
    """
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
    """
    Estimate how many different pixel values an image contains.

    Sampling large images keeps this check practical when looking for possible masks.
    """
    flattened = pixels.ravel()
    if flattened.size > maximum_samples:
        stride = math.ceil(flattened.size / maximum_samples)
        flattened = flattened[::stride]
    return int(np.unique(flattened).size)


@lru_cache(maxsize=50_000)
def inspect_dicom(path_string: str) -> DicomInfo:
    """
    Collect a DICOM image's dimensions and identifiers, and flag possible masks.

    The result helps the image resolver choose tissue crops and supports checks
    for duplicate images.
    """
    path = Path(path_string)
    dicom = pydicom.dcmread(str(path), stop_before_pixels=False)
    pixels = read_pixel_array(path)
    unique_count = sampled_unique_count(pixels)
    return DicomInfo(
        path=str(path.resolve()),
        rows=int(pixels.shape[0]),
        columns=int(pixels.shape[1]),
        bits_stored=int(dicom.get("BitsStored", 0) or 0),
        sop_instance_uid=str(dicom.get("SOPInstanceUID", "")),
        series_instance_uid=str(dicom.get("SeriesInstanceUID", "")),
        mask_like=unique_count <= 32,
    )


def file_sha256(path: str | Path) -> str:
    """Create a fingerprint of a file's contents to detect identical image files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
