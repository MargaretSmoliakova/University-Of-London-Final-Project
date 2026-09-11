from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import tensorflow as tf

from modules.common import LOGGER

from .gradcam import generate_gradcam_artifacts


def parse_gradcam_args(program_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument(
        "--model", choices=("baseline", "regularized", "transfer"), default="transfer"
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else args.results_dir / program_name
    )

    if args.samples < 1:
        parser.error("--samples < 1")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha out of range")

    return args


def configure_gradcam_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "gradcam.log", encoding="utf-8"),
        ],
        force=True,
    )


def run_gradcam(args: argparse.Namespace) -> None:
    configure_gradcam_logging(args.output_dir)

    model_dir = args.results_dir / args.model
    model_path = model_dir / "final_model.keras"
    predictions_path = model_dir / "official_test_predictions.csv"
    thresholds_path = model_dir / "final_calibration_thresholds.json"

    required_files = (model_path, predictions_path, thresholds_path)
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

    metadata = pd.read_csv(predictions_path)
    required_columns = {
        "patient_id",
        "case_id",
        "lesion_type",
        "label",
        "image_path",
        "probability",
    }
    missing_columns = sorted(required_columns.difference(metadata.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    youden_threshold = float(thresholds["youden_validation"])

    model = tf.keras.models.load_model(model_path, compile=False)
    model_image_size = int(model.input_shape[1])
    if model_image_size != args.image_size:
        raise ValueError(
            f"Image size mismatch: {args.image_size} != {model_image_size}"
        )

    LOGGER.info("Loaded completed %s model from %s", args.model, model_path)

    generate_gradcam_artifacts(
        model_name=args.model,
        model=model,
        test_metadata=metadata,
        probabilities=metadata["probability"].to_numpy(dtype=float),
        youden_threshold=youden_threshold,
        output_dir=args.output_dir,
        image_size=args.image_size,
        maximum_samples=args.samples,
        alpha=args.alpha,
        seed=args.seed,
    )

    LOGGER.info("Standalone Grad-CAM completed without model retraining")
