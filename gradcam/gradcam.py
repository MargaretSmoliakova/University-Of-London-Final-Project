from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw

from modules.common import LOGGER, write_json
from modules.data_pipeline import load_one_image_for_visualisation


def find_gradcam_target_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer

    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.MaxPooling2D, tf.keras.layers.Conv2D)):
            return layer

    raise RuntimeError("No Grad-CAM target layer")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    target_layer: tf.keras.layers.Layer,
) -> tuple[np.ndarray, float]:
    image_batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        value = image_batch
        feature_maps = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            value = layer(value, training=False)
            if layer is target_layer:
                feature_maps = value
                tape.watch(feature_maps)

        if feature_maps is None:
            raise RuntimeError(f"Layer not executed: {target_layer.name}")

        predictions = value
        malignant_score = predictions[:, 0]

    gradients = tape.gradient(malignant_score, feature_maps)
    if gradients is None:
        raise RuntimeError(f"No gradients: {target_layer.name}")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(feature_maps * channel_weights, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)

    maximum = float(tf.reduce_max(heatmap).numpy())
    if maximum > 0:
        heatmap = heatmap / maximum

    return np.asarray(heatmap, dtype=np.float32), float(predictions[0, 0].numpy())


def _colour_heatmap(heatmap: np.ndarray) -> np.ndarray:
    intensity = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0.0, 1.0)

    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _make_panel(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    alpha: float,
) -> Image.Image:
    image_size = int(base_image.shape[0])
    pixels = np.clip(base_image, 0, 255).astype(np.uint8)
    original_image = Image.fromarray(pixels, mode="RGB")

    heatmap_image = Image.fromarray(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L"
    )
    heatmap_image = heatmap_image.resize(
        (image_size, image_size), resample=Image.Resampling.BILINEAR
    )
    coloured_heatmap = Image.fromarray(
        _colour_heatmap(np.asarray(heatmap_image, dtype=np.float32) / 255.0)
    )
    overlay = Image.blend(original_image, coloured_heatmap, alpha=alpha)

    caption_height = 54
    panel = Image.new("RGB", (image_size * 3, image_size + caption_height), "white")
    panel.paste(original_image, (0, caption_height))
    panel.paste(coloured_heatmap, (image_size, caption_height))
    panel.paste(overlay, (image_size * 2, caption_height))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), title, fill="black")
    draw.text((8, 28), "Original", fill="black")
    draw.text((image_size + 8, 28), "Grad-CAM", fill="black")
    draw.text((image_size * 2 + 8, 28), "Overlay", fill="black")

    return panel


def select_explanation_rows(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    maximum_samples: int,
    seed: int,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)
    frame["outcome"] = np.select(
        [
            (frame.label == 1) & (frame.prediction == 1),
            (frame.label == 0) & (frame.prediction == 0),
            (frame.label == 0) & (frame.prediction == 1),
            (frame.label == 1) & (frame.prediction == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )

    rng = np.random.default_rng(seed)
    queues: dict[tuple[str, str], list[int]] = {}

    for key, indices in frame.groupby(["lesion_type", "outcome"], sort=True).groups.items():
        shuffled = np.asarray(list(indices), dtype=int)
        rng.shuffle(shuffled)
        queues[key] = shuffled.tolist()

    selected_indices: list[int] = []

    while len(selected_indices) < min(maximum_samples, len(frame)):
        added = False

        for key in sorted(queues):
            if queues[key] and len(selected_indices) < maximum_samples:
                selected_indices.append(queues[key].pop())
                added = True

        if not added:
            break

    return frame.loc[selected_indices].reset_index(drop=True)


def _save_gradcam_panel(
    model_name: str,
    model: tf.keras.Model,
    target_layer: tf.keras.layers.Layer,
    row: pd.Series,
    index: int,
    model_dir: Path,
    image_size: int,
    youden_threshold: float,
    alpha: float,
) -> dict[str, Any]:
    image = load_one_image_for_visualisation(str(row.image_path), image_size)
    heatmap, checked_probability = make_gradcam_heatmap(model, image, target_layer)
    probability_difference = abs(checked_probability - float(row.probability))

    if probability_difference > 1e-4:
        LOGGER.warning(
            "Grad-CAM probability mismatch for %s: saved=%.6f, recomputed=%.6f",
            row.case_id,
            float(row.probability),
            checked_probability,
        )

    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.case_id))
    filename = f"{index:02d}_{row.outcome}_{row.lesion_type}_{safe_case}.png"
    title = (
        f"{row.outcome} | label={int(row.label)} | "
        f"p(malignant)={checked_probability:.3f}"
    )

    _make_panel(image, heatmap, title, alpha).save(model_dir / filename)

    return {
        "model": model_name,
        "patient_id": row.patient_id,
        "case_id": row.case_id,
        "lesion_type": row.lesion_type,
        "label": int(row.label),
        "saved_test_probability": float(row.probability),
        "recomputed_probability": checked_probability,
        "absolute_probability_difference": probability_difference,
        "youden_threshold": youden_threshold,
        "prediction": int(row.prediction),
        "outcome": row.outcome,
        "gradcam_target_layer": target_layer.name,
        "image_path": row.image_path,
        "panel_file": filename,
    }


def generate_gradcam_artifacts(
    model_name: str,
    model: tf.keras.Model,
    test_metadata: pd.DataFrame,
    probabilities: np.ndarray,
    youden_threshold: float,
    output_dir: Path,
    image_size: int,
    maximum_samples: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    if maximum_samples <= 0:
        return pd.DataFrame()

    target_layer = find_gradcam_target_layer(model)
    selected = select_explanation_rows(
        test_metadata,
        probabilities,
        threshold=youden_threshold,
        maximum_samples=maximum_samples,
        seed=seed,
    )

    model_dir = output_dir / "gradcam" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, row in selected.iterrows():
        record = _save_gradcam_panel(
            model_name,
            model,
            target_layer,
            row,
            index,
            model_dir,
            image_size,
            youden_threshold,
            alpha,
        )
        records.append(record)

    manifest = pd.DataFrame(records)
    manifest.to_csv(model_dir / "gradcam_manifest.csv", index=False)

    write_json(
        model_dir / "gradcam_notes.json",
        {
            "model": model_name,
            "target_layer": target_layer.name,
            "sample_count": len(manifest),
            "selection": "balanced round-robin over lesion type and TP/TN/FP/FN",
            "class_explained": "malignant probability",
            "threshold_source": "reserved calibration subset (Youden)",
        },
    )

    LOGGER.info(
        "Generated %d Grad-CAM panels for %s using layer %s",
        len(manifest),
        model_name,
        target_layer.name,
    )

    return manifest
