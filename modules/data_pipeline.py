from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from .common import LOGGER
from .config import Config
from .dicom_io import percentile_window_to_255, read_pixel_array


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
