from __future__ import annotations

import logging
import random

import numpy as np
import tensorflow as tf

from .common import LOGGER
from .config import Config


def configure_runtime(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.output_dir / "experiment.log", encoding="utf-8"),
        ],
    )
    LOGGER.info(
        "All logs and experiment artifacts for %s.py are stored under %s",
        config.program_name,
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
