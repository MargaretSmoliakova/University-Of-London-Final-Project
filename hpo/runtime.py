from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from modules.common import LOGGER

try:
    import optuna
except ImportError:
    optuna = None


def configure_cpu_threads(cpu_threads: int) -> None:
    if cpu_threads <= 0:
        LOGGER.info("TensorFlow will select CPU thread counts automatically")

        return

    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        LOGGER.info("TensorFlow CPU threads: intra-op=%d, inter-op=2", cpu_threads)
    except RuntimeError as exc:
        LOGGER.warning("CPU thread setup failed: %s", type(exc).__name__)


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "optuna": optuna.__version__ if optuna is not None else None,
    }
