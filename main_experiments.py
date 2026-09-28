"""
Start the mammogram classification experiments from command-line settings.

The experiment workflow checks the data, trains the requested models, and saves
results for comparing how well they distinguish benign and malignant lesions.
"""

from pathlib import Path

from modules.config import parse_experiment_args
from modules.experiment import run_experiments


config = parse_experiment_args(Path(__file__).stem)
run_experiments(config)