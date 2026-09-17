from pathlib import Path

from modules.config import parse_experiment_args
from modules.experiment import run_experiments


config = parse_experiment_args(Path(__file__).stem)
run_experiments(config)