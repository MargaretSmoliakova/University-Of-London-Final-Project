from pathlib import Path

from modules.config import parse_experiment_args
from modules.experiment import run_experiments


run_experiments(parse_experiment_args(Path(__file__).stem))