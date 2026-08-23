from pathlib import Path

from modules.config import Config, parse_experiment_args
from modules.experiment import run_experiments


def parse_args() -> Config:
    return parse_experiment_args(Path(__file__).stem)


run_experiments(parse_args())
