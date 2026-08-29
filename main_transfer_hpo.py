from pathlib import Path

from hpo.config import HPOConfig, parse_hpo_args
from hpo.experiment import run_hpo


def parse_args() -> HPOConfig:
    return parse_hpo_args(Path(__file__).stem)


run_hpo(parse_args())
