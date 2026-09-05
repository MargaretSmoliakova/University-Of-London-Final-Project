from pathlib import Path

from hpo.config import HPOConfig, parse_hpo_args
from hpo.experiment import run_hpo

run_hpo(parse_hpo_args(Path(__file__).stem))
