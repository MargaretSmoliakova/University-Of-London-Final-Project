from pathlib import Path

from hpo.regularized_hpo import parse_hpo_args, run_hpo

run_hpo(parse_hpo_args(Path(__file__).stem))
