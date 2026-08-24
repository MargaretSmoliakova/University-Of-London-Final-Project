from pathlib import Path

from modules.hpo import HPOConfig, parse_hpo_args, run_hpo


def parse_args() -> HPOConfig:
    return parse_hpo_args(Path(__file__).stem)


run_hpo(parse_args())
