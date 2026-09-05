from .config import parse_hpo_args as parse_search_args
from .experiment import run_hpo
from .regularized_config import RegularizedHPOConfig as HPOConfig


def parse_hpo_args(program_name: str) -> HPOConfig:
    return parse_search_args(program_name, HPOConfig)
