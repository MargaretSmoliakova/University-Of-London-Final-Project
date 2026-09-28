"""
Connect the regularized model's settings to the shared search workflow.

This module supplies the command-line setup used by the regularized search entry
point and exposes the common search runner.
"""

from .config import parse_hpo_args as parse_search_args
from .experiment import run_hpo
from .regularized_config import RegularizedHPOConfig as HPOConfig


def parse_hpo_args(program_name: str) -> HPOConfig:
    """
    Read search options using the defaults and training choices for the regularized
    model.
    """
    return parse_search_args(program_name, HPOConfig)
