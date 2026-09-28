"""
Start a search for settings for the regularized mammogram classifier.

The search compares candidate settings and confirms promising choices for use
in the full experiment.
"""

from pathlib import Path

from hpo.regularized_hpo import parse_hpo_args, run_hpo

run_hpo(parse_hpo_args(Path(__file__).stem))
