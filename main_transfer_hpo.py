"""
Start a search for settings for the transfer-learning mammogram classifier.

The search explores how to adapt a pretrained model to mammogram images and
confirms promising settings for the full experiment.
"""

from pathlib import Path

from hpo.config import HPOConfig, parse_hpo_args
from hpo.experiment import run_hpo

run_hpo(parse_hpo_args(Path(__file__).stem))
