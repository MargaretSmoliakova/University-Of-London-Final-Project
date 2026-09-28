"""
Start visual explanations for a previously trained mammogram classifier.

This command uses saved experiment results to create Grad-CAM images showing
which regions contributed to the model's malignancy score.
"""

from pathlib import Path

from gradcam.gradcam_cli import parse_gradcam_args, run_gradcam


run_gradcam(parse_gradcam_args(Path(__file__).stem))
