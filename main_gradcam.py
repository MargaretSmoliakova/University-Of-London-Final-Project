from pathlib import Path

from gradcam.gradcam_cli import parse_gradcam_args, run_gradcam


run_gradcam(parse_gradcam_args(Path(__file__).stem))
