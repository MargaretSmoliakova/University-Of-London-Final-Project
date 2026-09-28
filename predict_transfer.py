"""Predict benign (0) or malignant (1) from one cropped DICOM lesion ROI.

Run from the project's code directory, where the modules/ package is available.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from modules.data_pipeline import load_one_image_for_visualisation


DEFAULT_RESULTS = Path(
    'D:/UoL_Final_Project_Dataset/results/cbis_full_selected_hpo/main_experiments/transfer'
)


def main():
    parser = argparse.ArgumentParser(description='Classify one cropped DICOM ROI.')
    parser.add_argument('dicom_path', type=Path)
    parser.add_argument('--model', type=Path, default=DEFAULT_RESULTS / 'final_model.keras')
    parser.add_argument('--thresholds', type=Path, default=DEFAULT_RESULTS / 'final_calibration_thresholds.json')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    # A directory named *.dcm is not a DICOM file.
    if not args.dicom_path.is_file():
        parser.error(f'DICOM file not found (or path is a directory): {args.dicom_path}')
    if not args.model.is_file():
        parser.error(f'Saved model not found: {args.model}')

    # Default: use the threshold selected on held-out calibration patients.
    if args.threshold is None:
        if not args.thresholds.is_file():
            parser.error(f'Thresholds JSON not found: {args.thresholds}')
        thresholds = json.loads(args.thresholds.read_text(encoding='utf-8'))
        threshold = float(thresholds['sensitivity_target_validation'])
    else:
        threshold = args.threshold
    if not 0.0 <= threshold <= 1.0:
        parser.error('Threshold must be between 0 and 1.')

    # Apply the project's DICOM preprocessing (224 x 224, three replicated channels).
    image = load_one_image_for_visualisation(args.dicom_path, 224)
    image_batch = image[None, ...] 

    # Load existing trained weights. No retraining or additional 1/255 scaling.
    model = tf.keras.models.load_model(str(args.model), compile=False)
    score = float(model.predict(image_batch, verbose=0)[0, 0])

    prediction = int(score >= threshold)
    print(f'Malignant score: {score:.4f}')
    print(f'Decision threshold: {threshold:.4f}')
    print(f'Predicted class: {prediction} ({"malignant" if prediction else "benign"})')
    print('Note: Research model for pre-localised lesion crops, not a clinical diagnosis.')


main()
