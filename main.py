"""
Main script for RPE image analysis pipeline.

This script orchestrates the full pipeline: feature extraction, data cleaning,
PCA, model training, and artifact saving based on a configuration file.
"""

import json
import os
from pathlib import Path
import argparse
import sys
import traceback
from typing import Any, Dict, List, Tuple

# feature extraction and analysis now live in scripts package
from scripts.feature_extraction import extract_features_from_directory
from scripts.analysis import clean_and_prepare, run_pca, train_stacking, save_artifacts, plot_pca_scree, plot_pca_cumulative, plot_pca_scatter, save_classification_report


def run_from_config(config: dict, verbose: bool = False):
    """
    Run the full RPE analysis pipeline from configuration.

    Args:
        config: Configuration dictionary.
        verbose: Enable verbose output.
    """
    from scripts.analysis import resolve_output_dirs
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)

    # 1) Feature extraction
    image_dir = config.get('paths', {}).get('image_directory')
    if not image_dir:
        raise ValueError('image_directory must be set in config.paths')

    print('Extracting features...')
    df = extract_features_from_directory(image_dir, config, verbose=verbose)
    # Determine where to save features CSV: use user-specified path if absolute,
    # otherwise place in <out_root>/reports/
    user_csv = config.get('paths', {}).get('output_features_csv')
    if user_csv and Path(user_csv).is_absolute():
        out_fp = user_csv
    else:
        out_fp = str(reports_dir / 'extracted_features.csv')
    df.to_csv(out_fp, index=False)
    print(f'Features saved to {out_fp} (n_samples={len(df)})')

    # 2) Clean and prepare
    (X_clean, y), preproc = clean_and_prepare(df, config)

    # 3) PCA for visualization
    pca, X_pca = run_pca(X_clean, config.get('analysis_params', {}))
    # pass the project output root to plotting/analysis helpers
    plot_pca_scree(pca.explained_variance_ratio_, str(out_root))
    plot_pca_cumulative(pca.explained_variance_ratio_, str(out_root))
    if X_pca.shape[1] >= 2:
        plot_pca_scatter(X_pca.values, y, str(out_root))

    # 4) Training
    print('Training stacking classifier...')
    model, training_metrics = train_stacking(X_clean, y, config.get('analysis_params', {}))
    # Do not print the training report to terminal; it will be saved in the JSON report

    # 5) Save artifacts and reports
    save_artifacts(preproc, pca, model, str(out_root))
    save_classification_report(y, model.predict(X_clean), str(out_root), training_metrics=training_metrics)



# Note: config file path is provided explicitly via CLI or falls back to the
# workspace absolute path supplied by the user. We intentionally do not search
# multiple locations to keep behavior deterministic.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPE image analysis pipeline")
    parser.add_argument("--config", "-c", dest="config", default=r"D:/GSU/CASA/Math Path 2/Code for RPE Crops/Morphological Feature Analysis of Retinal Pigment Epithelial Cell from C57BL6J Mice during Aging/config.json", help="Path to config.json (absolute)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    args = parser.parse_args()
    # Normalize path separators so both forward and backslashes work on Windows
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"Error: config.json not found at {config_path}")
        sys.exit(2)

    try:
        with open(config_path, 'r') as fh:
            cfg = json.load(fh)
    except Exception as exc:
        print(f'Error: failed to load config: {exc}')
        traceback.print_exc()
        sys.exit(2)

    try:
        # measure total pipeline execution time
        from scripts.timer import Timer
        with Timer('total_pipeline'):
            run_from_config(cfg, verbose=args.verbose)
    except Exception as exc:
        print(f'Pipeline failed: {exc}')
        traceback.print_exc()
        sys.exit(1)

