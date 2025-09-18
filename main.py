"""
Main script for RPE image analysis pipeline.

This script orchestrates the full pipeline: feature extract    # 6) Save artifacts and reports
    print('Saving artifacts and reports...')
    save_artifacts(preproc, pca, model, str(out_root))
    save_classification_report(labels_for_modeling, model.predict(features_for_modeling), str(out_root), training_metrics=training_report)

    # 7) Extract and save feature importances
    print('Extracting feature importances...')
    extract_and_save_feature_importances(model, list(features_for_modeling.columns), str(out_root)) data cleaning,
loading cleaned data from CSV, PCA, model training, and artifact saving based on a configuration file.
"""

import json
from pathlib import Path
import argparse
import sys
import traceback
from typing import Any, Dict, List, Tuple
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# feature extraction and analysis now live in scripts package
from scripts.feature_extraction import extract_features_from_directory
from scripts.clean_and_prepare import clean_and_prepare
from scripts.analysis import (
    run_pca,
    plot_pca_scree,
    plot_pca_cumulative,
    plot_pca_scatter,
    resolve_output_dirs,
    
)
from scripts.train_and_save_pipeline import (
    train_stacking,
    save_artifacts,
    save_classification_report,
)
from scripts.feature_insights import analyze_top_features, extract_and_save_feature_importances


def run_from_config(config: Dict[str, Any], verbose: bool = False) -> None:
    """
    Run the full RPE analysis pipeline from configuration.

    Pipeline steps:
    1. Feature extraction from images
    2. Clean and prepare data (saves cleaned CSV)
    3. Load cleaned data from CSV
    4. PCA for visualization
    5. Training stacking classifier
    6. Save artifacts and reports
    7. Extract and save feature importances

    Args:
        config: Configuration dictionary containing paths and parameters.
        verbose: Enable verbose output for detailed logging.
    """
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)

    # 1) Feature extraction
    image_directory = config.get('paths', {}).get('image_directory')
    if not image_directory:
        raise ValueError('image_directory must be set in config.paths')

    print('Extracting features...')
    features_df = extract_features_from_directory(image_directory, config, verbose=verbose)

    # Determine where to save features CSV: use user-specified path if absolute,
    # otherwise place in <out_root>/reports/
    user_csv = config.get('paths', {}).get('output_features_csv')
    if user_csv and Path(user_csv).is_absolute():
        output_features_path = user_csv
    else:
        output_features_path = str(reports_dir / 'extracted_features.csv')

    try:
        features_df.to_csv(output_features_path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to save features CSV: {exc}") from exc

    print(f'Features saved to {output_features_path} (n_samples={len(features_df)})')

    # 2) Clean and prepare
    print('Cleaning and preparing data...')
    (features_clean, labels), preproc = clean_and_prepare(features_df, config)

    # 3) Load cleaned data from CSV for subsequent steps
    print('Loading cleaned data from CSV...')
    cleaned_csv_path = reports_dir / "rpe_extracted_features_cleaned.csv"
    try:
        cleaned_df = pd.read_csv(cleaned_csv_path)
        print(f"Loaded cleaned data from {cleaned_csv_path} (n_samples={len(cleaned_df)})")
    except Exception as exc:
        raise RuntimeError(f"Failed to load cleaned CSV: {exc}") from exc

    # Extract features and labels from cleaned CSV
    target_column = config.get('analysis_params', {}).get('target_column', 'label')
    if target_column not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in cleaned CSV")

    features_for_modeling = cleaned_df.drop(columns=[target_column])
    labels_for_modeling = cleaned_df[target_column].astype(str)

    # 4) PCA for visualization
    print('Running PCA analysis...')
    pca, features_pca = run_pca(features_for_modeling, config.get('analysis_params', {}))
    # pass the project output root to plotting/analysis helpers
    plot_pca_scree(pca.explained_variance_ratio_, str(out_root))
    plot_pca_cumulative(pca.explained_variance_ratio_, str(out_root))
    if features_pca.shape[1] >= 2:
        plot_pca_scatter(features_pca.values, labels_for_modeling, str(out_root))

    # 5) Training
    print('Training stacking classifier...')
    model, training_metrics = train_stacking(features_for_modeling, labels_for_modeling, config.get('analysis_params', {}))
    # Do not print the training report to terminal; it will be saved in the JSON report

    # 6) Save artifacts and reports
    print('Saving artifacts and reports...')
    save_artifacts(preproc, pca, model, str(out_root))
    save_classification_report(labels_for_modeling, model.predict(features_for_modeling), str(out_root), training_metrics=training_metrics)

    # 7) Extract and save feature importances
    print('Extracting feature importances...')
    extract_and_save_feature_importances(model, list(features_for_modeling.columns), str(out_root), config)

    # 8) Feature insights analysis
    print('Running feature insights analysis...')
    analyze_top_features(str(out_root), config)

# Note: config file path is provided explicitly via CLI or falls back to the
# workspace absolute path supplied by the user. We intentionally do not search
# multiple locations to keep behavior deterministic.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPE image analysis pipeline")
    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        default=None,
        help="Path to config.json (optional, will search automatically if not provided)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    # Automatically detect config.json if not provided
    if args.config_path is None:
        config_file_path = None
        for current_path in (Path.cwd(), *Path.cwd().parents):
            if (current_path / 'config.json').exists():
                config_file_path = current_path / 'config.json'
                break
        if config_file_path is None:
            print("Error: config.json not found in current directory or parent directories")
            sys.exit(2)
    else:
        config_file_path = Path(args.config_path).resolve()
        if not config_file_path.is_file():
            print(f"Error: config.json not found at {config_file_path}")
            sys.exit(2)

    try:
        with open(config_file_path, 'r') as file_handle:
            config = json.load(file_handle)
    except Exception as exc:
        print(f'Error: failed to load config: {exc}')
        traceback.print_exc()
        sys.exit(2)

    try:
        # measure total pipeline execution time
        from scripts.timer import Timer
        with Timer('total_pipeline'):
            run_from_config(config, verbose=args.verbose)
    except Exception as exc:
        print(f'Pipeline failed: {exc}')
        traceback.print_exc()
        sys.exit(1)

