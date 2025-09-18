"""
Clean and prepare module for RPE image data preprocessing.

This module provides the clean_and_prepare function for data cleaning and preprocessing.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from scripts.analysis import resolve_output_dirs


def clean_and_prepare(features_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Tuple[pd.DataFrame, pd.Series], Pipeline]:
    """
    Clean and prepare the features DataFrame for modeling.
    Includes saving cleaned CSV and preprocessing bundle.

    Args:
        features_df: Raw features DataFrame with target column.
        config: Configuration dictionary.

    Returns:
        Tuple of ((cleaned features, target series), preprocessing pipeline).
    """
    features_df = features_df.copy()
    # support either full config or just analysis_params dict
    if config is None:
        analysis_config = {}
    elif 'paths' in config or 'feature_params' in config:
        analysis_config = config.get('analysis_params', {})
    else:
        analysis_config = config

    target_column = analysis_config.get('target_column', 'label')
    if target_column not in features_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    features = features_df.drop(columns=[target_column])
    labels = features_df[target_column].astype(str)

    features = features.select_dtypes(include=[np.number])
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Report NaN counts
    nan_counts = features.isna().sum()
    nan_fraction = nan_counts / len(features)
    print("Top columns by NaN fraction (showing >0):")
    print(nan_fraction[nan_fraction > 0].sort_values(ascending=False).head(30).to_string())

    column_threshold = float(analysis_config.get('max_nan_fraction', 0.5))
    columns_to_keep = features.columns[features.isna().mean() < column_threshold]
    dropped_columns = features.columns[features.isna().mean() >= column_threshold].tolist()
    if dropped_columns:
        print(f"Dropping {len(dropped_columns)} columns with NaN fraction > {column_threshold}: {dropped_columns}")
        features = features[columns_to_keep]
    else:
        print("No columns to drop based on NaN fraction threshold.")

    imputer = SimpleImputer(strategy=analysis_config.get('impute_strategy', 'median'))
    scaler = StandardScaler()
    preprocessing_pipeline = Pipeline([('imputer', imputer), ('scaler', scaler)])

    features_transformed = preprocessing_pipeline.fit_transform(features)
    features_clean = pd.DataFrame(features_transformed, columns=columns_to_keep, index=features.index)

    # Save cleaned CSV and bundle
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    cleaned_csv_path = reports_dir / "rpe_extracted_features_cleaned.csv"
    preprocessing_bundle_path = models_dir / "feature_preprocessing_bundle.joblib"

    features_clean_with_labels = features_clean.copy()
    features_clean_with_labels[target_column] = labels
    try:
        features_clean_with_labels.to_csv(cleaned_csv_path, index=False)
        print(f"Saved cleaned features to {cleaned_csv_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to save cleaned CSV: {exc}") from exc

    preprocessing_bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "dropped_columns": dropped_columns,
        "feature_columns": columns_to_keep.tolist(),
    }
    try:
        joblib.dump(preprocessing_bundle, preprocessing_bundle_path)
        print(f"Saved preprocessing bundle to {preprocessing_bundle_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to save preprocessing bundle: {exc}") from exc

    return (features_clean, labels), preprocessing_pipeline