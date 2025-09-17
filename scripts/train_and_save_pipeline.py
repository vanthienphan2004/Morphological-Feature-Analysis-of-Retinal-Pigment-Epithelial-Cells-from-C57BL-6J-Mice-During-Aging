"""
Training and pipeline saving module for RPE image data.

This module provides functions for training machine learning models,
saving trained artifacts, and generating classification reports.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json


def train_stacking(features: pd.DataFrame, labels: pd.Series, config: Dict[str, Any]) -> tuple[object, Dict[str, float]]:
    """
    Train a stacking classifier.

    Args:
        features: Features DataFrame.
        labels: Target series.
        config: Configuration dictionary.

    Returns:
        Tuple of trained model and training metrics.
    """
    rf_estimator = RandomForestClassifier(
        n_estimators=int(config.get('rf_n_estimators', 100)),
        random_state=42
    )
    estimators_list = [('rf', rf_estimator)]
    final_estimator = RandomForestClassifier(
        n_estimators=int(config.get('final_n_estimators', 200)),
        random_state=42
    )
    stacking_model = StackingClassifier(
        estimators=estimators_list,
        final_estimator=final_estimator,
        cv=5
    )

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels,
        test_size=float(config.get('test_size', 0.2)),
        random_state=42,
        stratify=labels
    )
    stacking_model.fit(features_train, labels_train)

    cross_val_scores = cross_val_score(stacking_model, features_train, labels_train, cv=5)
    training_report = {
        'cv_mean_score': float(np.mean(cross_val_scores)),
        'cv_std_score': float(np.std(cross_val_scores)),
        'test_score': float(stacking_model.score(features_test, labels_test)),
    }

    return stacking_model, training_report


def save_artifacts(pipeline: Pipeline, pca_model: PCA, trained_model: object, output_directory: str) -> None:
    """
    Save preprocessing pipeline, PCA, and model to disk.

    Args:
        pipeline: Preprocessing pipeline.
        pca_model: PCA object.
        trained_model: Trained model.
        output_directory: Output directory path.
    """
    # Import here to avoid circular imports
    from scripts.analysis import resolve_output_dirs

    # Save preprocessor, pca, and model into a dedicated models directory
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    # if a user provided a custom out_dir string path, prefer it
    try:
        user_output_dir = Path(output_directory)
        if user_output_dir.exists() and user_output_dir.is_dir():
            output_root = user_output_dir
            reports_dir = output_root / 'reports'
            models_dir = output_root / 'models'
            plots_dir = output_root / 'plots'
            reports_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    joblib.dump(pipeline, models_dir / 'preprocessor.joblib')
    joblib.dump(pca_model, models_dir / 'pca.joblib')
    joblib.dump(trained_model, models_dir / 'model.joblib')


def save_classification_report(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    output_directory: str,
    training_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save classification report (and optionally include training metrics).

    Args:
        true_labels: True labels.
        predicted_labels: Predicted labels.
        output_directory: Output directory path.
        training_metrics: Optional dict with keys like 'cv_mean_score', 'cv_std_score', 'test_score'.
    """
    # Import here to avoid circular imports
    from analysis import resolve_output_dirs

    classification_report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
    if training_metrics:
        # merge under a top-level key so it doesn't collide with class metrics
        classification_report_dict['_training_metrics'] = training_metrics
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    try:
        user_output_dir = Path(output_directory)
        if user_output_dir.exists() and user_output_dir.is_dir():
            output_root = user_output_dir
            reports_dir = output_root / 'reports'
            plots_dir = output_root / 'plots'
            reports_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    with open(reports_dir / 'classification_report.json', 'w') as file_handle:
        json.dump(classification_report_dict, file_handle, indent=2)

    # confusion matrix (store as a plot)
    confusion_matrix_array = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_array)
    plt.figure(figsize=(8, 6))
    display.plot(values_format='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(plots_dir / 'confusion_matrix.png')
    plt.close()
