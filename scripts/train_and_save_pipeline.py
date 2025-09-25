"""
Training and pipeline saving module for RPE image data.

This module provides functions for training machine learning models,
saving trained artifacts, and generating classification reports.
"""

# Standard library imports
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

# Third-party imports
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Boosting library imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def load_config_constants() -> Dict[str, Any]:
    """
    Load configuration constants from the config.json file.

    This function reads the project's config.json file and extracts key parameters
    used throughout the training pipeline. If the config file is not found or
    cannot be parsed, it falls back to sensible default values.

    Returns:
        Dictionary containing configuration constants:
        - random_state: Random seed for reproducibility
        - cv_folds: Number of cross-validation folds
        - test_size: Proportion of data for testing
        - n_estimators: Number of estimators for ensemble models
        - max_depth: Maximum depth for tree-based models (None for unlimited)
    """
    config_path = Path(__file__).resolve().parent.parent / 'config.json'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json: {e}. Using default values.")
        config = {}

    # Extract analysis parameters with defaults
    analysis_params = config.get('analysis_params', {})
    rf_params = analysis_params.get('rf_params', {})

    constants = {
        'random_state': analysis_params.get('random_state', 42),
        'cv_folds': analysis_params.get('cv_folds', 5),
        'test_size': analysis_params.get('test_size', 0.2),
        'n_estimators': rf_params.get('n_estimators', 100),
        'max_depth': rf_params.get('max_depth'),
    }

    return constants


# Load configuration constants
CONFIG_CONSTANTS = load_config_constants()

# Extract individual constants for easy access (loaded from config.json)
DEFAULT_RANDOM_STATE = CONFIG_CONSTANTS['random_state']
DEFAULT_CV_FOLDS = CONFIG_CONSTANTS['cv_folds']
DEFAULT_TEST_SIZE = CONFIG_CONSTANTS['test_size']
DEFAULT_N_ESTIMATORS = CONFIG_CONSTANTS['n_estimators']
DEFAULT_MAX_DEPTH = CONFIG_CONSTANTS['max_depth']


def train_stacking(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Dict[str, Any]
) -> Tuple[object, Dict[str, float]]:
    """
    Train a Boosting_Stacking classifier with XGBoost, LightGBM, and CatBoost base estimators.

    This function creates an ensemble model that combines three powerful gradient boosting
    algorithms with logistic regression as the final meta-learner for improved performance.
    All model parameters are loaded from the config.json file for consistency.

    Args:
        features: Input features DataFrame.
        labels: Target labels Series.
        config: Configuration dictionary containing model parameters.

    Returns:
        Tuple containing:
        - trained_model: The fitted StackingClassifier model
        - training_metrics: Dictionary with cross-validation and test scores

    Raises:
        ValueError: If configuration parameters are invalid.
    """
    # Extract configuration parameters with defaults from config file
    random_state = int(config.get('random_state', DEFAULT_RANDOM_STATE))
    cv_folds = int(config.get('cv_folds', DEFAULT_CV_FOLDS))
    test_size = float(config.get('test_size', DEFAULT_TEST_SIZE))

    # Validate inputs
    if features.empty or labels.empty:
        raise ValueError("Features and labels cannot be empty")
    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length")

    # Initialize base estimators for the stacking ensemble
    base_estimators = [
        ('xgb', XGBClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=random_state,
            verbosity=0
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=random_state,
            verbosity=-1
        )),
        ('catboost', CatBoostClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=random_state,
            verbose=False
        ))
    ]

    # Initialize final estimator
    final_estimator = LogisticRegression(random_state=random_state)

    # Create and configure the stacking model
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv_folds
    )

    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Train the stacking model
    print(f"Training Boosting_Stacking model with {cv_folds}-fold CV...")
    stacking_model.fit(features_train, labels_train)

    # Perform cross-validation on training data
    print("Performing cross-validation...")
    cross_val_scores = cross_val_score(
        stacking_model,
        features_train,
        labels_train,
        cv=cv_folds,
        n_jobs=-1  # Use all available cores
    )

    # Evaluate on test set
    test_score = stacking_model.score(features_test, labels_test)

    # Compile training metrics
    training_metrics = {
        'cv_mean_score': float(np.mean(cross_val_scores)),
        'cv_std_score': float(np.std(cross_val_scores)),
        'test_score': float(test_score),
        'cv_folds': cv_folds,
        'test_size': test_size
    }

    print(".3f")
    return stacking_model, training_metrics


def save_artifacts(
    pipeline: Pipeline,
    trained_model: object,
    output_directory: str
) -> None:
    """
    Save preprocessing pipeline and trained model to disk.

    Args:
        pipeline: Preprocessing pipeline (imputer + scaler).
        trained_model: Trained machine learning model.
        output_directory: Base output directory path.
    """
    # Import here to avoid circular imports
    from scripts.directories import resolve_output_dirs

    # Resolve output directories
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)

    # Handle custom output directory if provided
    if output_directory and output_directory != str(output_root):
        try:
            custom_output_dir = Path(output_directory)
            if custom_output_dir.exists() and custom_output_dir.is_dir():
                output_root = custom_output_dir
                reports_dir = output_root / 'reports'
                models_dir = output_root / 'models'
                plots_dir = output_root / 'plots'

                # Create directories if they don't exist
                reports_dir.mkdir(parents=True, exist_ok=True)
                models_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not use custom output directory: {e}")

    # Save model artifacts
    try:
        joblib.dump(pipeline, models_dir / 'preprocessor.joblib')
        print(f"Saved preprocessor to {models_dir / 'preprocessor.joblib'}")

        joblib.dump(trained_model, models_dir / 'model.joblib')
        print(f"Saved trained model to {models_dir / 'model.joblib'}")

    except Exception as e:
        raise RuntimeError(f"Failed to save model artifacts: {e}") from e


def save_classification_report(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    output_directory: str,
    training_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Generate and save classification report with metrics and confusion matrix visualization.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Model predictions.
        output_directory: Base output directory path.
        training_metrics: Optional training performance metrics to include.
    """
    # Import here to avoid circular imports
    from scripts.directories import resolve_output_dirs

    # Resolve output directories
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)

    # Handle custom output directory if provided
    if output_directory and output_directory != str(output_root):
        try:
            custom_output_dir = Path(output_directory)
            if custom_output_dir.exists() and custom_output_dir.is_dir():
                output_root = custom_output_dir
                reports_dir = output_root / 'reports'
                plots_dir = output_root / 'plots'

                # Create directories if they don't exist
                reports_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not use custom output directory: {e}")

    # Generate classification report
    try:
        classification_report_dict = classification_report(
            true_labels,
            predicted_labels,
            output_dict=True
        )

        # Add training metrics if provided
        if training_metrics:
            classification_report_dict['_training_metrics'] = training_metrics

        # Save classification report as JSON
        report_path = reports_dir / 'classification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(classification_report_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved classification report to {report_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to generate classification report: {e}") from e

    # Generate and save confusion matrix plot
    try:
        confusion_matrix_array = confusion_matrix(true_labels, predicted_labels)
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_array,
            display_labels=sorted(true_labels.unique())
        )

        plt.figure(figsize=(10, 8))
        display.plot(values_format='d', cmap='Blues', ax=plt.gca())
        plt.title('Confusion Matrix - RPE Cell Classification', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save confusion matrix plot
        cm_path = plots_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to {cm_path}")

    except Exception as e:
        print(f"Warning: Could not generate confusion matrix plot: {e}")
        plt.close()
