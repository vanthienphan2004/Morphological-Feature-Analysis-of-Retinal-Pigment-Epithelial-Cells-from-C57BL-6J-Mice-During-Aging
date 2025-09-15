"""
Analysis module for RPE image data processing and modeling.

This module provides functions for data cleaning, PCA, model training,
artifact saving, and visualization of RPE cell features.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from kneed import KneeLocator


def resolve_output_dirs(base: Path, config_path: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    """
    Resolve and create output directories (out_root, reports, models, plots).

    Args:
        base: Project root (usually Path(__file__).resolve().parents[0]).
        config_path: Optional path to config.json (if None, will check base/config.json).

    Returns:
        Tuple of (out_root, reports_dir, models_dir, plots_dir).
    """
    # Determine project root by searching upwards for config.json; fall back to provided base
    project_root: Optional[Path] = None
    for current_path in (base, *base.parents):
        if (current_path / 'config.json').exists():
            project_root = current_path
            break
    if project_root is None:
        project_root = base

    config_file_path = config_path or (project_root / 'config.json')
    if config_file_path.exists():
        try:
            with open(config_file_path, 'r') as file_handle:
                config_data = json.load(file_handle)
            output_root = Path(config_data.get('paths', {}).get('output_directory', project_root / 'analysis_results'))
        except Exception:
            output_root = project_root / 'analysis_results'
    else:
        output_root = project_root / 'analysis_results'

    reports_dir = output_root / 'reports'
    models_dir = output_root / 'models'
    plots_dir = output_root / 'plots'
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_root, reports_dir, models_dir, plots_dir


def run_pca(features: pd.DataFrame, config: Dict[str, Any]) -> Tuple[PCA, pd.DataFrame]:
    """
    Perform PCA on the features.

    Args:
        features: Cleaned features DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of PCA object and transformed features DataFrame.
    """
    num_components = int(config.get('pca_components', min(10, features.shape[1])))
    pca_model = PCA(n_components=num_components)
    features_pca = pca_model.fit_transform(features)
    column_names = [f'pc{i+1}' for i in range(features_pca.shape[1])]
    return pca_model, pd.DataFrame(features_pca, columns=column_names)


def train_stacking(features: pd.DataFrame, labels: pd.Series, config: Dict[str, Any]) -> Tuple[object, Dict[str, float]]:
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


def plot_pca_scree(explained_variance_ratio: np.ndarray, output_directory: str) -> None:
    """
    Plot PCA scree plot with elbow detection.

    Args:
        explained_variance_ratio: PCA explained variance ratios.
        output_directory: Output directory path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
        marker='o',
        linestyle='--'
    )
    try:
        knee_locator = KneeLocator(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            curve="convex",
            direction="decreasing"
        )
        if knee_locator.knee:
            plt.vlines(
                knee_locator.knee,
                plt.ylim()[0],
                plt.ylim()[1],
                linestyles='dashed',
                colors='r',
                label=f'Elbow at n_components={knee_locator.knee}'
            )
            plt.legend()
    except Exception:
        pass
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    try:
        # if user supplied output_directory string, prefer that
        user_output_dir = Path(output_directory)
        if user_output_dir.exists() and user_output_dir.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_output_dir)
    except Exception:
        pass
    plt.savefig(plots_dir / 'pca_scree_plot.png')
    plt.close()


def plot_pca_cumulative(explained_variance_ratio: np.ndarray, output_directory: str) -> None:
    """
    Plot cumulative explained variance.

    Args:
        explained_variance_ratio: PCA explained variance ratios.
        output_directory: Output directory path.
    """
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='.',
        linestyle='-'
    )
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    try:
        user_output_dir = Path(output_directory)
        if user_output_dir.exists() and user_output_dir.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_output_dir)
    except Exception:
        pass
    plt.savefig(plots_dir / 'pca_cumulative_plot.png')
    plt.close()


def plot_pca_scatter(pca_features: np.ndarray, labels: pd.Series, output_directory: str) -> None:
    """
    Plot PCA features. Use 3D scatter when 3+ components are available, otherwise 2D.

    Saves pca_scatter_3d.png (when 3D) or pca_scatter_plot.png (2D) into <output_directory>/plots.

    Args:
        pca_features: PCA-transformed features array.
        labels: Series of labels for coloring.
        output_directory: Output directory path.
    """
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)
    try:
        user_output_dir = Path(output_directory)
        if user_output_dir.exists() and user_output_dir.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_output_dir)
    except Exception:
        pass

    pca_array = np.asarray(pca_features)
    unique_labels = labels.unique()
    colormap = matplotlib.cm.get_cmap('viridis')

    # 3D scatter if at least 3 components
    if pca_array.ndim == 2 and pca_array.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            indices = labels == label
            ax.scatter(
                pca_array[indices, 0],
                pca_array[indices, 1],
                pca_array[indices, 2],
                color=colormap(i / max(1, len(unique_labels) - 1)),
                label=label,
                alpha=0.7
            )
        ax.set_title('3D PCA of RPE Features')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pca_scatter_3d.png')
        plt.close()
    else:
        # fallback 2D scatter
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            indices = labels == label
            plt.scatter(
                pca_array[indices, 0],
                pca_array[indices, 1],
                color=colormap(i / max(1, len(unique_labels) - 1)),
                label=label,
                alpha=0.7
            )
        plt.title('2D PCA of RPE Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pca_scatter_plot.png')
        plt.close()


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
