"""
Analysis module for RPE image data processing.

This module provides functions for data cleaning, PCA, visualization,
and feature importance analysis of RPE cell features.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    # Handle pca_components: if None or not specified, use number of features
    pca_components = config.get('pca_components')
    if pca_components is None:
        num_components = features.shape[1]
    else:
        num_components = int(pca_components)
    
    pca_model = PCA(n_components=num_components)
    features_pca = pca_model.fit_transform(features)
    column_names = [f'pc{i+1}' for i in range(features_pca.shape[1])]
    return pca_model, pd.DataFrame(features_pca, columns=column_names)


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

