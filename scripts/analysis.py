"""
Analysis module for RPE image data processing and modeling.

This module provides functions for data cleaning, PCA, model training,
artifact saving, and visualization of RPE cell features.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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


def clean_and_prepare(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Clean and prepare the features DataFrame for modeling.
    Includes saving cleaned CSV and preprocessing bundle.

    Args:
        df: Raw features DataFrame with target column.
        config: Configuration dictionary.

    Returns:
        Tuple of (cleaned features, target series), preprocessing pipeline.
    """
    df = df.copy()
    # support either full config or just analysis_params dict
    if config is None:
        cfg = {}
    elif 'paths' in config or 'feature_params' in config:
        cfg = config.get('analysis_params', {})
    else:
        cfg = config

    target = cfg.get('target_column', 'label')
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target].astype(str)

    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Report NaN counts
    nan_counts = X.isna().sum()
    nan_frac = nan_counts / len(X)
    print("Top columns by NaN fraction (showing >0):")
    print(nan_frac[nan_frac > 0].sort_values(ascending=False).head(30).to_string())

    col_thresh = float(cfg.get('drop_column_na_threshold', 0.5))
    keep = X.columns[X.isna().mean() < col_thresh]
    dropped = X.columns[X.isna().mean() >= col_thresh].tolist()
    if dropped:
        print(f"Dropping {len(dropped)} columns with NaN fraction > {col_thresh}: {dropped}")
        X = X[keep]
    else:
        print("No columns to drop based on NaN fraction threshold.")

    imputer = SimpleImputer(strategy=cfg.get('impute_strategy', 'median'))
    scaler = StandardScaler()
    pipe = Pipeline([('imputer', imputer), ('scaler', scaler)])

    X_trans = pipe.fit_transform(X)
    X_clean = pd.DataFrame(X_trans, columns=keep, index=X.index)

    # Save cleaned CSV and bundle
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    cleaned_csv = reports_dir / "rpe_extracted_features_cleaned.csv"
    prep_joblib = models_dir / "feature_preprocessing_bundle.joblib"

    features_clean = X_clean.copy()
    features_clean[target] = y
    features_clean.to_csv(cleaned_csv, index=False)
    print(f"Saved cleaned features to {cleaned_csv}")

    bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "dropped_columns": dropped,
        "feature_columns": keep.tolist(),
    }
    joblib.dump(bundle, prep_joblib)
    print(f"Saved preprocessing bundle to {prep_joblib}")

    return (X_clean, y), pipe


def resolve_output_dirs(base: Path, cfg_path: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    """Resolve and create output directories (out_root, reports, models, plots).

    base: project root (usually Path(__file__).resolve().parents[0])
    cfg_path: optional path to config.json (if None, will check base/config.json)
    """
    # Determine project root by searching upwards for config.json; fall back to provided base
    project_root: Optional[Path] = None
    for p in (base, *base.parents):
        if (p / 'config.json').exists():
            project_root = p
            break
    if project_root is None:
        project_root = base

    cfg_fp = cfg_path or (project_root / 'config.json')
    if cfg_fp.exists():
        try:
            with open(cfg_fp, 'r') as fh:
                cfg = json.load(fh)
            out_root = Path(cfg.get('paths', {}).get('output_directory', project_root / 'analysis_results'))
        except Exception:
            out_root = project_root / 'analysis_results'
    else:
        out_root = project_root / 'analysis_results'

    reports = out_root / 'reports'
    models = out_root / 'models'
    plots = out_root / 'plots'
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return out_root, reports, models, plots


def run_pca(X: pd.DataFrame, config: dict) -> Tuple[PCA, pd.DataFrame]:
    """
    Perform PCA on the features.

    Args:
        X: Cleaned features DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of PCA object and transformed features DataFrame.
    """
    n_components = int(config.get('pca_components', min(10, X.shape[1])))
    pca = PCA(n_components=n_components)
    X_p = pca.fit_transform(X)
    cols = [f'pc{i+1}' for i in range(X_p.shape[1])]
    return pca, pd.DataFrame(X_p, columns=cols)


def train_stacking(X: pd.DataFrame, y: pd.Series, config: dict) -> Tuple[object, dict]:
    """
    Train a stacking classifier.

    Args:
        X: Features DataFrame.
        y: Target series.
        config: Configuration dictionary.

    Returns:
        Tuple of trained model and training metrics.
    """
    rf = RandomForestClassifier(n_estimators=int(config.get('rf_n_estimators', 100)), random_state=42)
    estimators = [('rf', rf)]
    final_est = RandomForestClassifier(n_estimators=int(config.get('final_n_estimators', 200)), random_state=42)
    stack = StackingClassifier(estimators=estimators, final_estimator=final_est, cv=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config.get('test_size', 0.2)), random_state=42, stratify=y)
    stack.fit(X_train, y_train)

    scores = cross_val_score(stack, X_train, y_train, cv=5)
    report = {
        'cv_mean_score': float(np.mean(scores)),
        'cv_std_score': float(np.std(scores)),
        'test_score': float(stack.score(X_test, y_test)),
    }

    return stack, report


def save_artifacts(pipe, pca, model, out_dir: str):
    """
    Save preprocessing pipeline, PCA, and model to disk.

    Args:
        pipe: Preprocessing pipeline.
        pca: PCA object.
        model: Trained model.
        out_dir: Output directory path.
    """
    # Save preprocessor, pca, and model into a dedicated models directory
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    # if a user provided a custom out_dir string path, prefer it
    try:
        user_out = Path(out_dir)
        if user_out.exists() and user_out.is_dir():
            out_root = user_out
            reports_dir = out_root / 'reports'
            models_dir = out_root / 'models'
            plots_dir = out_root / 'plots'
            reports_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    joblib.dump(pipe, models_dir / 'preprocessor.joblib')
    joblib.dump(pca, models_dir / 'pca.joblib')
    joblib.dump(model, models_dir / 'model.joblib')


def plot_pca_scree(explained_variance_ratio: np.ndarray, out_dir: str) -> None:
    """
    Plot PCA scree plot with elbow detection.

    Args:
        explained_variance_ratio: PCA explained variance ratios.
        out_dir: Output directory path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    try:
        kneedle = KneeLocator(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, curve="convex", direction="decreasing")
        if kneedle.knee:
            plt.vlines(kneedle.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r', label=f'Elbow at n_components={kneedle.knee}')
            plt.legend()
    except Exception:
        pass
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    try:
        # if user supplied out_dir string, prefer that
        user_out = Path(out_dir)
        if user_out.exists() and user_out.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_out)
    except Exception:
        pass
    plt.savefig(plots_dir / 'pca_scree_plot.png')
    plt.close()


def plot_pca_cumulative(explained_variance_ratio: np.ndarray, out_dir: str) -> None:
    """
    Plot cumulative explained variance.

    Args:
        explained_variance_ratio: PCA explained variance ratios.
        out_dir: Output directory path.
    """
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    try:
        user_out = Path(out_dir)
        if user_out.exists() and user_out.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_out)
    except Exception:
        pass
    plt.savefig(plots_dir / 'pca_cumulative_plot.png')
    plt.close()


def plot_pca_scatter(pca_features: np.ndarray, labels: pd.Series, out_dir: str) -> None:
    """Plot PCA features. Use 3D scatter when 3+ components are available, otherwise 2D.

    Saves pca_scatter_3d.png (when 3D) or pca_scatter_plot.png (2D) into <out_dir>/plots.
    """
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    try:
        user_out = Path(out_dir)
        if user_out.exists() and user_out.is_dir():
            _, _, _, plots_dir = resolve_output_dirs(user_out)
    except Exception:
        pass

    pf = np.asarray(pca_features)
    unique_labels = labels.unique()
    cmap = matplotlib.cm.get_cmap('viridis')

    # 3D scatter if at least 3 components
    if pf.ndim == 2 and pf.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(unique_labels):
            idx = labels == label
            ax.scatter(pf[idx, 0], pf[idx, 1], pf[idx, 2], color=cmap(i / max(1, len(unique_labels) - 1)), label=label, alpha=0.7)
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
            idx = labels == label
            plt.scatter(pf[idx, 0], pf[idx, 1], color=cmap(i / max(1, len(unique_labels) - 1)), label=label, alpha=0.7)
        plt.title('2D PCA of RPE Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pca_scatter_plot.png')
        plt.close()


def save_classification_report(y_true, y_pred, out_dir: str, training_metrics: dict = None):
    """Save classification report (and optionally include training metrics).

    training_metrics should be a dict containing keys like 'cv_mean_score', 'cv_std_score', 'test_score'.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    if training_metrics:
        # merge under a top-level key so it doesn't collide with class metrics
        report['_training_metrics'] = training_metrics
    base = Path(__file__).resolve().parent
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)
    try:
        user_out = Path(out_dir)
        if user_out.exists() and user_out.is_dir():
            out_root = user_out
            reports_dir = out_root / 'reports'
            plots_dir = out_root / 'plots'
            reports_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    with open(reports_dir / 'classification_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)

    # confusion matrix (store as a plot)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 6))
    disp.plot(values_format='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(plots_dir / 'confusion_matrix.png')
    plt.close()
