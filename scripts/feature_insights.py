"""
Feature Insights Analysis Module

This module analyzes the top 10 important features from the trained model,
generating insights and visualizations for each feature.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from analysis import resolve_output_dirs


def extract_and_save_feature_importances(
    trained_model: object,
    feature_names: list,
    output_directory: str,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Extract feature importances from RandomForest base learners in stacking model
    and save to CSV and plot.

    Args:
        trained_model: Trained stacking model.
        feature_names: List of feature names.
        output_directory: Output directory path.
        config: Configuration dictionary (optional).
    """
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

    # Get top features count from config, default to 10
    top_features_count = config.get('feature_insights_params', {}).get('top_features_count', 10) if config else 10

    importances = None
    rf_estimator = None

    # Try to find RandomForest base learner in stacking model
    try:
        if hasattr(trained_model, 'named_estimators_'):
            named_estimators = trained_model.named_estimators_
            # Look for RandomForest in named estimators
            for name, estimator in named_estimators.items():
                if hasattr(estimator, 'feature_importances_'):
                    rf_estimator = estimator
                    break

        # If not found in named_estimators, check estimators_ attribute
        if rf_estimator is None and hasattr(trained_model, 'estimators_'):
            for estimator in trained_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    rf_estimator = estimator
                    break

        # Extract feature importances if RandomForest found
        if rf_estimator is not None and hasattr(rf_estimator, 'feature_importances_'):
            importances = rf_estimator.feature_importances_

    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")

    if importances is None:
        print("No RandomForest feature importances found in stacking model; skipping feature_importances.csv")
        return

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    # Save to CSV
    feature_importance_df.to_csv(reports_dir / 'feature_importances.csv', index=False)
    # print(f"Saved feature importances to {reports_dir / 'feature_importances.csv'}")

    # Create visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(top_features_count)  # Show top N features
    plt.barh(
        range(len(top_features)),
        top_features['importance'],
        align='center'
    )
    plt.yticks(
        range(len(top_features)),
        top_features['feature'],
        fontsize=8
    )
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Top {top_features_count} Feature Importances (RandomForest)')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to {plots_dir / 'feature_importances.png'}")


def categorize_features(features: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into types based on their names.

    Args:
        features: List of feature names.

    Returns:
        Dictionary with categories as keys and lists of features as values.
    """
    categories = {
        'Morphology': [],
        'Texture': [],
        'Intensity': [],
        'Gabor': [],
        'Spatial': [],
        'Other': []
    }
    
    for f in features:
        f_lower = f.lower()
        if any(k in f_lower for k in ['area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'convex']):
            categories['Morphology'].append(f)
        elif any(k in f_lower for k in ['lbp', 'glcm', 'haralick', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']):
            categories['Texture'].append(f)
        elif any(k in f_lower for k in ['intensity', 'mean', 'std', 'min', 'max', 'median', 'brightness']):
            categories['Intensity'].append(f)
        elif 'gabor' in f_lower:
            categories['Gabor'].append(f)
        elif any(k in f_lower for k in ['centroid', 'distance', 'density', 'nearest', 'cluster']):
            categories['Spatial'].append(f)
        else:
            categories['Other'].append(f)
    
    return {k: v for k, v in categories.items() if v}


def analyze_top_features(output_directory: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Analyze and visualize the top N important features with scatter plots and trendlines.

    Args:
        output_directory: Optional output directory path.
        config: Configuration dictionary (optional).
    """
    base_path = Path(__file__).resolve().parent
    output_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base_path)

    if output_directory:
        try:
            user_output_dir = Path(output_directory)
            if user_output_dir.exists() and user_output_dir.is_dir():
                output_root = user_output_dir
                reports_dir = output_root / 'reports'
                plots_dir = output_root / 'plots'
        except Exception:
            pass

    # Get top features count from config, default to 10
    top_features_count = config.get('feature_insights_params', {}).get('top_features_count', 10) if config else 10

    # Load feature importances
    feature_importances_path = reports_dir / 'feature_importances.csv'
    if not feature_importances_path.exists():
        print(f"Feature importances file not found: {feature_importances_path}")
        return

    feature_importances_df = pd.read_csv(feature_importances_path)
    top_features = feature_importances_df.head(top_features_count)['feature'].tolist()

    # Load cleaned features data
    cleaned_features_path = reports_dir / 'rpe_extracted_features_cleaned.csv'
    if not cleaned_features_path.exists():
        print(f"Cleaned features file not found: {cleaned_features_path}")
        return

    features_df = pd.read_csv(cleaned_features_path)

    # Assume there's a 'label' column for grouping
    if 'label' not in features_df.columns:
        print("No 'label' column found in features data. Cannot create grouped plots.")
        return

    # Categorize features
    categories = categorize_features(top_features)
    
    # Create a combined plot
    num_categories = len(categories)
    if num_categories == 0:
        print("No features to plot.")
        return
    
    fig, axes = plt.subplots(num_categories, 1, figsize=(12, 6 * num_categories))
    if num_categories == 1:
        axes = [axes]
    
    # Map labels to numbers for x-axis
    unique_labels = sorted(features_df['label'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors
    
    for idx, (category, feats) in enumerate(categories.items()):
        ax = axes[idx]
        ax.set_title(f'{category} Features')
        ax.set_xlabel('Label')
        ax.set_ylabel('Feature Value')
        
        for j, feature in enumerate(feats):
            if feature not in features_df.columns:
                continue
            
            # Prepare data
            x = [label_map[label] for label in features_df['label']]
            y = features_df[feature]
            
            # Scatter plot
            color = colors[j % len(colors)]
            ax.scatter(x, y, alpha=0.6, color=color, label=feature)
            
            # Linear trendline
            if len(set(x)) > 1:  # Need at least 2 unique x values
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                x_trend = [min(x), max(x)]
                y_trend = [slope * xi + intercept for xi in x_trend]
                ax.plot(x_trend, y_trend, color=color, linestyle='--', linewidth=2, 
                       label=f'{feature} trend (R²={r_value**2:.2f})')
        
        ax.set_xticks(list(label_map.values()))
        ax.set_xticklabels(list(label_map.keys()))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot_path = plots_dir / f'top_{top_features_count}_features_combined_insights.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # print(f"Saved combined insights plot to {combined_plot_path}")

    print(f"\nFeature insights analysis completed. Combined plot saved to {plots_dir}")


if __name__ == "__main__":
    analyze_top_features()
