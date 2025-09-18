"""
Feature Insights Analysis Module

This module provides functionality for extracting and visualizing feature importances
from trained models, and generating advanced feature insights with violin plots
and trendline analysis.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scripts.analysis import resolve_output_dirs


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
    print(f"Saved feature importances to {reports_dir / 'feature_importances.csv'}")

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


def create_violin_plots(output_directory: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Create violin plots for top features in a dynamic grid layout based on top_features_count.

    The number of features plotted is determined by the 'top_features_count' parameter
    in the config file under 'feature_insights_params'. The subplot grid automatically
    adjusts to accommodate the number of features (1x2, 2x2, 2x3, 3x3, or 4x4 max).

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

    # Load cleaned features data
    cleaned_features_path = reports_dir / 'rpe_extracted_features_cleaned.csv'
    if not cleaned_features_path.exists():
        print(f"Cleaned features file not found: {cleaned_features_path}")
        return

    features_df = pd.read_csv(cleaned_features_path)

    # Assume there's a 'label' column for grouping
    if 'label' not in features_df.columns:
        print("No 'label' column found in features data. Cannot create violin plots.")
        return

    # Get top features count from config, default to 4 for violin plots
    top_features_count = config.get('feature_insights_params', {}).get('top_features_count', 4) if config else 4

    # Load top important features from feature importances
    feature_importances_path = reports_dir / 'feature_importances.csv'
    if feature_importances_path.exists():
        feature_importances_df = pd.read_csv(feature_importances_path)
        features_to_plot = feature_importances_df.head(top_features_count)['feature'].tolist()
    else:
        # Fallback to default features if importances file doesn't exist
        features_to_plot = ['lbp_9', 'minor_axis_median', 'circularity_median', 'solidity_mean']
        features_to_plot = features_to_plot[:top_features_count]  # Limit to top_features_count

    # Ensure we don't exceed available features
    available_features = [f for f in features_to_plot if f in features_df.columns]
    features_to_plot = available_features[:top_features_count]

    # Create dynamic subplot grid based on number of features to plot
    num_features = len(features_to_plot)
    if num_features == 0:
        print("No valid features found for plotting.")
        return

    # Calculate grid dimensions
    if num_features <= 2:
        nrows, ncols = 1, num_features
    elif num_features <= 4:
        nrows, ncols = 2, 2
    elif num_features <= 6:
        nrows, ncols = 2, 3
    elif num_features <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 4  # Cap at 4x4 grid for readability

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    
    # Handle axes properly for different numbers of subplots
    if num_features == 1:
        axes = [axes]  # Single subplot returns a single Axes object
    elif hasattr(axes, 'flatten'):
        axes = axes.flatten()  # Multi-dimensional array
    else:
        axes = [axes]  # Fallback for any other case

    # Set style
    sns.set_style("whitegrid")

    for i, feature in enumerate(features_to_plot):
        if i >= len(axes):
            break  # Don't exceed available subplots
            
        ax = axes[i]

        if feature in features_df.columns:
            # Create violin plot
            sns.violinplot(data=features_df, x='label', y=feature, ax=ax, hue='label', palette='Set2', legend=False)

            # Add trendline
            unique_labels = sorted(features_df['label'].unique())
            label_map = {label: i for i, label in enumerate(unique_labels)}
            x_vals = [label_map[label] for label in features_df['label']]
            y_vals = features_df[feature]

            if len(set(x_vals)) > 1:  # Need at least 2 unique x values for trendline
                slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                x_trend = [min(x_vals), max(x_vals)]
                y_trend = [slope * xi + intercept for xi in x_trend]

                # Plot trendline
                ax.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2,
                       label=f'Trend (R²={r_value**2:.2f})')

            # Customize plot
            ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Age Group', fontsize=10)
            ax.set_ylabel(feature.replace("_", " ").title(), fontsize=10)
            ax.tick_params(axis='x', rotation=45)

            # Add legend for trendline
            ax.legend()

            # Add grid
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Feature {feature} not found',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{feature.replace("_", " ").title()} (Not Available)', fontsize=12)

    # Hide unused subplots if any
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    violin_plot_path = plots_dir / 'top_features_violin_plots.png'
    plt.savefig(violin_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved violin plots to {violin_plot_path}")
