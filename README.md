# Morphological Feature Analysis of Retinal Pigment Epithelial Cells from C57BL/6J Mice during Aging

A configuration-driven Python application for extracting, aggregating, and analyzing features from Retinal Pigment Epithelium (RPE) image crops. The pipeline produces cleaned feature tables, PCA visualizations, supervised classification (stacking ensemble), feature importance analysis, and reproducible model artifacts.

The codebase is written in clean, maintainable Python following PEP 8 standards, with comprehensive docstrings, type hints, and robust error handling.

This README provides a concise project overview, installation and usage instructions, configuration notes, and a quick reference for the included scripts and outputs.

## Project Summary

- **Inputs**: Labeled folders of TIFF crops (per-image) or precomputed feature CSVs.
- **Pipeline**: 8-step automated workflow (extraction → cleaning → CSV loading → PCA → training → feature importance → violin plots → artifacts).
- **Features**: Morphology, intensity stats, texture (LBP, GLCM-like), Gabor responses, and spatial measures (centroids, nearest neighbor, density).
- **Processing**: Aggregates per-region measurements into per-image summaries (count, mean, median, std, min, max).
- **Modeling**: Cleans data (inf→NaN), drops columns with excessive missingness, imputes (median), scales, runs PCA, trains a stacking classifier (RandomForest + XGBoost/HistGradientBoosting, logistic meta-learner), extracts feature importances, and generates advanced feature insights.
- **Outputs**: Cleaned/raw features CSV, PCA plots, confusion matrix, JSON report, feature importance CSV and plot, feature insights plots, and a saved model bundle for reproducible prediction.

## Quick Start (Windows PowerShell)

1. From the project folder, create and activate a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Edit `config.json` to set `paths.image_directory`, `paths.output_directory`, and any feature or analysis parameters.
3. Run the full pipeline (verbose):

```powershell
py -3 -u .\main.py -c .\config.json -v
```

This will run the complete 8-step pipeline: extract features, clean and preprocess them, load cleaned data from CSV, perform PCA, train the model, extract feature importances, create violin plots, and save all outputs to the configured `output_directory` (organized into `reports/`, `models/`, and `plots/` subfolders).

## Pipeline Overview

The automated pipeline consists of 8 steps:

1. **Feature Extraction**: Extracts morphological, texture, and intensity features from TIFF images.
2. **Clean & Prepare**: Cleans data, handles missing values, imputes, scales, and saves cleaned CSV.
3. **Load Cleaned Data**: Loads the cleaned CSV to ensure consistency for all modeling steps.
4. **PCA Analysis**: Performs PCA and generates visualization plots.
5. **Model Training**: Trains stacking classifier (RandomForest + XGBoost/HistGradientBoosting).
6. **Feature Importance**: Extracts and visualizes top feature importances from the trained model.
7. **Feature Insights**: Creates violin plots with trendlines for top features across age groups.
8. **Save Artifacts**: Saves model bundle, preprocessing objects, reports, and plots.

## Configuration

All runtime parameters are in `config.json` next to `main.py`. Important keys:

- `paths.image_directory` — Folder of labeled subfolders containing TIFF crops.
- `paths.output_directory` — Where CSVs, plots, and models are written.
- `paths.output_features_csv` — Path for the extracted features CSV.
- `feature_params` — Controls feature extraction (LBP points/radius, GLCM levels, `gabor_frequencies`, `gabor_angles`, `use_otsu`, channel thresholds).
- `feature_insights_params` — Controls `top_features_count` for number of features to analyze

Modify these values to tune extraction and modeling without editing code.

## Files and Scripts

- `main.py` — **Main entrypoint**. Orchestrates the complete 8-step pipeline.
- `config.json` — Runtime configuration (paths, feature params, analysis params).
- `requirements.txt` — Python dependencies.
- `scripts/` — Modular scripts for specific functionality:
  - `analysis.py` — Data cleaning, PCA, and visualization.
  - `clean_and_prepare.py` — Feature cleaning and preprocessing pipeline.
  - `feature_extraction.py` — Feature extraction from TIFF images.
  - `feature_insights.py` — Feature importance extraction and violin plot generation with trendlines.
  - `train_and_save_pipeline.py` — Model training, artifact saving, and classification reports.
  - `load_model_bundle_and_predict.py` — Prediction on new data using saved models.
  - `timer.py` — Timing utilities for performance measurement.
  - `visualize_channels.py` — Channel visualization utilities.

All scripts follow PEP 8 standards with type hints, comprehensive docstrings, and robust error handling.

## Outputs (Default Location: `analysis_results/`, Organized into `reports/`, `models/`, and `plots/`)

- `rpe_extracted_features.csv` — Raw extracted features (one row per image).
- `rpe_extracted_features_cleaned.csv` — Cleaned + scaled features.
- `pca_scree_plot.png`, `pca_cumulative_plot.png`, `pca_scatter_plot.png` — PCA visualizations.
- `confusion_matrix.png` — Confusion matrix for classifier predictions.
- `classification_report.json` — Classification metrics from cross-validated predictions.
- `feature_importances.csv` — Feature importances from the RandomForest base learner.
- `feature_importances.png` — Bar plot of top N feature importances (configurable).
- `top_features_violin_plots.png` — Violin plots with trendlines comparing distributions of top features across age groups.
- `model.joblib`, `pca.joblib`, `preprocessor.joblib` — Saved model and preprocessing artifacts.

## Predicting on New Data

To predict on new data using the saved model:

```powershell
py -3 scripts\load_model_bundle_and_predict.py --features "analysis_results\reports\rpe_extracted_features_cleaned.csv" --model "analysis_results\models\model.joblib" --output "analysis_results\reports\predictions.csv"
```

## Troubleshooting & Tips

- **Error Handling**: The code includes robust error handling for file I/O, image processing, and data operations.
- **Configuration**: Ensure `paths.image_directory` points to the correct folder with readable TIFF files.
- **Data Quality**: If many columns are dropped, lower `analysis_params.max_nan_fraction` or inspect the features CSV for missing values.
- **Dependencies**: If XGBoost is unavailable, the pipeline falls back to `HistGradientBoostingClassifier`.
- **Code Quality**: All scripts follow PEP 8 with type hints and docstrings for maintainability.

## Recent Updates

- **Modular Refactoring**: Separated analysis and training logic into dedicated modules for better maintainability.
- **Feature Importance**: Added extraction and visualization of feature importances from RandomForest base learners.
- **Dynamic Violin Plots**: Implemented configurable violin plots with trendlines for top N features (configurable via `feature_insights_params.top_features_count`).
- **Automatic Grid Layout**: Violin plots automatically adjust grid size (1×2, 2×2, 2×3, 3×3, or 4×4) based on number of features.
- **Configurable Analysis**: Made top features count fully configurable through `feature_insights_params.top_features_count`.
- **Enhanced Visualizations**: Added trendlines with R² statistics to violin plots for comprehensive feature exploration.
- **Automatic Config Detection**: Searches for `config.json` in the project root and parent directories.
- **Code Cleanup**: Removed unused functions and cleaned up documentation for better maintainability.

## License

MIT
