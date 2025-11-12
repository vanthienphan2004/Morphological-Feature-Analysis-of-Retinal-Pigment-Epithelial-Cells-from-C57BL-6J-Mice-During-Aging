# Morphological Feature Analysis of Retinal Pigment Epithelial Cells from C57BL/6J Mice during Aging

A configuration-driven Python application for extracting, aggregating, and analyzing morphological features from Retinal Pigment Epithelium (RPE) image crops. The pipeline produces cleaned feature tables, trains a stacking ensemble classifier (XGBoost + LightGBM + CatBoost), performs feature importance analysis, and generates reproducible model artifacts.

The codebase is written in clean, maintainable Python following PEP 8 standards, with comprehensive docstrings, type hints, and robust error handling.

**Version**: 1.0.0  
**Last Updated**: November 12, 2025  
**License**: MIT

This README provides a comprehensive project overview, installation and usage instructions, configuration notes, and a quick reference for all included scripts and outputs.

## Project Summary

- **Inputs**: Labeled folders of TIFF image crops (per-image) or precomputed feature CSVs
- **Pipeline**: 7-step automated workflow (extraction → cleaning → CSV loading → training → artifacts → feature importance → violin plots)
- **Features**: 133+ morphological features including shape, intensity stats, texture (LBP, GLCM, Gabor), nuclear features, and spatial measures
- **Processing**: Aggregates per-region measurements into per-image summaries (count, mean, median, std, min, max)
- **Modeling**: Cleans data (inf→NaN), drops columns with excessive missingness, imputes (median), scales, trains a stacking ensemble (XGBoost + LightGBM + CatBoost with LogisticRegression meta-learner), extracts feature importances, and generates advanced feature insights
- **Outputs**: Cleaned/raw features CSV, confusion matrix, JSON classification report, feature importance CSV and plot, feature insights plots, and saved model bundle for reproducible prediction

## Quick Start (Windows PowerShell)

1. From the project folder, create and activate a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Development Setup

For development, install pre-commit hooks to ensure code quality:

```powershell
pip install pre-commit
pre-commit install
```

This will run Black code formatting on commits.

## Usage
```

2. Edit `config.json` to set `paths.image_directory`, `paths.output_directory`, and any feature or analysis parameters.
3. Run the full pipeline (verbose):

```powershell
py -3 -u .\main.py -c .\config.json -v
```

This will run the complete 7-step pipeline: extract features, clean and preprocess them, load cleaned data from CSV, train the model, extract feature importances, create violin plots, and save all outputs to the configured `output_directory` (organized into `reports/`, `models/`, and `plots/` subfolders).

## Pipeline Overview

The automated pipeline consists of 7 steps:

1. **Feature Extraction**: Extracts morphological, texture, and intensity features from TIFF images using configurable parameters.
2. **Clean & Prepare**: Cleans data, handles missing values, imputes, scales, and saves cleaned CSV.
3. **Load Cleaned Data**: Loads the cleaned CSV to ensure consistency for all modeling steps.
4. **Model Training**: Trains stacking ensemble classifier with cross-validation.
5. **Save Artifacts**: Saves model bundle, preprocessing objects, reports, and plots.
6. **Feature Importance**: Extracts and visualizes top feature importances from the trained ensemble model.
7. **Feature Insights**: Creates violin plots with trendlines for top N features across age groups.

## Configuration

All runtime parameters are in `config.json` next to `main.py`. Key configuration sections:

- **`paths`**:

  - `image_directory` — Folder of labeled subfolders containing TIFF crops.
  - `output_directory` — Where CSVs, plots, and models are written.
  - `output_features_csv` — Optional path for the extracted features CSV.
- **`analysis_params`**:

  - `cv_folds` — Number of cross-validation folds (default: 5).
  - `random_state` — Random seed for reproducibility (default: 42).
  - `max_nan_fraction` — Maximum allowed NaN fraction for columns before dropping (default: 0.5).
  - `test_size` — Train/test split ratio (default: 0.2).
  - `impute_strategy` — Imputation method (default: "median").
  - `rf_params` — RandomForest parameters (n_estimators, max_depth).
  - `target_column` — Name of target column (default: "label").
- **`feature_params`**:

  - `lbp_points`, `lbp_radius` — Local Binary Pattern parameters.
  - `glcm_levels` — GLCM quantization levels.
  - `gabor_frequencies`, `gabor_angles` — Gabor filter parameters.
  - `use_otsu` — Whether to use Otsu thresholding (default: true).
  - Channel thresholds for segmentation.
- **`feature_insights_params`**:

  - `top_features_count` — Number of top features to analyze (default: 10).

Modify these values to tune extraction and modeling without editing code.

## Project Structure

```
Morphological-Feature-Analysis-of-Retinal-Pigment-Epithelial-Cells-from-C57BL-6J-Mice-During-Aging/
├── .git/                          # Git repository
├── .gitignore                     # Git ignore patterns
├── analysis_results/              # Generated outputs (reports/, models/, plots/)
├── config.json                    # Runtime configuration file
├── main.py                        # Main entrypoint script
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
└── scripts/                       # Modular analysis scripts
    ├── directories.py             # Directory management utilities
    ├── clean_and_prepare.py       # Data cleaning and preprocessing
    ├── feature_extraction.py      # Feature extraction from images
    ├── feature_insights.py        # Feature importance and visualization
    ├── load_model_bundle_and_predict.py  # Prediction on new data
    ├── timer.py                   # Performance timing utilities
    ├── train_and_save_pipeline.py # Model training and artifact saving
    └── visualize_channels.py      # Image channel visualization
```

## Files and Scripts

- `main.py` — **Main entrypoint**. Orchestrates the complete 7-step pipeline.
- `config.json` — Runtime configuration (paths, feature params, analysis params).
- `requirements.txt` — Python dependencies.
- `scripts/` — Modular scripts for specific functionality:
  - `directories.py` — Directory management utilities.
  - `clean_and_prepare.py` — Feature cleaning and preprocessing pipeline.
  - `feature_extraction.py` — Feature extraction from TIFF images.
  - `feature_insights.py` — Feature importance extraction and violin plot generation with trendlines.
  - `train_and_save_pipeline.py` — Model training, artifact saving, and classification reports.
  - `load_model_bundle_and_predict.py` — Prediction on new data using saved models.
  - `timer.py` — Timing utilities for performance measurement.
  - `visualize_channels.py` — Channel visualization utilities.

All scripts follow PEP 8 standards with type hints, comprehensive docstrings, and robust error handling.

## Outputs (Default Location: `analysis_results/`, Organized into `reports/`, `models/`, and `plots/`)

- **`reports/rpe_extracted_features.csv`** — Raw extracted features (one row per image).
- **`reports/rpe_extracted_features_cleaned.csv`** — Cleaned + scaled features.
- **`plots/confusion_matrix.png`** — Confusion matrix for classifier predictions.
- **`reports/classification_report.json`** — Classification metrics from cross-validated predictions.
- **`reports/feature_importances.csv`** — Feature importances from the ensemble model.
- **`plots/feature_importances.png`** — Bar plot of top N feature importances.
- **`plots/top_features_violin_plots.png`** — Violin plots with trendlines comparing distributions of top features across age groups.
- **`models/model.joblib`**, **`models/preprocessor.joblib`** — Saved model and preprocessing artifacts.

## Predicting on New Data

To predict on new data using the saved model:

```powershell
py -3 scripts\load_model_bundle_and_predict.py --features "analysis_results\reports\rpe_extracted_features_cleaned.csv" --model "analysis_results\models\model.joblib" --output "analysis_results\reports\predictions.csv"
```

This will load the saved model, preprocess the features using the saved preprocessing pipeline, make predictions, and save them to a CSV file.

## Troubleshooting & Tips

- **Dependencies**: The project requires XGBoost, LightGBM, and CatBoost. On some systems, you may need to install these separately:
  ```powershell
  pip install xgboost lightgbm catboost
  ```
  
- **Error Handling**: The code includes robust error handling for file I/O, image processing, and data operations.

- **Configuration**: Ensure `paths.image_directory` points to the correct folder with readable TIFF files.

- **Data Quality**: If many columns are dropped, lower `analysis_params.max_nan_fraction` or inspect the features CSV for missing values.

- **Memory Usage**: For large image datasets, consider reducing `feature_params.glcm_levels` or `gabor_frequencies` arrays.

- **Model Training**: The stacking ensemble uses 5-fold cross-validation by default. Adjust `cv_folds` for faster training or more robust validation.

- **Code Quality**: All scripts follow PEP 8 with type hints and docstrings for maintainability.

## License

MIT
