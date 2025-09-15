# Morphologica## Project summary (elevator pitch)

- **Inputs**: labeled folders of TIFF crops (per-image) or precomputed feature CSVs.
- **Pipeline**: 6-step automated workflow (extraction → cleaning → CSV loading → PCA → training → artifacts)
- **Features**: morphology, intensity stats, texture (LBP, GLCM-like), Gabor responses, and spatial measures (centroids, nearest neighbor, density)
- **Processing**: Aggregates per-region measurements into per-image summaries (count, mean, median, std, min, max)
- **Modeling**: Cleans data (inf→NaN), drops columns with excessive missingness, imputes (median), scales, runs PCA, and trains a stacking classifier (RandomForest + XGBoost/HistGradientBoosting, logistic meta-learner)
- **Outputs**: cleaned/raw features CSV, PCA plots, confusion matrix, JSON report, and a saved model bundle containing the model + imputer + scaler for reproducible predictionysis of Retinal Pigment Epithelial Cell from C57BL/6J Mice during Aging

A configuration-driven Python application for extracting, aggregating, and
analyzing features from Retinal Pigment Epithelium (RPE) image crops. The
pipeline produces cleaned feature tables, PCA visualizations, supervised
classification (stacking ensemble), and reproducible model artifacts.

The codebase is written in clean, maintainable Python following PEP 8 standards,
with comprehensive docstrings, type hints, and robust error handling.

This README gives a concise project overview, installation and usage
instructions, configuration notes, and a quick reference for the included
scripts and outputs.

## Project summary (elevator pitch)

- Inputs: labeled folders of TIFF crops (per-image) or precomputed feature CSVs.
- Extracts features: morphology, intensity stats, texture (LBP,
  GLCM-like), Gabor responses, and spatial measures (centroids, nearest
  neighbor, density).
- Aggregates per-region measurements into per-image summaries (count, mean,
  median, std, min, max) so each image is one feature vector.
- Cleans data (inf→NaN), drops columns with excessive missingness (configurable),
  imputes (median), scales, runs PCA, and trains a stacking classifier
  (RandomForest + XGBoost/HistGradientBoosting, logistic meta-learner).
- Outputs: cleaned/raw features CSV, PCA plots, confusion matrix, JSON report,
  and a saved model bundle containing the model + imputer + scaler for
  reproducible prediction.

## Quick start (Windows PowerShell)

1. From the project folder, create and activate a venv and install deps:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Edit `config.json` to set `paths.image_directory`, `paths.output_directory` and
   any feature or analysis parameters.
3. Run the full pipeline (verbose):

```powershell
py -3 -u .\main.py -c .\config.json -v
```

This will run the complete 6-step pipeline: extract features, clean and preprocess them, load cleaned data from CSV, perform PCA, train the model, and save all outputs to the configured `output_directory` (organized into `reports/`, `models/`, and `plots/` subfolders).

## Pipeline Overview

The automated pipeline consists of 6 steps:

1. **Feature Extraction**: Extracts morphological, texture, and intensity features from TIFF images
2. **Clean & Prepare**: Cleans data, handles missing values, imputes, scales, and saves cleaned CSV
3. **Load Cleaned Data**: Loads the cleaned CSV to ensure consistency for all modeling steps
4. **PCA Analysis**: Performs PCA and generates visualization plots
5. **Model Training**: Trains stacking classifier (RandomForest + XGBoost/HistGradientBoosting)
6. **Save Artifacts**: Saves model bundle, preprocessing objects, reports, and plots

## Configuration

All runtime parameters live in `config.json` next to `main.py`. Important keys:

- `paths.image_directory` — folder of labeled subfolders containing TIFF crops.
- `paths.output_directory` — where CSVs, plots, and models are written.
- `paths.output_features_csv` — path for the extracted features CSV.
- `feature_params` — controls feature extraction (LBP points/radius, GLCM
  levels, `gabor_frequencies`, `gabor_angles`, `use_otsu`, channel thresholds).
- `analysis_params` — model & CV settings (cv_folds, random_state, rf_params),
  and `max_nan_fraction` (fraction above which a column is dropped before
  imputation; default 0.5).

Modify these values to tune extraction and modeling without editing code.

## Files and scripts

- `main.py` — **Main entrypoint**. Orchestrates the complete 6-step pipeline: feature extraction → cleaning → CSV loading → PCA → training → export. Uses cleaned CSV data for modeling steps to ensure consistency.
- `config.json` — runtime configuration (paths, feature params, analysis params)
- `requirements.txt` — Python dependencies
- `scripts/` — modular scripts for specific functionality:
  - `feature_extraction.py` — morphological, texture, and intensity feature extraction
  - `clean_and_prepare.py` — data cleaning, imputation, and preprocessing
  - `analysis.py` — PCA, model training, and artifact saving
  - `load_model_bundle_and_predict.py` — prediction on new data using saved models
  - `timer.py` — execution timing utilities
  - `train_and_save_pipeline.py` — alternative training script
  - `visualize_channels.py` — channel visualization for debugging

All scripts have been comprehensively refactored for PEP 8 compliance, type hints, error handling, and maintainability.

```powershell
py - <<'PY'
import joblib, pandas as pd
from pathlib import Path
models = Path(r"<output_directory>\models")
bundle = joblib.load(models / 'stacking_model_bundle.joblib')
df = pd.read_csv(r"<output_directory>\reports\rpe_extracted_features.csv")
imp = bundle.get('imputer')
sc = bundle.get('scaler')
model = bundle['model']
X = df[bundle.get('feature_columns', df.columns)]
X = X.replace([float('inf'), float('-inf')], pd.NA)
if imp is not None:
  X = imp.transform(X)
if sc is not None:
  X = sc.transform(X)
preds = model.predict(X)
out = df.copy()
out['prediction'] = preds
out.to_csv(r"<output_directory>\reports\rpe_extracted_features_predictions.csv", index=False)
print('Wrote predictions')
PY
```

Alternatively, a consolidated helper is available:

```powershell
py -3 scripts\load_model_bundle_and_predict.py --features "analysis_results\reports\rpe_extracted_features_cleaned.csv" --model "analysis_results\models\model.joblib" --force-legacy --analyze
```

If you need the original helper scripts (for compatibility), they are archived in `scripts/archived_predictors/`.

## Outputs (default location: `analysis_results/`, organized into `reports/`, `models/`, and `plots/`)

- `rpe_extracted_features.csv` — raw extracted features (one row per image).
- `rpe_extracted_features_cleaned.csv` — cleaned + scaled features (generated
  automatically in the pipeline).
- `pca_scree_plot.png`, `pca_cumulative_plot.png`, `pca_scatter_plot.png` — PCA
  visualizations.
- `confusion_matrix.png` — confusion matrix for the classifier predictions.
- `classification_report.json` — classification metrics from cross-validated
  predictions.
- `stacking_model_bundle.joblib` — recommended artifact: a dict containing the
  trained model under `'model'`, plus `'imputer'`, `'scaler'`, and
  `'dropped_columns'` for reproducible predictions.

## Recent important fixes (what changed)

- **Complete Code Refactoring**: The entire codebase has been comprehensively refactored for PEP 8 compliance, including:
  - Snake_case variable naming throughout
  - Comprehensive type hints on all functions
  - Modern Python idioms (f-strings, pathlib)
  - Enhanced error handling with try-except blocks
  - Detailed docstrings for all functions and classes
  - Consistent code formatting and structure

- **Pipeline Architecture**: Updated to 6-step workflow that uses cleaned CSV for modeling steps:
  1. Feature extraction from images
  2. Clean and prepare data (saves cleaned CSV)
  3. Load cleaned data from CSV (ensures consistency)
  4. PCA for visualization
  5. Training stacking classifier
  6. Save artifacts and reports

- **Modular Script Organization**: All functionality moved to dedicated scripts in `scripts/` folder for better maintainability

- **Enhanced Error Handling**: Robust error handling for file I/O, image processing, and data operations with clear error messages

- **Gabor Feature Stability**: Gabor responses now use numerically stable hypot-based magnitude and nan-safe aggregations

- **Preprocessing Pipeline**: Automatically handles inf→NaN conversion, column dropping based on missingness, median imputation, and feature scaling

- **Model Bundle**: Saved with preprocessing objects for fully reproducible predictions without re-training

## Predicting on an existing features CSV

See the PowerShell/joblib snippet earlier under "Files and scripts" for a ready-to-run example that loads the model bundle from `models/` and writes predictions to `reports/`.

In short: load `models/stacking_model_bundle.joblib` with `joblib.load`, read `reports/rpe_extracted_features.csv`, apply the saved imputer and scaler (if present), run `model.predict`, and save the resulting CSV into `reports/`.

## Troubleshooting & tips

- **Enhanced Error Handling**: The refactored code includes comprehensive error handling for file I/O, image processing, and data operations, providing clear error messages for debugging.
- **Pipeline Steps**: The 6-step pipeline ensures data consistency by using the cleaned CSV for all modeling steps (PCA, training, predictions).
- **Configuration**: If the pipeline errors opening TIFFs, confirm `paths.image_directory` points to the correct folder and that files are readable.
- **Data Quality**: If many columns are dropped, lower `analysis_params.max_nan_fraction` or inspect `rpe_extracted_features.csv` to see which features have missing values.
- **Dependencies**: If XGBoost is not available, the pipeline automatically falls back to `HistGradientBoostingClassifier`.
- **Code Quality**: All scripts follow PEP 8 standards with type hints, comprehensive docstrings, and modern Python practices for maintainability.

## Next steps / suggested improvements

- **✅ Code Quality**: Complete PEP 8 refactoring with type hints, docstrings, and error handling (COMPLETED)
- **✅ Pipeline Architecture**: 6-step workflow with CSV-based data consistency (COMPLETED)
- **Testing**: Add unit tests for feature extraction and CI for linting/tests on push
- **CLI Enhancement**: Add subcommands (`extract`, `clean`, `train`, `predict`) for separate pipeline phases
- **Visualization**: Create Jupyter notebook for feature importance analysis and model interpretation
- **Documentation**: Add API documentation and usage examples for individual scripts

## License

MIT
