# Morphological Feature Analysis of Retinal Pigment Epithelial Cell from C57BL/6J Mice during Aging

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

1. From the `Final Code` folder, create and activate a venv and install deps:

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

This will extract features, clean and preprocess them automatically, perform PCA, train the model, and save all outputs to the configured `output_directory` (organized into `reports/`, `models/`, and `plots/` subfolders).

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

- `main.py` — single entrypoint. Orchestrates extraction → cleaning → PCA →
  training → export. Implements a Singleton pipeline manager and contains
  modular functions for feature extraction and plotting. Cleaning is now
  integrated and runs automatically, saving cleaned features and preprocessing
  bundle.
- `config.json` — runtime configuration (paths, feature params, analysis params).
- `requirements.txt` — Python dependencies used by the project.
  Note: To run predictions you can load the saved model bundle and preprocessing objects directly from the `models/` folder. Example PowerShell snippet:

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

- **Code Quality Improvements**: The entire codebase has been refactored for PEP 8 compliance, including snake_case variable naming, comprehensive docstrings, type hints, and modern Python idioms (f-strings, pathlib). Error handling has been enhanced with try-except blocks for file I/O and other risky operations.
- **Integrated Cleaning**: The cleaning functionality from `clean_features.py` has been merged into the main pipeline in `analysis.py`. Cleaning now runs automatically after feature extraction, saving the cleaned CSV and preprocessing bundle without needing a separate step.
- Gabor feature stability: Gabor responses now use numerically stable
  hypot-based magnitude and nan-safe aggregations to avoid inf/overflow values.
- Robust preprocessing: pipeline replaces inf with NaN, drops columns with
  > `max_nan_fraction` NaNs, imputes remaining NaNs with median, and scales
  > features before PCA/training.
- Model bundle saved with preprocessing objects so downstream predictions are
  reproducible without re-training.

## Predicting on an existing features CSV

See the PowerShell/joblib snippet earlier under "Files and scripts" for a ready-to-run example that loads the model bundle from `models/` and writes predictions to `reports/`.

In short: load `models/stacking_model_bundle.joblib` with `joblib.load`, read `reports/rpe_extracted_features.csv`, apply the saved imputer and scaler (if present), run `model.predict`, and save the resulting CSV into `reports/`.

## Troubleshooting & tips

- The code now includes enhanced error handling for file I/O, image processing, and data operations, providing clear error messages for debugging.
- If the pipeline errors opening TIFFs, confirm `paths.image_directory` points
  to the correct folder and that files are readable.
- If many columns are dropped, lower `analysis_params.max_nan_fraction` or
  inspect `rpe_extracted_features.csv` to see which features have missing
  values.
- If XGBoost is not available, the pipeline automatically falls back to
  `HistGradientBoostingClassifier`.

## Next steps / suggested improvements

- Add unit tests for feature extraction (including a small synthetic image to
  assert no infs are produced), and CI to run linting and tests on push.
- Add a small CLI with subcommands (`extract`, `clean`, `train`, `predict`) to
  separate pipeline phases.
- Add a short Jupyter notebook that visualizes top features and their importances
  for easier interpretation.

## License

MIT
