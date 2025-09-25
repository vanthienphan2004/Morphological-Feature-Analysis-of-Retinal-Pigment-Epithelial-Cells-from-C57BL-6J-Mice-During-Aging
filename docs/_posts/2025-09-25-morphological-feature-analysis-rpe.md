---
title: "Morphological Feature Analysis of RPE Cells — a reproducible Python pipeline"
description: "A configuration-driven pipeline to extract, analyze, and model morphological features from retinal pigment epithelial (RPE) image crops using XGBoost, LightGBM, and CatBoost."
author: "vanthienphan2004"
date: 2025-09-25
tags: [Python, Image Processing, Machine Learning, XGBoost, LightGBM, CatBoost, Reproducible Research]
image: /Pipeline_Diagram.png
---

# Morphological Feature Analysis of RPE Cells — a reproducible Python pipeline

Understanding how cell morphology changes with age can reveal important biological signatures. To make this analysis reproducible and extensible, I built a configuration-driven Python pipeline that extracts morphological and texture features from retinal pigment epithelium (RPE) image crops, cleans and aggregates per-region measurements into per-image summaries, and trains a robust stacking ensemble for classification and feature discovery.

## Why this project

- **Reproducibility**: Everything is driven by `config.json` so you can re-run experiments by changing parameters, not code.
- **Interpretability**: The pipeline saves feature importances and visualizations (bar plots, violin plots with trendlines).
- **Practical ML**: Uses a stacking ensemble (XGBoost + LightGBM + CatBoost with a logistic meta-learner) for robust performance across heterogeneous cell features.

## What it does (high level)

- **Input**: labeled folders of TIFF image crops, or a precomputed feature CSV.
- **Feature extraction**: shape, intensity statistics, texture (LBP, GLCM, Gabor), nuclear and spatial metrics (133+ features).
- **Cleaning & aggregation**: removes columns with excessive missingness, imputes, scales, and saves a cleaned CSV.
- **Modeling**: trains a stacking ensemble with cross-validation, saves a model bundle (model + preprocessing).
- **Insights**: exports feature importance CSV and a top-features violin plot with trendlines across age groups.
- **Prediction**: reusable `scripts/load_model_bundle_and_predict.py` for batch predictions.

## Key files

- `main.py` — orchestrates the 7-step pipeline.
- `config.json` — runtime configuration (paths, feature & analysis params).
- `scripts/feature_extraction.py` — extraction logic from TIFF images.
- `scripts/clean_and_prepare.py` — cleaning, imputation, scaling.
- `scripts/train_and_save_pipeline.py` — model training and artifact saving.
- `scripts/feature_insights.py` — feature importances & violin plots.
- `scripts/load_model_bundle_and_predict.py` — model loading & prediction utilities.

## Quick start (Windows PowerShell)

1. Create a virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Edit `config.json` (set `paths.image_directory`, `paths.output_directory`).

3. Run the full pipeline:

```powershell
py -3 -u .\main.py -c .\config.json -v
```

## Outputs

- `analysis_results/reports/rpe_extracted_features.csv` — raw features
- `analysis_results/reports/rpe_extracted_features_cleaned.csv` — cleaned features
- `analysis_results/reports/feature_importances.csv` and `analysis_results/plots/feature_importances.png`
- `analysis_results/plots/top_features_violin_plots.png`
- `analysis_results/models/` — saved model artifacts

## Architecture notes & reproducibility tips

- The pipeline is modular and config-driven. Try varying:
  - `feature_params.glcm_levels` (memory/time tradeoff)
  - `analysis_params.max_nan_fraction` (controls column dropping)
  - `analysis_params.cv_folds` (validation robustness)

- If you use a different dataset, ensure labels are present and consistent with `analysis_params.target_column`.

## Where to go from here

- Add more feature extractors (e.g., deep-learning embeddings).
- Add permutation importance for more robust interpretability.
- Add a small CI job or GitHub Action that runs a smoke-test on a tiny sample.

## Acknowledgements & licensing

- MIT license in the repository.
- If you use the code in your research, please cite or link back to the repo and share results.


---

*If you'd like, I can also create a short LinkedIn-friendly summary file and/or a Git tag / release entry for this commit.*