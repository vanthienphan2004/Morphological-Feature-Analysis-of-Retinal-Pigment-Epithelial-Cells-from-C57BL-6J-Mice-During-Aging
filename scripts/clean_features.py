"""
Clean and impute the existing extracted features CSV.
- Replaces inf/-inf with NaN
- Reports per-column NaN/inf counts
- Drops columns with NaN fraction > config.analysis_params.max_nan_fraction (defaults 0.5)
- Imputes remaining NaNs with median and scales with StandardScaler
- Saves cleaned CSV to analysis_results/rpe_extracted_features_cleaned.csv
- Saves imputer+scaler bundle to analysis_results/feature_preprocessing_bundle.joblib

Run from project root (Final Code):
    py -3 scripts\clean_features.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
from analysis import resolve_output_dirs
out_root, reports_dir, models_dir, _ = resolve_output_dirs(ROOT)
cfg_path = ROOT / 'config.json'
if cfg_path.exists():
    try:
        with open(cfg_path, 'r') as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
else:
    cfg = {}

csv_path = Path(cfg.get("paths", {}).get("output_features_csv", reports_dir / "rpe_extracted_features.csv"))
cleaned_csv = reports_dir / "rpe_extracted_features_cleaned.csv"
prep_joblib = models_dir / "feature_preprocessing_bundle.joblib"

print(f"Loading features from {csv_path}")
df = pd.read_csv(csv_path)

# Replace inf with NaN
n_inf_before = df.isin([np.inf, -np.inf]).sum().sum()
print(f"Total inf values before cleanup: {int(n_inf_before)}")
df = df.replace([np.inf, -np.inf], np.nan)

# Report columns with NaNs
nan_counts = df.isna().sum()
nan_frac = nan_counts / len(df)
print("Top columns by NaN fraction (showing >0):")
print(nan_frac[nan_frac > 0].sort_values(ascending=False).head(30).to_string())

# Determine drop threshold
max_nan_frac = float(cfg.get("analysis_params", {}).get("max_nan_fraction", 0.5))
cols_to_drop = nan_frac[nan_frac > max_nan_frac].index.tolist()
if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} columns with NaN fraction > {max_nan_frac}: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
else:
    print("No columns to drop based on NaN fraction threshold.")

# Separate labels (if present)
label_col = "label" if "label" in df.columns else None
labels = None
if label_col:
    labels = df[label_col].copy()
    features_df = df.drop(columns=[label_col])
else:
    features_df = df.copy()

# Impute
imputer = SimpleImputer(strategy="median")
imputed = imputer.fit_transform(features_df)

# Scale
scaler = StandardScaler()
scaled = scaler.fit_transform(imputed)

# Convert back to dataframe (keeps feature names)
features_clean = pd.DataFrame(scaled, columns=features_df.columns, index=features_df.index)
if labels is not None:
    features_clean["label"] = labels

# Save cleaned CSV
features_clean.to_csv(cleaned_csv, index=False)
print(f"Saved cleaned features to {cleaned_csv}")

# Save preprocessing bundle
bundle = {
    "imputer": imputer,
    "scaler": scaler,
    "dropped_columns": cols_to_drop,
    "feature_columns": features_df.columns.tolist(),
}
joblib.dump(bundle, prep_joblib)
print(f"Saved preprocessing bundle to {prep_joblib}")

# Summary
n_inf_after = features_clean.isin([np.inf, -np.inf]).sum().sum()
n_nan_after = features_clean.isna().sum().sum()
print(f"Total inf after cleanup: {int(n_inf_after)}; total NaN after cleanup (should be 0): {int(n_nan_after)}")
print("Done.")
