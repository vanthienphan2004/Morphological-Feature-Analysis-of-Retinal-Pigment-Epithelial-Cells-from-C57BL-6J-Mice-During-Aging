"""
Train a supervised stacking pipeline from extracted features and save a Pipeline including the scaler.
Outputs:
 - pipeline.joblib (Pipeline with scaler and model)
 - classification_report.json
 - confusion_matrix.png
 - feature_importances.csv and feature_importances.png (from RandomForest base learner if present)
 - feature_names.json (to record the expected column order)

Usage:
    py scripts/train_and_save_pipeline.py --features <path_to_csv> --outdir <out_dir>

"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def build_estimators(random_state: int = 42, rf_params: Optional[Dict[str, Any]] = None) -> Tuple[RandomForestClassifier, Tuple[str, Any]]:
    rf_params = rf_params or {}
    rf = RandomForestClassifier(random_state=random_state, **rf_params)
    try:
        from xgboost import XGBClassifier  # type: ignore
        gb = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=random_state)
        gb_name = 'xgboost'
    except Exception:
        gb = HistGradientBoostingClassifier(random_state=random_state)
        gb_name = 'hist_gb'
    return rf, (gb_name, gb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="Path to extracted features CSV")
    parser.add_argument("--outdir", help="Output directory to write pipeline and reports")
    parser.add_argument("--cv", type=int, default=5, help="CV folds")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]

    # Resolve output directories using centralized helper
    from scripts.analysis import resolve_output_dirs
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)

    features_path = Path(args.features) if args.features else reports_dir / 'rpe_extracted_features.csv'
    outdir = Path(args.outdir) if args.outdir else out_root

    if not features_path.exists():
        raise SystemExit(f"Features CSV not found: {features_path}")

    try:
        features_df = pd.read_csv(features_path)
    except Exception as e:
        raise SystemExit(f'Error loading features CSV: {e}')
        
    if 'label' not in features_df.columns:
        raise SystemExit("Features CSV must contain a 'label' column")

    features_matrix = features_df.drop(columns=['label']).copy()
    labels = features_df['label'].copy()

    # Save feature names/order for future checks (reports)
    feature_names = list(features_matrix.columns)
    try:
        with open(reports_dir / 'feature_names.json', 'w') as fh:
            json.dump(feature_names, fh, indent=2)
    except Exception as e:
        print(f'Warning: Could not save feature names: {e}')

    rf, gb_pair = build_estimators(random_state=args.random_state, rf_params=None)
    estimators = [('rf', rf), gb_pair]
    final_estimator = LogisticRegression(max_iter=2000)
    stacking = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=args.cv, passthrough=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', stacking),
    ])

    print("Starting cross-validated predictions (this may take a bit)...")
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    try:
        predictions = cross_val_predict(pipeline, features_matrix, labels, cv=skf, n_jobs=-1)
    except Exception as exc:
        print("cross_val_predict failed, falling back to single-fit predict. Error:", exc)
        pipeline.fit(features_matrix, labels)
        predictions = pipeline.predict(features_matrix)

    report = classification_report(labels, predictions, output_dict=True)
    print("Classification report (CV):")
    print(classification_report(labels, predictions))

    # confusion matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 6))
    disp.plot(values_format='d', cmap='Blues')
    plt.title('Confusion Matrix (CV)')
    plt.savefig(plots_dir / 'confusion_matrix_cv.png')
    plt.close()

    # Fit pipeline on full data and save
    print("Fitting final pipeline on full data...")
    pipeline.fit(features_matrix, labels)
    try:
        joblib.dump(pipeline, models_dir / 'pipeline.joblib')
        print(f"Saved pipeline to {models_dir / 'pipeline.joblib'}")
    except Exception as e:
        raise SystemExit(f'Error saving pipeline: {e}')

    # Save classification report
    try:
        with open(reports_dir / 'classification_report_cv.json', 'w') as fh:
            json.dump(report, fh, indent=2)
    except Exception as e:
        print(f'Warning: Could not save classification report: {e}')

    # Feature importances from RF base learner in stacking (if present)
    try:
        model = pipeline.named_steps.get('model')
        rf_bt = None
        # Try to find a RandomForestClassifier among named_estimators_
        try:
            from sklearn.ensemble import RandomForestClassifier as _RFC
        except Exception:
            _RFC = None

        if model is not None and hasattr(model, 'named_estimators_') and _RFC is not None:
            named = model.named_estimators_
            # Preferred key 'rf'
            if 'rf' in named:
                rf_bt = named.get('rf')
            else:
                # Fallback: search for an estimator instance of RandomForestClassifier
                for nm, est in named.items():
                    try:
                        if isinstance(est, _RFC):
                            rf_bt = est
                            break
                    except Exception:
                        continue

        # As a last resort, check attribute estimators_ (fitted estimators list)
        if rf_bt is None and model is not None and hasattr(model, 'estimators_') and _RFC is not None:
            for est in getattr(model, 'estimators_', []):
                try:
                    if isinstance(est, _RFC):
                        rf_bt = est
                        break
                except Exception:
                    continue

        importances = None
        if rf_bt is not None:
            importances = getattr(rf_bt, 'feature_importances_', None)

        if importances is None:
            print("No RandomForest feature importances found in stacking model; skipping feature_importances.csv")
        else:
            fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
            fi = fi.sort_values('importance', ascending=False)
            fi.to_csv(reports_dir / 'feature_importances.csv', index=False)

            plt.figure(figsize=(10, 6))
            plt.bar(fi['feature'].head(30), fi['importance'].head(30))
            plt.xticks(rotation=90)
            plt.title('Top 30 Feature Importances (RF)')
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importances.png')
            plt.close()
    except Exception as exc:
        print("Could not extract feature importances:", exc)

    print("Training and saving complete. Outputs written to:")
    print(" -", models_dir / 'pipeline.joblib')
    print(" -", reports_dir / 'classification_report_cv.json')
    print(" -", plots_dir / 'confusion_matrix_cv.png')
    print(" -", reports_dir / 'feature_importances.csv' if (reports_dir / 'feature_importances.csv').exists() else "(no importances)")


if __name__ == '__main__':
    main()
