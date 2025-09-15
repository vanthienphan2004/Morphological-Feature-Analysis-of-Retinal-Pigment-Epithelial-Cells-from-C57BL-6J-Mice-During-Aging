r"""
Consolidated predictor script.

Detects and supports three saved model forms:
 - model bundle (dict) with keys: 'model', 'imputer', 'scaler', 'dropped_columns', 'feature_columns'
 - sklearn Pipeline-like object (has named_steps)
 - legacy estimator (plain estimator)

Usage:
    py -3 scripts\predict.py --model <path_or_dir> --features <path_to_csv> [--out <out_dir>] [--analyze]

Flags:
  --force-legacy  : allow using a legacy estimator (best-effort, unsafe if columns/order mismatch)
  --analyze       : also generate classification report and confusion matrix (requires true labels in CSV)
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from analysis import resolve_output_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help='Path to model bundle file or folder containing models (optional)')
    parser.add_argument('--features', '-f', help='Path to features CSV to predict', required=True)
    parser.add_argument('--out', '-o', help='Path to output predictions CSV', default=None)
    parser.add_argument('--force-legacy', action='store_true', help='Allow using a legacy estimator (columns/order must match training)')
    parser.add_argument('--analyze', action='store_true', help='Also write classification report and confusion matrix (requires true labels in CSV)')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    out_root, reports_dir, models_dir, plots_dir = resolve_output_dirs(base)

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        # prefer a bundle in models_dir
        bundle_path = models_dir / 'stacking_model_bundle.joblib'
        pipeline_path = models_dir / 'pipeline.joblib'
        if bundle_path.exists():
            model_path = bundle_path
        elif pipeline_path.exists():
            model_path = pipeline_path
        else:
            raise SystemExit('No model specified and no pipeline/bundle found in models/')

    # if a directory is passed, look for common names
    if model_path.is_dir():
        b = model_path / 'stacking_model_bundle.joblib'
        p = model_path / 'pipeline.joblib'
        legacy = model_path / 'stacking_model.joblib'
        if b.exists():
            model_path = b
        elif p.exists():
            model_path = p
        elif legacy.exists():
            model_path = legacy
        else:
            raise SystemExit(f'No recognized model artifact found in {model_path}')

    if not model_path.exists():
        raise SystemExit(f'Model path not found: {model_path}')

    features_csv = Path(args.features)
    if not features_csv.exists():
        raise SystemExit(f'Features CSV not found: {features_csv}')

    out_csv = Path(args.out) if args.out else (reports_dir / (features_csv.stem + '_predictions.csv'))

    print(f'Loading features from {features_csv}')
    df = pd.read_csv(features_csv)

    print(f'Loading model from {model_path}')
    loaded = joblib.load(model_path)

    # default processing variables
    model = None
    imputer = None
    scaler = None
    dropped = []
    feature_cols = None

    # Case 1: bundle dict
    if isinstance(loaded, dict) and 'model' in loaded:
        print('Detected model bundle with preprocessing')
        model = loaded['model']
        imputer = loaded.get('imputer')
        scaler = loaded.get('scaler')
        dropped = loaded.get('dropped_columns', []) or []
        feature_cols = loaded.get('feature_columns')

        df_proc = df.copy()
        for c in dropped:
            if c in df_proc.columns:
                df_proc = df_proc.drop(columns=[c])

        if feature_cols:
            missing = [c for c in feature_cols if c not in df_proc.columns]
            if missing:
                print(f'Warning: missing expected feature columns: {missing}')
            df_proc = df_proc[[c for c in feature_cols if c in df_proc.columns]]

        df_proc = df_proc.replace([np.inf, -np.inf], np.nan)
        if imputer is None:
            imputer = SimpleImputer(strategy='median')
        X = imputer.transform(df_proc)
        if scaler is None:
            scaler = StandardScaler()
        Xs = scaler.transform(X)
        preds = model.predict(Xs)

    # Case 2: Pipeline-like or estimator supporting raw input
    else:
        model = loaded
        # If it looks like a Pipeline, use directly
        if hasattr(model, 'named_steps'):
            print('Detected sklearn Pipeline — using it directly')
            X = df.drop(columns=['label']) if 'label' in df.columns else df.copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = model.predict(X)
        else:
            # plain estimator — require force flag
            if not args.force_legacy:
                raise SystemExit('Loaded a legacy estimator; re-run with --force-legacy to permit best-effort prediction')
            print('Legacy estimator: applying best-effort predict (fill NaN with 0)')
            X = df.drop(columns=['label']) if 'label' in df.columns else df.copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = model.predict(X)

    # Save predictions
    out = df.copy()
    out['prediction'] = preds
    out.to_csv(out_csv, index=False)
    print(f'Wrote predictions to {out_csv}')

    # Optional analysis
    if args.analyze and 'label' in df.columns:
        y_true = df['label']
        # ensure types are compatible (best-effort)
        y_true = y_true.astype(str)
        preds = preds.astype(str)
        try:
            report = classification_report(y_true, preds, output_dict=True)
        except Exception:
            report = None
        # write JSON report if possible
        if report is not None:
            with open(reports_dir / (features_csv.stem + '_prediction_report.json'), 'w') as fh:
                json.dump(report, fh, indent=2)
            print(f'Wrote classification report to {reports_dir / (features_csv.stem + "_prediction_report.json")}')

        # confusion matrix
        try:
            cm = confusion_matrix(y_true, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            plt.figure(figsize=(8, 6))
            disp.plot(values_format='d', cmap='Blues')
            plt.title('Confusion Matrix (provided CSV)')
            plt.savefig(plots_dir / (features_csv.stem + '_confusion_matrix.png'))
            plt.close()
            print(f'Wrote confusion matrix to {plots_dir / (features_csv.stem + "_confusion_matrix.png")}')
        except Exception as exc:
            print('Could not create confusion matrix:', exc)


if __name__ == '__main__':
    main()
