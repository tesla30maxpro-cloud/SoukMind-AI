"""Evaluate SoukMind v0.1 on the existing synthetic sales CSV.

Run after: python src/train.py
Usage:     python src/evaluate.py

This script does not modify train.py and re-fits its own reproducible
ordinary least-squares model using the saved data and chronological split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sales.csv"
METRICS_FILE = ROOT / "reports" / "experiment_001_metrics.json"
FEATURES = ["price", "ad_budget", "discount_pct", "weekend"]
TARGET = "units_sold"


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    if not DATA_FILE.is_file():
        raise SystemExit("Missing data/sales.csv — run: python src/train.py")

    df = pd.read_csv(DATA_FILE)
    needed = ["date", *FEATURES, TARGET]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing CSV columns: {missing}")
    if df[needed].isna().any().any():
        raise SystemExit("CSV contains missing values in required columns")
    if len(df) < 10:
        raise SystemExit("Dataset is too small for this example split")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    split = int(0.8 * len(df))
    train, test = df.iloc[:split], df.iloc[split:]

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGET])

    train_predictions = model.predict(train[FEATURES])
    test_predictions = model.predict(test[FEATURES])
    baseline_train = np.full(len(train), train[TARGET].mean())
    baseline_test = np.full(len(test), train[TARGET].mean())

    model_train = metrics(train[TARGET], train_predictions)
    model_test = metrics(test[TARGET], test_predictions)
    baseline_test_metrics = metrics(test[TARGET], baseline_test)

    report = {
        "experiment": "001",
        "dataset": "synthetic retail sales",
        "total_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split_method": "chronological_80_20",
        "features": FEATURES,
        "target": TARGET,
        "baseline_train": metrics(train[TARGET], baseline_train),
        "baseline_test": baseline_test_metrics,
        "linear_regression_train": model_train,
        "linear_regression_test": model_test,
        "generalization_gap": {
            key: float(model_test[key] - model_train[key]) for key in ("MAE", "RMSE")
        },
        "intercept": float(model.intercept_),
        "coefficients": {key: float(coef) for key, coef in zip(FEATURES, model.coef_)},
        "caveat": "Synthetic data only; no real-world validation or causal interpretation.",
    }

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n===== SoukMind v0.1 — Experiment 001 =====")
    print(f"Rows: train={len(train)}, test={len(test)}")
    for label, result in [
        ("Baseline TEST", baseline_test_metrics),
        ("Linear TRAIN", model_train),
        ("Linear TEST", model_test),
    ]:
        print(f"{label:15s} | MAE={result['MAE']:.4f} | RMSE={result['RMSE']:.4f}")
    print(
        "Generalization gap (test - train): "
        f"MAE={report['generalization_gap']['MAE']:.4f}, "
        f"RMSE={report['generalization_gap']['RMSE']:.4f}"
    )
    print(f"Saved report: {METRICS_FILE}")


if __name__ == "__main__":
    main()
