"""
Phase 6 — Model Evaluation

Evaluates trained PhysioNet models on the held-out test split using:
AUROC, AUPRC, F1, Sensitivity, Specificity.

Inputs:
    results/models/lr_model.joblib
    results/models/lr_scaler.joblib
    results/models/xgb_model.joblib
    results/models/train_test_split.joblib

Outputs:
    results/tables/evaluation_metrics.csv
    results/tables/evaluation_metrics.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "results", "models")
TABLES_DIR = os.path.join(ROOT, "results", "tables")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.joblib")
LR_SCALER_PATH = os.path.join(MODELS_DIR, "lr_scaler.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
SPLIT_PATH = os.path.join(MODELS_DIR, "train_test_split.joblib")

CSV_OUT = os.path.join(TABLES_DIR, "evaluation_metrics.csv")
JSON_OUT = os.path.join(TABLES_DIR, "evaluation_metrics.json")


def sensitivity_specificity(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Return sensitivity (recall for positive class) and specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(sensitivity), float(specificity)


def evaluate_model(name: str, y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    sens, spec = sensitivity_specificity(y_true, y_pred)
    return {
        "model": name,
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": sens,
        "specificity": spec,
    }


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)

    required_paths = [LR_MODEL_PATH, LR_SCALER_PATH, XGB_MODEL_PATH, SPLIT_PATH]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run scripts/train_models.py first.\n"
            + "\n".join(missing)
        )

    print("=" * 60)
    print("PHASE 6 — Model Evaluation")
    print("=" * 60)

    lr_model = joblib.load(LR_MODEL_PATH)
    lr_scaler = joblib.load(LR_SCALER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    split = joblib.load(SPLIT_PATH)

    x_test = split["x_test"]
    y_test = split["y_test"]

    print(f"Loaded test split: {len(x_test):,} rows | Sepsis rate: {float(y_test.mean()):.1%}")

    # Logistic Regression (scaled features)
    x_test_scaled = lr_scaler.transform(x_test)
    lr_prob = lr_model.predict_proba(x_test_scaled)[:, 1]
    lr_pred = lr_model.predict(x_test_scaled)
    lr_metrics = evaluate_model("LogisticRegression", y_test, lr_prob, lr_pred)

    # XGBoost (unscaled features)
    xgb_prob = xgb_model.predict_proba(x_test)[:, 1]
    xgb_pred = xgb_model.predict(x_test)
    xgb_metrics = evaluate_model("XGBoost", y_test, xgb_prob, xgb_pred)

    df = pd.DataFrame([lr_metrics, xgb_metrics])
    df = df[["model", "auroc", "auprc", "f1", "sensitivity", "specificity"]]
    df.to_csv(CSV_OUT, index=False)

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print("\nEvaluation results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved:")
    print(f"  - {CSV_OUT}")
    print(f"  - {JSON_OUT}")
    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()
