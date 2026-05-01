"""
Phase 5 — Model Training

Trains two baseline models on PhysioNet engineered features:
1) Logistic Regression (with scaling + class_weight="balanced")
2) XGBoost (with scale_pos_weight for class imbalance)

Default input:
    data/features/physionet_features_full.csv

Outputs:
    results/models/lr_model.joblib
    results/models/lr_scaler.joblib
    results/models/xgb_model.joblib
    results/models/feature_columns.json
    results/models/train_test_split.joblib
    results/tables/training_summary.json
"""

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "data", "features", "physionet_features_full.csv")
MODELS_DIR = os.path.join(ROOT, "results", "models")
TABLES_DIR = os.path.join(ROOT, "results", "tables")

TARGET_COL = "sepsis_label"
DROP_COLS = {"patient_id", "source", "stay_hours", "pre_onset_hours", "window_hours", TARGET_COL}
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class ModelMetrics:
    auroc: float
    auprc: float
    f1: float


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found after dropping metadata columns.")
    x = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()
    return x, y


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        auroc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
    )


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("=" * 60)
    print("PHASE 5 — Model Training")
    print("=" * 60)
    print(f"\nLoading features from: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run scripts/feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    x, y = build_xy(df)

    print(f"Rows: {len(df):,} | Features: {x.shape[1]} | Sepsis rate: {y.mean():.1%}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(x_train):,} rows | Test: {len(x_test):,} rows")

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    lr.fit(x_train_scaled, y_train)

    lr_prob = lr.predict_proba(x_test_scaled)[:, 1]
    lr_pred = lr.predict(x_test_scaled)
    lr_metrics = evaluate_predictions(y_test, lr_prob, lr_pred)
    print(
        f"LR metrics -> AUROC: {lr_metrics.auroc:.4f}, "
        f"AUPRC: {lr_metrics.auprc:.4f}, F1: {lr_metrics.f1:.4f}"
    )

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    print("\nTraining XGBoost...")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        random_state=RANDOM_STATE,
    )
    xgb.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=25,
    )

    xgb_prob = xgb.predict_proba(x_test)[:, 1]
    xgb_pred = xgb.predict(x_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_prob, xgb_pred)
    print(
        f"XGB metrics -> AUROC: {xgb_metrics.auroc:.4f}, "
        f"AUPRC: {xgb_metrics.auprc:.4f}, F1: {xgb_metrics.f1:.4f}"
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    joblib.dump(lr, os.path.join(MODELS_DIR, "lr_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "lr_scaler.joblib"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(x.columns)}, f, indent=2)

    split_artifact = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    joblib.dump(split_artifact, os.path.join(MODELS_DIR, "train_test_split.joblib"))

    summary = {
        "input_file": INPUT_PATH,
        "n_rows": int(len(df)),
        "n_features": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_sepsis_rate": float(y_train.mean()),
        "test_sepsis_rate": float(y_test.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "logistic_regression": asdict(lr_metrics),
        "xgboost": asdict(xgb_metrics),
    }
    summary_path = os.path.join(TABLES_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved artifacts:")
    print(f"  - {os.path.join(MODELS_DIR, 'lr_model.joblib')}")
    print(f"  - {os.path.join(MODELS_DIR, 'lr_scaler.joblib')}")
    print(f"  - {os.path.join(MODELS_DIR, 'xgb_model.joblib')}")
    print(f"  - {os.path.join(MODELS_DIR, 'feature_columns.json')}")
    print(f"  - {os.path.join(MODELS_DIR, 'train_test_split.joblib')}")
    print(f"  - {summary_path}")
    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
