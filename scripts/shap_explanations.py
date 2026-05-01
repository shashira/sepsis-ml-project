"""
Phase 7 — SHAP Explanations

Generates SHAP explanations for trained PhysioNet models:
1) XGBoost using shap.TreeExplainer
2) Logistic Regression using shap.LinearExplainer

Inputs:
    results/models/lr_model.joblib
    results/models/lr_scaler.joblib
    results/models/xgb_model.joblib
    results/models/train_test_split.joblib

Outputs:
    results/tables/shap_importance_lr_physionet.csv
    results/tables/shap_importance_xgb_physionet.csv
    results/tables/shap_top10_lr_physionet.csv
    results/tables/shap_top10_xgb_physionet.csv
    results/tables/shap_values_lr_physionet.npy
    results/tables/shap_values_xgb_physionet.npy
    results/figures/shap_lr_physionet.png
    results/figures/shap_xgb_physionet.png
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "results", "models")
TABLES_DIR = os.path.join(ROOT, "results", "tables")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.joblib")
LR_SCALER_PATH = os.path.join(MODELS_DIR, "lr_scaler.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
SPLIT_PATH = os.path.join(MODELS_DIR, "train_test_split.joblib")


def mean_abs_shap_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    values = np.abs(shap_values).mean(axis=0)
    return pd.Series(values, index=feature_names, name="mean_abs_shap").sort_values(ascending=False)


def save_bar_plot(series: pd.Series, out_path: str, title: str, top_k: int = 10) -> None:
    top = series.head(top_k).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top.index, top.values)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    required_paths = [LR_MODEL_PATH, LR_SCALER_PATH, XGB_MODEL_PATH, SPLIT_PATH]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run scripts/train_models.py first.\n"
            + "\n".join(missing)
        )

    print("=" * 60)
    print("PHASE 7 — SHAP Explanations")
    print("=" * 60)

    lr_model = joblib.load(LR_MODEL_PATH)
    lr_scaler = joblib.load(LR_SCALER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    split = joblib.load(SPLIT_PATH)

    x_train = split["x_train"]
    x_test = split["x_test"]

    feature_names = list(x_test.columns)
    x_train_scaled = lr_scaler.transform(x_train)
    x_test_scaled = lr_scaler.transform(x_test)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=feature_names, index=x_test.index)

    # ------------------------------------------------------------------
    # XGBoost SHAP (TreeExplainer)
    # ------------------------------------------------------------------
    print("\nComputing SHAP for XGBoost...")
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(x_test)
    xgb_importance = mean_abs_shap_importance(np.array(xgb_shap_values), feature_names)
    xgb_top10 = xgb_importance.head(10)

    xgb_importance.to_csv(os.path.join(TABLES_DIR, "shap_importance_xgb_physionet.csv"), header=True)
    xgb_top10.to_csv(os.path.join(TABLES_DIR, "shap_top10_xgb_physionet.csv"), header=True)
    np.save(os.path.join(TABLES_DIR, "shap_values_xgb_physionet.npy"), np.array(xgb_shap_values))
    save_bar_plot(
        xgb_importance,
        os.path.join(FIGURES_DIR, "shap_xgb_physionet.png"),
        "PhysioNet — XGBoost Top SHAP Features",
    )
    print("Top 10 XGBoost features:")
    print(xgb_top10.to_string(float_format=lambda x: f"{x:.6f}"))

    # ------------------------------------------------------------------
    # Logistic Regression SHAP (LinearExplainer)
    # ------------------------------------------------------------------
    print("\nComputing SHAP for Logistic Regression...")
    lr_explainer = shap.LinearExplainer(lr_model, x_train_scaled, feature_perturbation="interventional")
    lr_shap_values = lr_explainer.shap_values(x_test_scaled)
    lr_importance = mean_abs_shap_importance(np.array(lr_shap_values), feature_names)
    lr_top10 = lr_importance.head(10)

    lr_importance.to_csv(os.path.join(TABLES_DIR, "shap_importance_lr_physionet.csv"), header=True)
    lr_top10.to_csv(os.path.join(TABLES_DIR, "shap_top10_lr_physionet.csv"), header=True)
    np.save(os.path.join(TABLES_DIR, "shap_values_lr_physionet.npy"), np.array(lr_shap_values))
    save_bar_plot(
        lr_importance,
        os.path.join(FIGURES_DIR, "shap_lr_physionet.png"),
        "PhysioNet — Logistic Regression Top SHAP Features",
    )
    print("Top 10 Logistic Regression features:")
    print(lr_top10.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nSaved outputs:")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_importance_xgb_physionet.csv')}")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_importance_lr_physionet.csv')}")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_top10_xgb_physionet.csv')}")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_top10_lr_physionet.csv')}")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_values_xgb_physionet.npy')}")
    print(f"  - {os.path.join(TABLES_DIR, 'shap_values_lr_physionet.npy')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'shap_xgb_physionet.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'shap_lr_physionet.png')}")
    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
