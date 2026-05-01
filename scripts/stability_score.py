"""
Phase 8 — Stability Score

Computes explanation stability using:
1) Top-K feature overlap
2) Spearman rank correlation
Final Stability Score = (overlap + spearman_rho) / 2

The script supports all 4 target comparisons and computes whichever inputs
exist in results/tables. Missing comparisons are marked as PENDING.

Inputs expected (from Phase 7):
    results/tables/shap_importance_lr_physionet.csv
    results/tables/shap_importance_xgb_physionet.csv
    results/tables/shap_importance_lr_eicu.csv          (optional now)
    results/tables/shap_importance_xgb_eicu.csv         (optional now)

Outputs:
    results/tables/stability_scores.csv
    results/tables/stability_scores.json
"""

import json
import os
from typing import Optional

import pandas as pd
from scipy.stats import spearmanr


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(ROOT, "results", "tables")

PATHS = {
    "lr_physionet": os.path.join(TABLES_DIR, "shap_importance_lr_physionet.csv"),
    "xgb_physionet": os.path.join(TABLES_DIR, "shap_importance_xgb_physionet.csv"),
    "lr_eicu": os.path.join(TABLES_DIR, "shap_importance_lr_eicu.csv"),
    "xgb_eicu": os.path.join(TABLES_DIR, "shap_importance_xgb_eicu.csv"),
}

CSV_OUT = os.path.join(TABLES_DIR, "stability_scores.csv")
JSON_OUT = os.path.join(TABLES_DIR, "stability_scores.json")


def load_importance(path: str) -> Optional[pd.Series]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Files saved via Series.to_csv(header=True) produce columns:
    # unnamed feature column + mean_abs_shap
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected importance file format: {path}")
    feature_col = df.columns[0]
    value_col = df.columns[1]
    series = pd.Series(df[value_col].values, index=df[feature_col].astype(str).values)
    return series.sort_values(ascending=False)


def stability_score(importance_a: pd.Series, importance_b: pd.Series, k: int = 10) -> dict:
    top_k_a = set(importance_a.head(k).index)
    top_k_b = set(importance_b.head(k).index)
    overlap = len(top_k_a & top_k_b) / k

    all_features = importance_a.index.union(importance_b.index)
    a_aligned = importance_a.reindex(all_features, fill_value=0)
    b_aligned = importance_b.reindex(all_features, fill_value=0)
    rho, _ = spearmanr(a_aligned, b_aligned)
    rho = float(0.0 if pd.isna(rho) else rho)

    return {
        "overlap": float(overlap),
        "spearman_rho": rho,
        "stability_score": float((overlap + rho) / 2),
    }


def build_row(label: str, a: Optional[pd.Series], b: Optional[pd.Series], k: int = 10) -> dict:
    if a is None or b is None:
        return {
            "comparison": label,
            "status": "PENDING",
            "top_k": k,
            "overlap": None,
            "spearman_rho": None,
            "stability_score": None,
        }
    score = stability_score(a, b, k=k)
    return {
        "comparison": label,
        "status": "DONE",
        "top_k": k,
        "overlap": score["overlap"],
        "spearman_rho": score["spearman_rho"],
        "stability_score": score["stability_score"],
    }


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("=" * 60)
    print("PHASE 8 — Stability Score")
    print("=" * 60)

    imp = {name: load_importance(path) for name, path in PATHS.items()}

    rows = [
        build_row("LR vs XGBoost (PhysioNet)", imp["lr_physionet"], imp["xgb_physionet"]),
        build_row("LR vs XGBoost (eICU)", imp["lr_eicu"], imp["xgb_eicu"]),
        build_row("XGBoost: PhysioNet vs eICU", imp["xgb_physionet"], imp["xgb_eicu"]),
        build_row("LR: PhysioNet vs eICU", imp["lr_physionet"], imp["lr_eicu"]),
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CSV_OUT, index=False)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nStability matrix:")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    done = int((out_df["status"] == "DONE").sum())
    pending = int((out_df["status"] == "PENDING").sum())

    print(f"\nComputed: {done} comparison(s) | Pending: {pending}")
    print(f"Saved:")
    print(f"  - {CSV_OUT}")
    print(f"  - {JSON_OUT}")

    if done > 0:
        completed = out_df[out_df["status"] == "DONE"].copy()
        completed = completed.sort_values("stability_score", ascending=False)
        best = completed.iloc[0]
        print(
            f"\nBest current stability: {best['comparison']} "
            f"-> {best['stability_score']:.4f}"
        )

    print("\nPhase 8 complete.")


if __name__ == "__main__":
    main()
