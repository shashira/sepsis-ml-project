"""
Phase 4 — Feature Engineering
Converts the time-series integrated dataset (one row per hour per patient)
into flat per-patient feature tables (one row per patient) suitable for ML.

Input  : data/processed/integrated_dataset.csv
Outputs (one file per dataset × time window):
  data/features/physionet_features_full.csv   — full pre-onset stay  (40 features)
  data/features/physionet_features_6h.csv     — last  6 h before onset (40 features)
  data/features/physionet_features_12h.csv    — last 12 h before onset (40 features)
  data/features/physionet_features_24h.csv    — last 24 h before onset (40 features)
  data/features/eicu_features_full.csv
  data/features/eicu_features_6h.csv
  data/features/eicu_features_12h.csv
  data/features/eicu_features_24h.csv

Each file has 40 features: 8 vitals × 5 statistics (mean, max, min, std, slope).

The four files are used to train SEPARATE models so that performance can be
compared across time horizons — this is the temporal analysis described in
Phase 6. A high AUROC at 24h means the model can predict sepsis one day early.

Temporal leakage prevention
----------------------------
For sepsis patients the row-level sepsis_label switches from 0 → 1 at onset.
All feature windows are computed from PRE-ONSET rows only, so the model
never sees physiological values recorded during active sepsis.
For non-sepsis patients all rows are used.

Usage:
    python scripts/feature_engineering.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT, "data", "processed", "integrated_dataset.csv")
FEATURES_DIR = os.path.join(ROOT, "data", "features")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "heart_rate", "resp_rate", "temperature",
    "sbp", "dbp", "spo2", "wbc", "lactate",
]

# (label, window_size_hours)  None = full pre-onset stay
WINDOWS = [
    ("full", None),
    ("6h",   6),
    ("12h",  12),
    ("24h",  24),
]

METADATA_COLS = ["patient_id", "source", "sepsis_label"]

# ---------------------------------------------------------------------------
# Core statistics for a single Series
# ---------------------------------------------------------------------------

def _vital_stats(series: pd.Series, col: str, suffix: str) -> dict:
    """
    Return mean / max / min / std / slope for one vital-sign window.
    slope = mean of consecutive differences (measures trend direction).
    NaN std (single-row window) and NaN slope are filled with 0.
    """
    slope = series.diff().mean()
    std   = series.std()
    return {
        f"{col}_mean{suffix}":  series.mean(),
        f"{col}_max{suffix}":   series.max(),
        f"{col}_min{suffix}":   series.min(),
        f"{col}_std{suffix}":   0.0 if pd.isna(std)   else std,
        f"{col}_slope{suffix}": 0.0 if pd.isna(slope) else slope,
    }


# ---------------------------------------------------------------------------
# Per-patient feature extraction
# ---------------------------------------------------------------------------

def extract_patient_features(group: pd.DataFrame, window_label: str,
                             n_hours: int | None) -> dict:
    """
    group        : all rows for one patient in temporal order (one row = one ICU hour)
    window_label : 'full' | '6h' | '12h' | '24h'  — used only for diagnostics
    n_hours      : None means use the full pre-onset stay; otherwise last N rows

    Returns a flat dict with 40 numeric features + metadata columns.
    40 = 8 vitals × 5 statistics (mean, max, min, std, slope)
    """
    pid    = group["patient_id"].iloc[0]
    source = group["source"].iloc[0]

    # Patient-level label: 1 if sepsis ever occurred, else 0
    patient_label = int(group["sepsis_label"].max())

    # ---------------------------------------------------------------
    # Build the pre-onset time-series (leakage-free)
    # For sepsis patients: use only rows where sepsis_label == 0
    # (the hours before onset).  For non-sepsis patients: all rows.
    # ---------------------------------------------------------------
    if patient_label == 1:
        onset_pos = int(np.argmax(group["sepsis_label"].values))
        pre_onset = group.iloc[:onset_pos]
        # Edge case: onset on the very first row — fall back to full stay
        if len(pre_onset) == 0:
            pre_onset = group
    else:
        pre_onset = group

    # ---------------------------------------------------------------
    # Apply the time window (last N hours of the pre-onset period)
    # ---------------------------------------------------------------
    if n_hours is None:
        window = pre_onset
    else:
        window = pre_onset.tail(n_hours)
        if len(window) == 0:
            window = pre_onset   # defensive fallback

    features = {
        "patient_id":      pid,
        "source":          source,
        "sepsis_label":    patient_label,
        "stay_hours":      len(group),
        "pre_onset_hours": len(pre_onset),
        "window_hours":    len(window),
    }

    # 40 features: 8 vitals × 5 statistics
    for col in FEATURE_COLS:
        features.update(_vital_stats(window[col], col, ""))

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_table(df: pd.DataFrame, source_name: str,
                        window_label: str, n_hours: int | None) -> pd.DataFrame:
    """
    Process all patients for one source + one time window.
    Returns a DataFrame with 40 features per patient.
    """
    subset         = df[df["source"] == source_name].copy()
    patient_groups = list(subset.groupby("patient_id", sort=False))
    n_patients     = len(patient_groups)

    rows = []
    for _, group in tqdm(patient_groups,
                         desc=f"  {source_name}/{window_label}",
                         unit="patient"):
        rows.append(extract_patient_features(group, window_label, n_hours))

    feature_df = pd.DataFrame(rows)

    # Metadata first, then features alphabetically
    meta_cols = [c for c in ["patient_id", "source", "stay_hours",
                              "pre_onset_hours", "window_hours", "sepsis_label"]
                 if c in feature_df.columns]
    feat_cols = sorted([c for c in feature_df.columns if c not in meta_cols])
    return feature_df[meta_cols + feat_cols]


def print_summary(df: pd.DataFrame, name: str) -> None:
    diag_cols  = ["patient_id", "source", "stay_hours",
                  "pre_onset_hours", "window_hours", "sepsis_label"]
    feat_cols  = [c for c in df.columns if c not in diag_cols]
    n_patients = len(df)
    n_sepsis   = int(df["sepsis_label"].sum())
    rate       = n_sepsis / n_patients
    missing    = df[feat_cols].isnull().any().any()

    print(f"  Shape       : {df.shape}  "
          f"({n_patients:,} patients × {len(feat_cols)} features)")
    print(f"  Sepsis rate : {n_sepsis:,}/{n_patients:,}  ({rate:.1%})")
    print(f"  Missing vals: {'YES' if missing else 'None'}")


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("=" * 60)
    print("PHASE 4 — Feature Engineering")
    print("=" * 60)
    print("Each output file: 40 features per patient (8 vitals × 5 stats)")
    print("Four files per dataset — one per time window for temporal analysis")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"\n[1/2] Loading integrated dataset ...")
    if not os.path.exists(INPUT_PATH):
        print(f"  [ERROR] File not found: {INPUT_PATH}", file=sys.stderr)
        print(f"  Run scripts/integrate_datasets.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {len(df):,} rows  |  {df['patient_id'].nunique():,} patients")

    # ------------------------------------------------------------------
    # Build one feature table per source × window
    # ------------------------------------------------------------------
    print(f"\n[2/2] Building feature tables ...")

    diag_cols  = {"patient_id", "source", "stay_hours",
                  "pre_onset_hours", "window_hours", "sepsis_label"}
    saved_files = []

    for source in ["physionet", "eicu"]:
        print(f"\n  --- {source.upper()} ---")
        last_pn_feat_cols = None

        for win_label, n_hours in WINDOWS:
            out_path = os.path.join(ROOT, "data",
                                    "features", f"{source}_features_{win_label}.csv")
            feat_df  = build_feature_table(df, source, win_label, n_hours)
            feat_df.to_csv(out_path, index=False)

            feat_cols = [c for c in feat_df.columns if c not in diag_cols]
            print(f"\n  Window [{win_label:>4}]  →  {out_path.split('data/')[-1]}")
            print_summary(feat_df, f"{source}/{win_label}")

            # Track feature columns for cross-dataset alignment check
            if source == "physionet" and win_label == "full":
                pn_full_feat_cols = set(feat_cols)
            if source == "eicu" and win_label == "full":
                eicu_full_feat_cols = set(feat_cols)

            saved_files.append(out_path)

    # ------------------------------------------------------------------
    # Cross-dataset column alignment check (critical for Phase 8)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Cross-dataset column alignment check (full window) ...")
    if pn_full_feat_cols == eicu_full_feat_cols:
        print(f"  [OK] Identical feature columns across both datasets "
              f"({len(pn_full_feat_cols)} features)")
    else:
        only_pn   = pn_full_feat_cols   - eicu_full_feat_cols
        only_eicu = eicu_full_feat_cols - pn_full_feat_cols
        if only_pn:
            print(f"  [WARN] Only in PhysioNet : {sorted(only_pn)}")
        if only_eicu:
            print(f"  [WARN] Only in eICU      : {sorted(only_eicu)}")

    print(f"\nFeature engineering complete.  8 files saved to data/features/")
    print(f"  physionet_features_full.csv  ← use this for the main model (Phase 5)")
    print(f"  physionet_features_6h.csv")
    print(f"  physionet_features_12h.csv")
    print(f"  physionet_features_24h.csv   ← earliest prediction horizon")
    print(f"  eicu_features_*.csv          ← same structure, used in Phase 9")


if __name__ == "__main__":
    main()
