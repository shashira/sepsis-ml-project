"""
Phase 3 — Integration & Validation
Member 4: Integration Lead

Combines the PhysioNet and eICU cleaned datasets into a single unified dataset.
Runs schema validation, fixes unit inconsistencies, handles missing labels,
and produces a side-by-side summary table for the presentation.

Usage:
    python scripts/integrate_datasets.py

Output:
    data/processed/integrated_dataset.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHYSIONET_PATH_CANDIDATES = [
    os.path.join(ROOT, "data", "raw", "physionet_cleaned.csv"),                   # standardized location
    os.path.join(ROOT, "scripts", "physionet", "physionet_cleaned.csv"),           # legacy location
]
OUTPUT_PATH    = os.path.join(ROOT, "data", "processed", "integrated_dataset.csv")
EICU_PATH_CANDIDATES = [
    os.path.join(ROOT, "data", "raw", "eicu_final_output.csv"),                    # standardized location
    os.path.join(ROOT, "scripts", "eICU", "eicu_dataset", "eicu_final_output.csv"),  # current repo layout
    os.path.join(ROOT, "eicu_final_output.csv"),                                      # legacy layout
]

# ---------------------------------------------------------------------------
# Column definitions (shared with constants.py)
# ---------------------------------------------------------------------------
FEATURE_COLS   = ["heart_rate", "resp_rate", "temperature", "sbp", "dbp", "spo2", "wbc", "lactate"]
EXPECTED_COLS  = ["patient_id"] + FEATURE_COLS + ["sepsis_label"]
FINAL_COL_ORDER = ["patient_id", "source"] + FEATURE_COLS + ["sepsis_label"]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, name: str) -> None:
    """Raise if any expected column is missing or non-numeric feature found."""
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

    for col in FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"[{name}] Column '{col}' must be numeric, got {df[col].dtype}")

    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    print(f"  [OK] Schema valid — {len(df.columns)} columns present")
    if extra:
        print(f"  [INFO] Extra columns found (will be dropped): {extra}")


def check_missing(df: pd.DataFrame, name: str) -> None:
    """Report missing value percentages per feature column."""
    missing_pct = df[FEATURE_COLS].isnull().mean() * 100
    any_missing = (missing_pct > 0).any()
    if any_missing:
        print(f"  [WARN] Missing values detected in [{name}]:")
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"         {col}: {pct:.2f}%")
    else:
        print(f"  [OK] No missing values in feature columns")


def check_temperature_unit(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Detect Fahrenheit temperatures and convert to Celsius.
    Human core temperature in Celsius is ~36–38°C; in Fahrenheit ~96–100°F.
    Any mean above 50 reliably indicates Fahrenheit.
    """
    temp_mean = df["temperature"].mean()
    if temp_mean > 50:
        print(f"  [FIX] Temperature is in Fahrenheit (mean={temp_mean:.2f}°F). Converting to Celsius.")
        df = df.copy()
        df["temperature"] = (df["temperature"] - 32.0) * 5.0 / 9.0
        print(f"  [OK] Converted. New mean: {df['temperature'].mean():.2f}°C")
    else:
        print(f"  [OK] Temperature is in Celsius (mean={temp_mean:.2f}°C)")
    return df


def check_physiological_ranges(df: pd.DataFrame, name: str) -> None:
    """
    Warn about values outside expected physiological ranges.
    These are coarse sanity checks, not filters.
    """
    ranges = {
        "heart_rate":   (20,  300),
        "resp_rate":    (4,   60),
        "temperature":  (32,  42),   # Celsius
        "sbp":          (40,  300),
        "dbp":          (10,  200),
        "spo2":         (50,  100),
        "wbc":          (0,   100),
        "lactate":      (0,   30),
    }
    any_outliers = False
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            pct = n_out / len(df) * 100
            print(f"  [WARN] {col}: {n_out:,} rows ({pct:.2f}%) outside [{lo}, {hi}]")
            any_outliers = True
    if not any_outliers:
        print(f"  [OK] All features within expected physiological ranges")


# ---------------------------------------------------------------------------
# eICU-specific: derive sepsis label via qSOFA proxy
# ---------------------------------------------------------------------------

def derive_eicu_sepsis_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    eICU does not include a direct sepsis label in the exported CSV.
    We derive a proxy row-level label using two of the three qSOFA criteria
    available in the vitals data:

        qSOFA criterion 1: Respiratory rate >= 22 breaths/min
        qSOFA criterion 2: Systolic BP <= 100 mmHg
        (criterion 3 — altered mentation — requires GCS, not available here)

    A row is flagged sepsis=1 when BOTH available criteria are met (qSOFA >= 2).

    Limitation: This is a clinical proxy, not a ground-truth label.
    It will under-detect sepsis patients who present with altered mentation
    alone, and may over-detect patients in haemorrhagic shock.
    Label should be replaced with ICD-9/ICD-10 diagnosis codes when available.
    """
    print("\n  [WARN] eICU 'sepsis_label' column is entirely NaN.")
    print("  [INFO] Deriving proxy label using qSOFA criteria:")
    print("         - resp_rate >= 22  AND")
    print("         - sbp <= 100")
    print("         (Row flagged sepsis=1 when both criteria are met)")

    df = df.copy()
    qsofa_positive = (df["resp_rate"] >= 22) & (df["sbp"] <= 100)
    df["sepsis_label"] = qsofa_positive.astype(int)

    # Patient-level stats for reporting
    patient_labels = df.groupby("patient_id")["sepsis_label"].max()
    n_sepsis  = int(patient_labels.sum())
    n_total   = len(patient_labels)
    rate      = n_sepsis / n_total

    print(f"\n  [INFO] qSOFA-derived label distribution:")
    print(f"         Sepsis patients (ever qSOFA>=2): {n_sepsis}/{n_total} ({rate:.1%})")
    print(f"         Non-sepsis patients:             {n_total - n_sepsis}/{n_total} ({1-rate:.1%})")

    return df


# ---------------------------------------------------------------------------
# Patient ID prefixing — prevents collisions between datasets
# ---------------------------------------------------------------------------

def prefix_patient_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Both datasets use the same p000001, p000002, ... ID scheme.
    Prefix with dataset source to guarantee uniqueness in the merged table.
    Example: p000001 -> pn_p000001  or  eicu_p000001
    """
    df = df.copy()
    df["patient_id"] = prefix + "_" + df["patient_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-dataset validation summary
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame, name: str) -> dict:
    """Return a summary dict for the presentation comparison table."""
    patient_labels = df.groupby("patient_id")["sepsis_label"].max()
    n_patients = df["patient_id"].nunique()
    n_rows     = len(df)
    n_sepsis   = int(patient_labels.sum())
    rate       = n_sepsis / n_patients if n_patients > 0 else 0.0
    n_features = len(FEATURE_COLS)
    missing_any = df[FEATURE_COLS].isnull().any().any()

    return {
        "Dataset":          name,
        "Rows":             f"{n_rows:,}",
        "Patients":         f"{n_patients:,}",
        "Features":         n_features,
        "Sepsis Patients":  f"{n_sepsis:,}",
        "Sepsis Rate":      f"{rate:.1%}",
        "Missing Values":   "Yes" if missing_any else "None",
    }


def resolve_eicu_path() -> str:
    """Return first existing eICU CSV path from known locations."""
    for path in EICU_PATH_CANDIDATES:
        if os.path.exists(path):
            return path
    return EICU_PATH_CANDIDATES[0]


def resolve_physionet_path() -> str:
    """Return first existing PhysioNet cleaned CSV path from known locations."""
    for path in PHYSIONET_PATH_CANDIDATES:
        if os.path.exists(path):
            return path
    return PHYSIONET_PATH_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
    eicu_path = resolve_eicu_path()
    physionet_path = resolve_physionet_path()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 3 — Dataset Integration & Validation")
    print("=" * 60)

    print(f"\n[1/7] Loading datasets...")
    for path, label in [(physionet_path, "PhysioNet"), (eicu_path, "eICU")]:
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)

    pn   = pd.read_csv(physionet_path)
    eicu = pd.read_csv(eicu_path)
    print(f"  PhysioNet loaded : {pn.shape[0]:,} rows, {pn.shape[1]} columns")
    print(f"  eICU loaded      : {eicu.shape[0]:,} rows, {eicu.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Schema validation
    # ------------------------------------------------------------------
    print(f"\n[2/7] Validating schemas...")
    validate_schema(pn,   "PhysioNet")
    validate_schema(eicu, "eICU")

    # Drop any extra columns, keep only expected
    pn   = pn[EXPECTED_COLS].copy()
    eicu = eicu[EXPECTED_COLS].copy()

    # ------------------------------------------------------------------
    # 3. Missing value check
    # ------------------------------------------------------------------
    print(f"\n[3/7] Checking missing values...")
    check_missing(pn,   "PhysioNet")
    check_missing(eicu, "eICU")

    # ------------------------------------------------------------------
    # 4. Temperature unit normalization
    # ------------------------------------------------------------------
    print(f"\n[4/7] Normalizing temperature units...")
    pn   = check_temperature_unit(pn,   "PhysioNet")
    eicu = check_temperature_unit(eicu, "eICU")

    # ------------------------------------------------------------------
    # 5. Derive eICU sepsis label if missing
    # ------------------------------------------------------------------
    print(f"\n[5/7] Checking sepsis labels...")
    if eicu["sepsis_label"].isnull().all():
        eicu = derive_eicu_sepsis_label(eicu)
    else:
        print(f"  [OK] eICU sepsis_label present")

    pn["sepsis_label"]   = pn["sepsis_label"].astype(int)
    eicu["sepsis_label"] = eicu["sepsis_label"].astype(int)

    pn_label_rate   = pn.groupby("patient_id")["sepsis_label"].max().mean()
    eicu_label_rate = eicu.groupby("patient_id")["sepsis_label"].max().mean()
    print(f"\n  PhysioNet sepsis rate (patient-level): {pn_label_rate:.1%}")
    print(f"  eICU      sepsis rate (patient-level): {eicu_label_rate:.1%}")

    # Warn if sepsis rate is far outside expected ~10%
    for label, rate in [("PhysioNet", pn_label_rate), ("eICU", eicu_label_rate)]:
        if rate < 0.03 or rate > 0.35:
            print(f"  [WARN] {label} sepsis rate ({rate:.1%}) is outside expected 3–35% range.")

    # ------------------------------------------------------------------
    # 6. Physiological range checks (post unit-fix)
    # ------------------------------------------------------------------
    print(f"\n[6/7] Running physiological range checks...")
    print(f"  --- PhysioNet ---")
    check_physiological_ranges(pn,   "PhysioNet")
    print(f"  --- eICU ---")
    check_physiological_ranges(eicu, "eICU")

    # ------------------------------------------------------------------
    # 7. Prefix patient IDs and concatenate
    # ------------------------------------------------------------------
    print(f"\n[7/7] Integrating datasets...")
    pn   = prefix_patient_ids(pn,   "pn")
    eicu = prefix_patient_ids(eicu, "eicu")

    pn["source"]   = "physionet"
    eicu["source"] = "eicu"

    integrated = pd.concat([pn, eicu], ignore_index=True)
    integrated = integrated[FINAL_COL_ORDER]

    integrated.to_csv(OUTPUT_PATH, index=False)
    print(f"  Integrated dataset saved → {OUTPUT_PATH}")
    print(f"  Total rows: {len(integrated):,}  |  Total patients: {integrated['patient_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Presentation summary table
    # ------------------------------------------------------------------
    pn_sum   = build_summary(pn,          "PhysioNet")
    eicu_sum = build_summary(eicu,        "eICU")
    int_sum  = build_summary(integrated,  "Integrated")

    summary_df = pd.DataFrame([pn_sum, eicu_sum, int_sum])

    print(f"\n{'=' * 60}")
    print("COMBINED DATASET SUMMARY  (use this for your presentation slide)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print()

    # Column name consistency check
    pn_cols   = set(pn.drop(columns=["source"]).columns)
    eicu_cols = set(eicu.drop(columns=["source"]).columns)
    if pn_cols == eicu_cols:
        print("[OK] Column names are identical across both datasets")
    else:
        only_pn   = pn_cols   - eicu_cols
        only_eicu = eicu_cols - pn_cols
        if only_pn:
            print(f"[WARN] Columns only in PhysioNet: {only_pn}")
        if only_eicu:
            print(f"[WARN] Columns only in eICU: {only_eicu}")

    print(f"\nIntegration complete.")


if __name__ == "__main__":
    main()
