import pandas as pd
import numpy as np

BASE = "eICU_dataset/"

PATIENT_FILE = BASE + "patient.csv"
VITAL_FILE = BASE + "vitalPeriodic.csv"
LAB_FILE = BASE + "lab.csv"
OUTPUT_FILE = BASE + "eicu_final_output.csv"

# -----------------------------
# Load files
# -----------------------------
patient_df = pd.read_csv(PATIENT_FILE)
vital_df = pd.read_csv(VITAL_FILE)
lab_df = pd.read_csv(LAB_FILE)

# -----------------------------
# Standardize key column
# -----------------------------
for df in [patient_df, vital_df, lab_df]:
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")

# -----------------------------
# Convert time columns
# -----------------------------
vital_df["charttime"] = pd.to_datetime(vital_df["charttime"], errors="coerce")
lab_df["charttime"] = pd.to_datetime(lab_df["charttime"], errors="coerce")

# Remove bad rows
patient_df = patient_df.dropna(subset=["patient_id"])
vital_df = vital_df.dropna(subset=["patient_id", "charttime"])
lab_df = lab_df.dropna(subset=["patient_id", "charttime"])

# -----------------------------
# Keep only needed lab values
# -----------------------------
lab_df["lab_name"] = lab_df["lab_name"].astype(str).str.strip().str.lower()
lab_df["lab_value"] = pd.to_numeric(lab_df["lab_value"], errors="coerce")

# Handle common lab-name variations
lab_name_map = {
    "wbc": "wbc",
    "white blood cells": "wbc",
    "wbc count": "wbc",
    "lactate": "lactate",
    "lactic acid": "lactate"
}

lab_df["lab_name_clean"] = lab_df["lab_name"].map(lab_name_map)
lab_filtered = lab_df[lab_df["lab_name_clean"].isin(["wbc", "lactate"])].copy()

# Convert long lab table to wide format
lab_wide = (
    lab_filtered.pivot_table(
        index=["patient_id", "charttime"],
        columns="lab_name_clean",
        values="lab_value",
        aggfunc="last"
    )
    .reset_index()
)

lab_wide.columns.name = None
lab_wide = lab_wide.rename(columns={"charttime": "charttime_lab"})

# -----------------------------
# Merge patient info into vitals
# -----------------------------
merged = vital_df.merge(patient_df, on="patient_id", how="left")

# -----------------------------
# Merge vitals with labs by patient_id
# then keep only labs at or before vital charttime
# -----------------------------
temp = merged.merge(
    lab_wide,
    on="patient_id",
    how="left"
)

# Keep only rows where lab time is earlier than or equal to vital time
temp = temp[temp["charttime_lab"].isna() | (temp["charttime_lab"] <= temp["charttime"])].copy()

# For each vital row, keep the latest earlier lab row
temp = temp.sort_values(["patient_id", "charttime", "charttime_lab"])
temp = temp.groupby(["patient_id", "charttime"], as_index=False).last()

# -----------------------------
# Rename columns to match Member 2 output
# -----------------------------
rename_map = {
    "HR": "heart_rate",
    "Resp": "resp_rate",
    "Temp": "temperature",
    "SBP": "sbp",
    "DBP": "dbp",
    "SpO2": "spo2",
}

temp = temp.rename(columns=rename_map)

# -----------------------------
# Final required columns
# -----------------------------
final_cols = [
    "patient_id",
    "heart_rate",
    "resp_rate",
    "temperature",
    "sbp",
    "dbp",
    "spo2",
    "wbc",
    "lactate"
]

final_df = temp[final_cols].copy()

# -----------------------------
# Convert patient_id to p000001 format
# -----------------------------
unique_patients = sorted(final_df["patient_id"].dropna().unique())
patient_id_map = {
    pid: f"p{idx:06d}" for idx, pid in enumerate(unique_patients, start=1)
}
final_df["patient_id"] = final_df["patient_id"].map(patient_id_map)

# -----------------------------
# Forward fill within each patient
# -----------------------------
feature_cols = [
    "heart_rate",
    "resp_rate",
    "temperature",
    "sbp",
    "dbp",
    "spo2",
    "wbc",
    "lactate"
]

final_df = final_df.sort_values(["patient_id"]).reset_index(drop=True)
final_df[feature_cols] = final_df.groupby("patient_id")[feature_cols].ffill()

# -----------------------------
# Median fill remaining missing values
# -----------------------------
for col in feature_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
    final_df[col] = final_df[col].fillna(final_df[col].median())

# -----------------------------
# Derive sepsis label
# -----------------------------
# Simple rule-based proxy:
# Infection/inflammation evidence:
#   - wbc > 12 or wbc < 4
#   - OR lactate > 2
#
# Organ dysfunction / instability:
#   - sbp < 90
#   - OR spo2 < 90
#   - OR resp_rate > 22
#   - OR heart_rate > 100
#
# If both are present -> sepsis_label = 1 else 0

infection_flag = (
    (final_df["wbc"] > 12) |
    (final_df["wbc"] < 4) |
    (final_df["lactate"] > 2)
)

organ_flag = (
    (final_df["sbp"] < 90) |
    (final_df["spo2"] < 90) |
    (final_df["resp_rate"] > 22) |
    (final_df["heart_rate"] > 100)
)

final_df["sepsis_label"] = np.where(infection_flag & organ_flag, 1, 0)

# -----------------------------
# Final column order
# -----------------------------
final_df = final_df[
    [
        "patient_id",
        "heart_rate",
        "resp_rate",
        "temperature",
        "sbp",
        "dbp",
        "spo2",
        "wbc",
        "lactate",
        "sepsis_label"
    ]
]

# -----------------------------
# Save output
# -----------------------------
final_df.to_csv(OUTPUT_FILE, index=False)

print("Done")
print("Saved to:", OUTPUT_FILE)
print("Shape:", final_df.shape)
print("\nSepsis label counts:")
print(final_df["sepsis_label"].value_counts(dropna=False))
print("\nSample output:")
print(final_df.head())