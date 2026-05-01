# Temporal Explainable ML for Early Sepsis Detection
## Implementation Guide

> **Core deliverable:** A *Stability Score* — a novel metric that quantifies how consistently two models (or two datasets) agree on which features drive sepsis predictions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Phase 1 — Environment Setup](#phase-1--environment-setup)
4. [Phase 2 — Data Collection](#phase-2--data-collection)
5. [Phase 3 — Data Loading & Cleaning](#phase-3--data-loading--cleaning)
6. [Phase 4 — Feature Engineering](#phase-4--feature-engineering)
7. [Phase 5 — Model Training](#phase-5--model-training)
8. [Phase 6 — Model Evaluation](#phase-6--model-evaluation)
9. [Phase 7 — SHAP Explanations](#phase-7--shap-explanations)
10. [Phase 8 — Stability Score](#phase-8--stability-score)
11. [Phase 9 — Repeat on eICU](#phase-9--repeat-on-eicu)
12. [Phase 10 — Visualizations](#phase-10--visualizations)
13. [Team Division & Timeline](#team-division--timeline)
14. [Key Rules](#key-rules)

---

## Project Overview

| Item | Detail |
|------|--------|
| Goal | Predict early sepsis onset from ICU vitals and explain model decisions |
| Datasets | PhysioNet Sepsis Challenge 2019 + eICU Collaborative Research Database |
| Models | Logistic Regression, XGBoost (+ optional LSTM) |
| Novel contribution | Stability Score computed from SHAP explanations |
| Primary metric | AUROC (never raw accuracy — data is imbalanced) |

---

## Folder Structure

```
sepsis-ml-project/
├── data/
│   ├── raw/
│   │   ├── physionet_raw.csv
│   │   ├── physionet_cleaned.csv
│   │   └── eicu_final_output.csv
│   ├── processed/
│   │   └── integrated_dataset.csv    ← Phase 3 output (unified)
│   └── features/                     ← Phase 4 outputs
│       ├── physionet_features_full.csv
│       ├── physionet_features_6h.csv
│       ├── physionet_features_12h.csv
│       ├── physionet_features_24h.csv
│       ├── eicu_features_full.csv
│       ├── eicu_features_6h.csv
│       ├── eicu_features_12h.csv
│       └── eicu_features_24h.csv
├── scripts/
│   ├── physionet/
│   │   ├── load_physionet.py
│   │   ├── clean_physionet.py
│   │   ├── physionet_raw.csv
│   │   ├── physionet_cleaned.csv
│   │   └── physionet_labels.csv
│   ├── utils/
│   │   ├── constants.py              ← shared column names
│   │   └── preprocessing_utils.py   ← shared ffill / median fill
│   ├── integrate_datasets.py         ← Phase 3 ✓ complete
│   ├── feature_engineering.py        ← Phase 4 ✓ complete
│   ├── train_models.py               ← Phase 5 ✓ complete
│   ├── evaluate_models.py            ← Phase 6 ✓ complete
│   ├── shap_explanations.py          ← Phase 7 ✓ complete
│   └── stability_score.py            ← Phase 8 ✓ complete
├── notebooks/
│   └── pipeline_audit.ipynb          ← end-to-end audit/demo notebook
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── setup_check.py
├── README.md
└── IMPLEMENTATION.md
```

---

## Phase 1 — Environment Setup

**Owner:** Member 1 | **Deadline:** Day 1

### Steps

**1. Clone / share the repository**
```bash
git clone <repo-url>
cd sepsis-ml-project
```

**2. Create a virtual environment**

*macOS / Linux*
```bash
python3 -m venv venv
source venv/bin/activate
```

*Windows (PowerShell)*
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify the environment**
```bash
python setup_check.py
```
All imports should print `OK`. Fix any that fail before proceeding.

**5. Create missing output folders**
```bash
mkdir -p data results/figures results/tables notebooks
```

### Required Libraries

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `scikit-learn` | Logistic Regression, scaling, evaluation |
| `xgboost` | Gradient boosted tree model |
| `shap` | SHAP explainability values |
| `matplotlib`, `seaborn` | Visualizations |
| `scipy` | Spearman rank correlation |
| `tqdm` | Progress bars for long loops |
| `torch` / `tensorflow` | Optional LSTM (skip if behind schedule) |

---

## Phase 2 — Data Collection

**Owner:** Members 2 (PhysioNet) & 3 (eICU) | **Deadline:** Day 1–2

### Dataset 1 — PhysioNet Sepsis Challenge 2019

1. Go to [physionet.org](https://physionet.org) and create a free account
2. Navigate to **PhysioNet Computing in Cardiology Challenge 2019**
3. Sign the **Data Use Agreement**
4. Download **Training Set A** (~20,000 patients, `.psv` files)
5. Place all files in a local folder — access is instant, no approval wait
6. Each file is one patient with hourly rows of vitals and a `SepsisLabel` column

### Dataset 2 — eICU Collaborative Research Database

1. Use the same PhysioNet account
2. Navigate to the **eICU-CRD** page and sign its separate Data Use Agreement
3. Download three tables:
   - `patient.csv`
   - `vitalPeriodic.csv`
   - `lab.csv`
4. Access is instant — no additional training required

> **Critical day-1 action:** Members 2 and 3 must agree on the exact column names they will both produce before writing any cleaning code. See the canonical list in [Phase 3](#phase-3--data-loading--cleaning).

---

## Phase 3 — Data Loading & Cleaning

**Owner:** Member 2 (PhysioNet), Member 3 (eICU), Member 4 (integration) | **Deadline:** Day 2–4

### Canonical Column Names (agree on these first)

Both datasets must produce a table with exactly these column names:

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Unique patient identifier |
| `heart_rate` | float | Heart rate (bpm) |
| `resp_rate` | float | Respiratory rate (breaths/min) |
| `temperature` | float | Body temperature — **Celsius only** |
| `sbp` | float | Systolic blood pressure (mmHg) |
| `dbp` | float | Diastolic blood pressure (mmHg) |
| `spo2` | float | Oxygen saturation (%) |
| `wbc` | float | White blood cell count (×10³/µL) |
| `lactate` | float | Lactate level (mmol/L) |
| `sepsis_label` | int | 1 if patient ever develops sepsis, else 0 |

> **eICU note:** Temperature in the raw eICU export is in Fahrenheit. The integration script auto-detects and converts. Do not manually convert — let the script handle it to avoid double-conversion.

### PhysioNet Cleaning (`scripts/physionet/clean_physionet.py`)

```
1. Load all .psv patient files from Training Set A
2. Add a `patient_id` column to each file (derived from filename)
3. Concatenate all files into one DataFrame
4. Rename columns to the canonical names above
5. Keep only the 10 canonical columns
6. Derive patient-level label: sepsis_label = 1 if SepsisLabel ever == 1
7. Forward fill within each patient group
8. Median fill any remaining NaNs (column-level median)
9. Save to scripts/physionet/physionet_cleaned.csv
10. Save patient-level labels to scripts/physionet/physionet_labels.csv
```

### eICU Cleaning

```
1. Load patient.csv, vitalPeriodic.csv, lab.csv
2. Join all three on patientunitstayid (the eICU patient key)
3. Rename all columns to match the canonical names exactly
4. Keep only the 10 canonical columns
5. Forward fill within each patient group
6. Median fill any remaining NaNs
7. Derive the sepsis label from diagnosis codes (apacheadmissiondx or ICD codes)
   — do NOT leave sepsis_label blank; the integration script cannot fill it accurately
8. Save to `data/raw/eicu_final_output.csv`
```

> **Important:** `data/raw/eicu_final_output.csv` currently contains labels, but the patient-level sepsis rate is unusually high (~68%). This should be validated against diagnosis-code logic before final cross-dataset conclusions.

### Integration (`scripts/integrate_datasets.py`) ✓ Complete

Run this once both cleaned files are ready:

```bash
python scripts/integrate_datasets.py
```

What the script does automatically:
- Validates schema on both datasets
- Detects and converts Fahrenheit → Celsius
- Derives a qSOFA proxy label if `sepsis_label` is missing (eICU fallback only)
- Prefixes patient IDs (`pn_p000001`, `eicu_p000001`) to prevent collisions
- Adds a `source` column (`physionet` or `eicu`)
- Concatenates and saves to `data/processed/integrated_dataset.csv`
- Prints a side-by-side validation summary

**Expected output:**

| Dataset | Rows | Patients | Sepsis Rate |
|---------|------|----------|-------------|
| PhysioNet | 789,484 | 20,317 | ~8.8% |
| eICU | 16,638 | 132 | currently ~68% (needs validation) |
| Integrated | 806,122 | 20,449 | ~9.2% |

---

## Phase 4 — Feature Engineering

**Owner:** Member 1 or 2 | **Deadline:** Day 4–5  
**Script:** `scripts/feature_engineering.py`  
**Input:** `data/processed/integrated_dataset.csv`  
**Output:** `data/features/physionet_features_*.csv`, `data/features/eicu_features_*.csv`

### Goal

Convert the time-series table (one row per hour per patient) into a flat table (one row per patient) suitable for ML models.

### Aggregate Features per Patient

For each of the 8 vital signs, compute these 5 statistics:

| Statistic | Formula | Captures |
|-----------|---------|---------|
| `mean` | Average over all hours | Typical condition |
| `max` | Maximum value | Worst recorded state |
| `min` | Minimum value | Best recorded state |
| `std` | Standard deviation | How much it fluctuated |
| `slope` | Mean of consecutive differences | Whether it's trending up or down |

This gives **8 vitals × 5 stats = 40 features** per patient, plus the label.

### Time-Window Features

Compute the same 5 statistics separately for three lookback windows:

| Window | Description |
|--------|------------|
| Last 6 hours | Closest to onset — highest signal |
| Last 12 hours | Medium window |
| Last 24 hours | Earliest practical prediction horizon |

This enables temporal analysis: compare model performance at 6h vs 24h before sepsis onset.

### Implementation Sketch

```python
def compute_patient_features(group, suffix=""):
    """
    group: DataFrame of hourly rows for one patient
    suffix: "_6h", "_12h", "_24h", or "" for full stay
    """
    features = {}
    for col in FEATURE_COLS:
        features[f"{col}_mean{suffix}"] = group[col].mean()
        features[f"{col}_max{suffix}"]  = group[col].max()
        features[f"{col}_min{suffix}"]  = group[col].min()
        features[f"{col}_std{suffix}"]  = group[col].std()
        features[f"{col}_slope{suffix}"] = group[col].diff().mean()
    return features
```

### Expected Output Shape

- One row per patient
- ~40 features for full-stay aggregation + ~40 per time window = ~160 features total
- Plus `patient_id`, `source`, `sepsis_label`

---

## Phase 5 — Model Training

**Owner:** Member 2 (PhysioNet), Member 3 (eICU) | **Deadline:** Day 5–7  
**Script:** `scripts/train_models.py`

**Current status:** ✅ Implemented and validated on PhysioNet (`full` window).
Artifacts saved to `results/models/` and `results/tables/training_summary.json`.

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          # preserve sepsis ratio in both splits
    random_state=42
)
```

### Model 1 — Logistic Regression

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(
    class_weight="balanced",   # handles ~10% sepsis imbalance
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)
```

### Model 2 — XGBoost

```python
import xgboost as xgb

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=neg / pos,   # handles class imbalance
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

> XGBoost does **not** need feature scaling. Use raw `X_train` / `X_test`.

### LSTM (Optional — decide by end of Week 2)

- Takes sequences as input rather than flat features — natural fit for time-series
- Implement only if the team is on schedule
- If it causes delays, skip entirely — the paper is complete with two models

---

## Phase 6 — Model Evaluation

**Owner:** Member 4 | **Deadline:** Day 7–8  
**Script:** `scripts/evaluate_models.py`

**Current status:** ✅ Implemented and validated.
Outputs: `results/tables/evaluation_metrics.csv` and `.json`.

### Metrics

| Metric | Why it matters |
|--------|---------------|
| **AUROC** | Primary metric — ranks sick vs healthy; unaffected by class imbalance |
| **AUPRC** | Better than AUROC for imbalanced data; measures precision/recall tradeoff |
| **F1 Score** | Balance of precision and recall at a fixed threshold |
| **Sensitivity** | % of true sepsis cases caught (clinical priority) |
| **Specificity** | % of healthy patients correctly cleared (reduces false alarms) |

### Implementation

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auroc = roc_auc_score(y_test, y_prob)
auprc = average_precision_score(y_test, y_prob)
f1    = f1_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Temporal Evaluation

Repeat evaluation at each time window (6h, 12h, 24h feature sets) to show how early the model can reliably predict. This becomes the temporal analysis section of the paper.

### Paper Table 1 Template

| Model | Dataset | AUROC | AUPRC | F1 | Sensitivity | Specificity |
|-------|---------|-------|-------|----|-------------|-------------|
| Logistic Regression | PhysioNet | | | | | |
| XGBoost | PhysioNet | | | | | |
| Logistic Regression | eICU | | | | | |
| XGBoost | eICU | | | | | |

---

## Phase 7 — SHAP Explanations

**Owner:** Member 2 or 3 | **Deadline:** Day 8–9  
**Script:** `scripts/shap_explanations.py`

**Current status:** ✅ Implemented for PhysioNet.
Outputs include SHAP importance tables (`results/tables/`) and SHAP bar plots (`results/figures/`).

### Steps

```python
import shap

# For XGBoost
explainer  = shap.TreeExplainer(xgb_model)
shap_vals  = explainer.shap_values(X_test)

# For Logistic Regression
explainer_lr = shap.LinearExplainer(lr, X_train_scaled)
shap_vals_lr = explainer_lr.shap_values(X_test_scaled)

# Global feature importance = mean absolute SHAP per feature
mean_abs_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0),
    index=X_test.columns
).sort_values(ascending=False)

top10 = mean_abs_shap.head(10)
```

### Output per model

- Ranked list of top 10 features by mean |SHAP|
- Summary bar plot (saved to `results/figures/shap_<model>_<dataset>.png`)
- Raw SHAP values saved for use in Phase 8

### Run on PhysioNet first, then repeat identically on eICU in Phase 9.

---

## Phase 8 — Stability Score

**Owner:** Member 4 | **Deadline:** Day 9–10  
**Script:** `scripts/stability_score.py`

**Current status:** ✅ Implemented.
Currently computes available comparisons and marks unavailable ones as `PENDING` until eICU SHAP artifacts are generated.

### Formula

Given two ranked lists of feature importances A and B:

**Sub-metric 1 — Top-K Feature Overlap**
```
overlap = |top_k(A) ∩ top_k(B)| / K
```
*If 7 of the top 10 features appear in both lists: overlap = 0.7*

**Sub-metric 2 — Spearman Rank Correlation**
```
rho = spearman_correlation(rank(A), rank(B))
```
*Measures whether the same features appear in the same order of importance.*

**Final Stability Score**
```
stability = (overlap + rho) / 2
```

### Implementation

```python
from scipy.stats import spearmanr

def stability_score(importance_a: pd.Series, importance_b: pd.Series, k: int = 10) -> dict:
    """
    importance_a / importance_b: Series indexed by feature name,
    values are mean absolute SHAP (already sorted descending).
    """
    top_k_a = set(importance_a.head(k).index)
    top_k_b = set(importance_b.head(k).index)

    overlap = len(top_k_a & top_k_b) / k

    # Align on the union of all features
    all_features = importance_a.index.union(importance_b.index)
    a_aligned    = importance_a.reindex(all_features, fill_value=0)
    b_aligned    = importance_b.reindex(all_features, fill_value=0)
    rho, _       = spearmanr(a_aligned, b_aligned)

    score = (overlap + rho) / 2
    return {"overlap": overlap, "spearman_rho": rho, "stability_score": score}
```

### Comparisons to Compute

| Comparison | Type |
|-----------|------|
| LR vs XGBoost on PhysioNet | Cross-model, same dataset |
| LR vs XGBoost on eICU | Cross-model, same dataset |
| XGBoost PhysioNet vs XGBoost eICU | Cross-dataset, same model |
| LR PhysioNet vs LR eICU | Cross-dataset, same model |

### Expected Output — Stability Score Matrix

| Comparison | Overlap | Spearman ρ | Stability Score |
|-----------|---------|------------|-----------------|
| LR vs XGB (PhysioNet) | | | |
| LR vs XGB (eICU) | | | |
| XGB: PhysioNet vs eICU | | | |
| LR: PhysioNet vs eICU | | | |

High scores (> 0.7) = explanations are consistent and trustworthy.  
Low scores (< 0.4) = explanations are unreliable despite high accuracy.

---

## Phase 9 — Repeat on eICU

**Owner:** Member 3 | **Deadline:** Day 10–11

Run Phases 4–8 again using only the eICU rows from `data/processed/integrated_dataset.csv`.

```python
eicu_df = integrated_df[integrated_df["source"] == "eicu"].copy()
```

Because column names are identical across both datasets, **every script from Phase 4 onward runs without modification**. The only change is the input DataFrame.

Pass the resulting eICU SHAP values into the `stability_score()` function alongside the PhysioNet SHAP values to compute the cross-dataset scores.

---

## Phase 10 — Visualizations

**Owner:** Member 1 or 4 | **Deadline:** Day 11–12  
**Output folder:** `results/figures/`

### Visual 1 — SHAP Bar Plots (4 total)

One bar chart per model per dataset showing top 10 features by mean |SHAP|.

```python
shap.summary_plot(shap_vals, X_test, plot_type="bar", max_display=10)
```

Files: `shap_lr_physionet.png`, `shap_xgb_physionet.png`, `shap_lr_eicu.png`, `shap_xgb_eicu.png`

### Visual 2 — Stability Heatmap (centerpiece of paper)

A color-coded grid of stability scores. Green = high stability, red = low.

```python
import seaborn as sns

sns.heatmap(
    stability_matrix,
    annot=True, fmt=".2f",
    cmap="RdYlGn", vmin=0, vmax=1,
    linewidths=0.5
)
```

File: `stability_heatmap.png`

### Visual 3 — Feature Rank Comparison Chart

Side-by-side bar chart showing how features are ranked differently across models, making agreement or disagreement visually obvious.

### Visual 4 — Performance Table

Clean table of AUROC and F1 for each model on each dataset — goes directly into the paper as Table 1.

---

## Team Division & Timeline

| Member | Owns | Key Deliverable |
|--------|------|----------------|
| **Member 1** — Environment Lead | Phase 1 | Working environment for all teammates by Day 1 |
| **Member 2** — PhysioNet Owner | Phase 2 + 3 (PhysioNet) + 5 + 7 | `physionet_cleaned.csv`, trained LR & XGB, SHAP values |
| **Member 3** — eICU Owner | Phase 2 + 3 (eICU) + 9 | `eicu_final_output.csv` with real labels, eICU SHAP values |
| **Member 4** — Integration Lead | Phase 3 (integration) + 6 + 8 + 10 | `integrated_dataset.csv`, evaluation table, Stability Score |

### Day-by-Day Schedule

| Day | Milestone |
|-----|-----------|
| 1 | Member 1 shares folder structure. Members 2 & 3 agree on column names. Data download begins. |
| 2 | PhysioNet and eICU data downloaded. Cleaning scripts started. |
| 3 | Members 2 & 3 share cleaned DataFrames (even if not polished). |
| 4 | Member 4 runs integration script. Validation summary reviewed by team. Phase 3 complete. |
| 5 | Feature engineering complete. Flat per-patient feature tables ready. |
| 6–7 | Models trained on PhysioNet. Evaluation metrics recorded. |
| 8–9 | SHAP values extracted. Stability scores computed. |
| 10–11 | eICU pipeline complete. Cross-dataset stability scores computed. |
| 12 | All visualizations finalized. Paper draft circulated. |

---

## Key Rules

1. **Never use accuracy as your primary metric.** The dataset is ~10% sepsis (imbalanced). Always lead with AUROC.

2. **Column names must match exactly across both datasets.** A single mismatch breaks the cross-dataset Stability Score comparison. Use `scripts/utils/constants.py` as the single source of truth.

3. **Never copy-paste preprocessing logic.** Both datasets must use the same `forward_fill` and `median_fill` functions from `scripts/utils/preprocessing_utils.py`.

4. **The Stability Score is the novelty.** Accuracy improvements are secondary. The paper's argument is that high accuracy alone is insufficient — explanations must also be stable.

5. **LSTM is optional.** If Week 2 is on track and a team member is comfortable with PyTorch or Keras, add it. Otherwise skip it — the paper is complete with two models.

6. **Prefix patient IDs in the integrated dataset.** `pn_p000001` and `eicu_p000001` are different patients. The integration script handles this automatically.

7. **eICU sepsis labels must come from the diagnosis table** — not from clinical heuristics. The current qSOFA proxy in the integration script is a fallback only.
