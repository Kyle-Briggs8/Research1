# Bayesian Network Structure Learning for Healthcare Outcome Prediction

Research project (Semester 1) exploring Bayesian Network (BN) structure learning, Markov Blanket discovery, and feature importance analysis applied to hospital discharge data to understand predictors of clinical and billing outcomes.

---

## Research Overview

The core question: **which patient and admission variables form the minimal predictive neighborhood (Markov Blanket) around LENGTH_OF_STAY and TOTAL_CHARGES in Texas hospital discharge records?**

The project combines three complementary approaches:
- Greedy and exact BN structure learning (Hill Climbing, ILP/BIC)
- Bootstrap Markov Blanket stability analysis
- Multi-method feature importance (RF Gini, Permutation Importance, Mutual Information)

A parallel track on the UCI Adult Income dataset examines decision tree accuracy and feature importance disaggregated by race.

---

## Datasets

### Texas PUDF Q1 2019
Texas Department of State Health Services Public Use Data File — 706,472 inpatient hospital discharge records.

| Variable | Description |
|---|---|
| `PAT_AGE` | Patient age |
| `SEX_CODE` | Patient sex |
| `RACE` | Race category |
| `ETHNICITY` | Ethnicity category |
| `PAT_ZIP` | Patient ZIP code |
| `SOURCE_OF_ADMISSION` | How the patient was admitted |
| `EMERGENCY_DEPT_FLAG` | Whether admitted through emergency |
| `PRINC_DIAG_CODE` | Principal ICD-10 diagnosis code |
| `CARE_TYPE` | Type of care received |
| `FIRST_PAYMENT_SRC` | Primary payer (Medicare, Medicaid, etc.) |
| `SECONDARY_PAYMENT_SRC` | Secondary payer |
| `PAT_STATUS` | Discharge status |
| `LENGTH_OF_STAY` | Days in hospital (target) |
| `TOTAL_CHARGES` | Total billed charges (target) |

> Raw data files are excluded from this repo due to size (~400 MB). The cleaned patient vault (`Cleaned_Patient_Vault.csv`, ~84 MB) is also excluded.

### UCI Adult Income
Standard benchmark dataset for income classification (>$50K/year). Used to analyze decision tree accuracy and feature importance, stratified by race.

---

## Notebooks

### [`dt.ipynb`](dt.ipynb) — Decision Trees on Adult Income Dataset
- Loads UCI Adult Income dataset
- Filters to white subpopulation and full population
- Binarizes income target (>50K vs. ≤50K)
- Evaluates decision tree accuracy and AUC across depths
- Computes and aggregates OHE feature importances back to original features

### [`texas.ipynb`](texas.ipynb) — Initial Data Processing & BN Learning
- Merges PUDF base files (base1 + base2 + charges)
- Cleans and discretizes variables
- Initial Bayesian Network structure learning via Hill Climbing (BIC score)
- Visualizes learned DAG

### [`texas2.ipynb`](texas2.ipynb) — Markov Blanket Stability & Feature Importance
- Bootstrap Markov Blanket stability (100 runs, 100k samples each)
- Multi-method feature importance for both targets:
  - RF Gini importance
  - Permutation importance (test set)
  - Mutual information
- Combined ranking table
- Diagnostic: tests whether PAT_ZIP and PRINC_DIAG_CODE add signal beyond the Markov Blanket

### [`groupedTexas.ipynb`](groupedTexas.ipynb) — Grouped Variables & BDeu Scoring
- Reduces cardinality via ICD-10 chapter grouping (11,346 → 20 clinical categories)
- Reduces ZIP cardinality via 3-digit prefix grouping (1,740 → 72 regions)
- Re-runs bootstrap BN structure learning with grouped features
- Compares Markov Blanket stability and DAG topology under both scoring criteria (BIC and BDeu)

### [`texasIPL.ipynb`](texasIPL.ipynb) — ILP-Based Exact DAG Structure Learning
- Replaces greedy Hill Climbing with exact ILP formulation
- Uses `scipy.optimize.milp` (HiGHS solver) — no external solver packages
- Pre-computes BIC score for every candidate parent set (up to `max_parents`)
- Finds the globally optimal BIC-scoring DAG

### [`IAMB.ipynb`](IAMB.ipynb) — IAMB Algorithm
- Incremental Association Markov Blanket (IAMB) algorithm implementation

---

## Scripts

### [`feature_importance_texas.py`](feature_importance_texas.py)
Standalone script for RF-based feature importance:
- Targets: `LENGTH_OF_STAY` and `TOTAL_CHARGES`
- Handles mixed numeric/categorical features with a preprocessing pipeline
- Truncates `PRINC_DIAG_CODE` to 3-character ICD category level to reduce cardinality
- Aggregates one-hot-encoded importances back to original feature names
- Saves bar chart to `feature_importance_texas.png`

---

## Key Findings

### Feature Importance (Texas PUDF)

**LENGTH_OF_STAY predictors** (avg rank across RF Gini, Permutation, Mutual Info):
1. `PRINC_DIAG_CODE` — dominant across all methods
2. `TOTAL_CHARGES` — high permutation importance (correlated outcome)
3. `PAT_STATUS` / `PAT_AGE` — consistent mid-tier importance
4. `RACE`, `ETHNICITY` — rank near-last in all methods

**TOTAL_CHARGES predictors:**
1. `PRINC_DIAG_CODE` — dominant
2. `LENGTH_OF_STAY` — highly correlated
3. `PAT_AGE`, `PAT_ZIP` — significant
4. `RACE`, `ETHNICITY` — again rank last

### Markov Blanket (Bayesian Network)
The bootstrap-stable Markov Blanket for `LENGTH_OF_STAY` includes:
`TOTAL_CHARGES`, `PRINC_DIAG_CODE`, `PAT_STATUS`, `PAT_AGE`, `SOURCE_OF_ADMISSION`

### Cardinality Reduction
Grouping `PRINC_DIAG_CODE` to ICD-10 chapters (11,346 → 20 categories) and ZIP to 3-digit prefixes (1,740 → 72) substantially improves BN stability without sacrificing predictive structure.

---

## Generated Figures

| File | Description |
|---|---|
| `feature_importance.png` | RF feature importance bar chart (original) |
| `markov_blanket_graphs.png` | Markov Blanket subgraphs (BIC, raw features) |
| `markov_blanket_stability.png` | Edge stability heatmap across bootstrap runs |
| `full_dag_grouped.png` | Full learned DAG with grouped variables |
| `markov_blanket_graphs_grouped.png` | Markov Blanket subgraphs (grouped features) |
| `markov_blanket_stability_grouped.png` | Stability heatmap (grouped) |
| `full_dag_bdeu.png` | Full DAG learned with BDeu scoring |
| `markov_blanket_graphs_bdeu.png` | Markov Blanket subgraphs (BDeu) |
| `markov_blanket_stability_bdeu.png` | Stability heatmap (BDeu) |
| `blanket_diagnostic.png` | Diagnostic: blanket vs. blanket+ZIP+DIAG accuracy |
| `figs/` | Distribution plots for individual features |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install pgmpy scikit-learn pandas numpy matplotlib seaborn networkx scipy pyro-ppl torch transformers
```

Key packages:
- `pgmpy` — Bayesian Network structure learning and inference
- `scikit-learn` — Random Forest, preprocessing, evaluation
- `scipy.optimize.milp` — ILP-based exact structure learning
- `networkx` — graph construction and visualization
- `ucimlrepo` — UCI dataset loading
