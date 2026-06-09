import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ── 1. Load cleaned data ──────────────────────────────────────────────────────
df = pd.read_csv("Cleaned_Patient_Vault.csv")
print(f"Loaded {len(df):,} rows")
print(df.dtypes)

TARGETS = ["LENGTH_OF_STAY", "TOTAL_CHARGES"]
# PAT_ZIP has ~thousands of unique zip codes — too high cardinality for OHE
DROP_COLS = ["RECORD_ID", "PAT_ZIP"] + TARGETS

# ── 2. Build feature matrix ───────────────────────────────────────────────────
X = df.drop(columns=DROP_COLS, errors="ignore")

# Truncate PRINC_DIAG_CODE to first 3 chars (ICD category level) to reduce
# cardinality from thousands of codes to ~hundreds
if "PRINC_DIAG_CODE" in X.columns:
    X["PRINC_DIAG_CODE"] = X["PRINC_DIAG_CODE"].astype(str).str[:3]

# Ensure PAT_AGE is numeric
if "PAT_AGE" in X.columns:
    X["PAT_AGE"] = pd.to_numeric(X["PAT_AGE"], errors="coerce")

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# Strip whitespace in categoricals
for c in cat_cols:
    X[c] = X[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

preproc = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ],
    remainder="drop",
)

# ── 3. Feature importance helper ─────────────────────────────────────────────
def run_importance(target_col: str, n_estimators: int = 200, random_state: int = 0):
    y = df[target_col].copy()
    mask = y.notna()
    X_clean = X[mask].copy()
    y_clean = y[mask]

    pipe = Pipeline([
        ("prep", preproc),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    pipe.fit(X_clean, y_clean)

    rf = pipe.named_steps["rf"]
    prep = pipe.named_steps["prep"]

    raw_importances = rf.feature_importances_

    # Aggregate OHE columns back to original feature name
    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    ohe_dims = [len(cats) for cats in ohe.categories_]

    feat_importance = {}
    idx = 0
    for f in num_cols:
        feat_importance[f] = float(raw_importances[idx])
        idx += 1
    for f, width in zip(cat_cols, ohe_dims):
        feat_importance[f] = float(np.sum(raw_importances[idx:idx + width]))
        idx += width

    result = (pd.DataFrame(
        [{"feature": k, "importance": v} for k, v in feat_importance.items()]
    ).sort_values("importance", ascending=False).reset_index(drop=True))

    return result

# ── 4. Run for both targets ───────────────────────────────────────────────────
print("\nRunning feature importance for LENGTH_OF_STAY ...")
imp_los = run_importance("LENGTH_OF_STAY")
print(imp_los.to_string(index=False))

print("\nRunning feature importance for TOTAL_CHARGES ...")
imp_tc = run_importance("TOTAL_CHARGES")
print(imp_tc.to_string(index=False))

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, imp_df, title in zip(
    axes,
    [imp_los, imp_tc],
    ["Feature Importance — Length of Stay", "Feature Importance — Total Charges"],
):
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_xlabel("Importance (Mean Decrease Impurity)")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig("feature_importance_texas.png", dpi=150)
print("\nPlot saved as feature_importance_texas.png")
plt.show()
