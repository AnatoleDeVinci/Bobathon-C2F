"""Sanity check for add_patient_features — read-only probe."""
from __future__ import annotations

import pandas as pd
from parkinson import PROJECT_ROOT
from parkinson.features import add_patient_features

DATA_DIR = PROJECT_ROOT / "data"

X = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
X_feat = add_patient_features(X)

print("Shape before:", X.shape)
print("Shape after: ", X_feat.shape)
new_cols = [c for c in X_feat.columns if c not in X.columns]
print(f"\n{len(new_cols)} new columns added:")
for c in new_cols:
    print(f"  {c:30s}  null={X_feat[c].isna().mean():.2%}")

# Spot-check: n_visits should match known ~8 per patient
print(f"\nn_visits min={X_feat['n_visits'].min()}, max={X_feat['n_visits'].max()}, mean={X_feat['n_visits'].mean():.1f}")
# disease_duration: no NaN (age and age_at_diagnosis both have low missingness)
print(f"disease_duration null: {X_feat['disease_duration'].isna().mean():.2%}")
# visit_number: 1-based
print(f"visit_number min={X_feat['visit_number'].min()}, max={X_feat['visit_number'].max()}")
