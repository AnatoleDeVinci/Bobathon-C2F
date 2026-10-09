# %% [markdown]
# # Experiment: 18_leak_hunt — data-leakage investigation
#
# Three checks:
#   1. Absolute correlation of every raw column (+ derived IDs/flags) vs target.
#   2. Patient-ID overlap between train and test; temporal ordering of overlapping visits.
#   3. Determinism check: is (patient_id, rater_id, disease_duration) → target exact?

# %%
import numpy as np
import pandas as pd

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")          # Index kept as column
y_train = pd.read_csv(DATA_DIR / "y_train.csv")           # Index, target
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

# Merge train features + target on Index
train = X_train.merge(y_train, on="Index")

SEP = "=" * 65

# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — absolute Pearson correlation of every column vs target
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("CHECK 1 — |correlation| with target (top 15)")
print(SEP)

# Encode categorical columns as integer codes
analysis = train.copy()
for col in analysis.select_dtypes(include="object").columns:
    analysis[col] = pd.Categorical(analysis[col]).codes.astype(float)

# Add derived flag columns
analysis["off_missing"]  = train["off"].isna().astype(float)
analysis["on_missing"]   = train["on"].isna().astype(float)
analysis["ledd_missing"] = train["ledd"].isna().astype(float)
analysis["disease_duration"] = train["age"] - train["age_at_diagnosis"]

corrs = (
    analysis.drop(columns=["target"])
    .corrwith(analysis["target"])
    .abs()
    .dropna()
    .sort_values(ascending=False)
)

print(f"\n  {'Column':<30}  {'|corr|':>8}")
print("  " + "-" * 42)
for col, val in corrs.head(15).items():
    print(f"  {col:<30}  {val:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — patient_id overlap between train and test
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("CHECK 2 — patient_id overlap (train ∩ test)")
print(SEP)

train_pids   = set(train["patient_id"].unique())
test_pids    = set(X_test["patient_id"].unique())
overlap_pids = train_pids & test_pids

print(f"\n  Train patients : {len(train_pids)}")
print(f"  Test  patients : {len(test_pids)}")
print(f"  Overlap        : {len(overlap_pids)}")

if overlap_pids:
    print(f"\n  For overlapping patients — are test visits OLDER than train visits?")
    temporal_issues = []
    for pid in sorted(overlap_pids)[:20]:   # inspect up to 20
        tr_ages = train.loc[train["patient_id"] == pid, "age"].values
        te_ages = X_test.loc[X_test["patient_id"] == pid, "age"].values
        # "Leak" scenario: test visit ages fall within / before train visit ages
        te_min, te_max = te_ages.min(), te_ages.max()
        tr_min, tr_max = tr_ages.min(), tr_ages.max()
        interleaved = te_min < tr_max   # test visits are not all after all train visits
        temporal_issues.append({
            "patient_id": pid,
            "n_train": len(tr_ages),
            "n_test":  len(te_ages),
            "train_age_range": f"{tr_min:.1f}–{tr_max:.1f}",
            "test_age_range":  f"{te_min:.1f}–{te_max:.1f}",
            "test_before_train_end": interleaved,
        })
    df_temp = pd.DataFrame(temporal_issues)
    n_interleaved = df_temp["test_before_train_end"].sum()
    print(f"  Patients shown (up to 20): {len(df_temp)}")
    print(f"  Of those, test visits overlap / precede train range: {n_interleaved}")
    print()
    print(df_temp.to_string(index=False))
else:
    print("\n  No overlap — no patient-level temporal leakage possible.")

# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — "magic feature" determinism: (patient_id, rater_id, disease_duration) → target
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("CHECK 3 — determinism: (patient_id, rater_id, disease_duration) → target")
print(SEP)

key = ["patient_id", "rater_id", "disease_duration"]
train["disease_duration"] = train["age"] - train["age_at_diagnosis"]

# Count rows per key tuple
counts = train.groupby(key)["target"].agg(["count", "nunique", "std"]).reset_index()
counts.columns = key + ["n", "n_unique_targets", "target_std"]

total_keys   = len(counts)
exact_keys   = (counts["n_unique_targets"] == 1).sum()
multi_keys   = (counts["n_unique_targets"] > 1).sum()
max_std      = counts["target_std"].max()
mean_std_multi = counts.loc[counts["n_unique_targets"] > 1, "target_std"].mean()

print(f"\n  Unique (patient_id, rater_id, disease_duration) tuples: {total_keys}")
print(f"  Tuples with EXACTLY one unique target:                  {exact_keys}  ({100*exact_keys/total_keys:.1f}%)")
print(f"  Tuples with multiple distinct targets:                  {multi_keys}  ({100*multi_keys/total_keys:.1f}%)")
print(f"  Max within-tuple target std (multi-target tuples):      {max_std:.4f}")
print(f"  Mean within-tuple target std (multi-target tuples):     {mean_std_multi:.4f}")

# Also check simpler keys
for combo in [
    ["patient_id"],
    ["rater_id"],
    ["patient_id", "rater_id"],
    ["patient_id", "age"],
    ["Index"],
]:
    if all(c in train.columns for c in combo):
        g = train.groupby(combo)["target"].nunique()
        perfect = (g == 1).all()
        print(f"  Key {combo}: all deterministic = {perfect}  "
              f"(max unique targets per key = {g.max()})")

print()
print(SEP)
print("SUMMARY")
print(SEP)
top1_col, top1_corr = corrs.index[0], corrs.iloc[0]
print(f"  Highest single-column |corr| with target: {top1_col}  ({top1_corr:.4f})")
print(f"  Patient overlap train/test: {len(overlap_pids)}")
if total_keys > 0:
    print(f"  (patient_id, rater_id, disease_duration) exact determinism: "
          f"{100*exact_keys/total_keys:.1f}%  — {'SUSPICIOUS' if exact_keys/total_keys > 0.95 else 'normal'}")
print()
