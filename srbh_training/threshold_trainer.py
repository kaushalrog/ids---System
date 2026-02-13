import pandas as pd
import numpy as np
import json

INPUT = "srbh_processed.csv"
OUTPUT = "trained_thresholds.json"

print("Loading processed SRBH dataset...")
df = pd.read_csv(INPUT)

print("\nAvailable columns:")
print(list(df.columns))

# --- Create combo_attack feature ---
df["combo_attack"] = (
    (df["has_sql_kw"] == 1).astype(int) +
    (df["has_cmd_kw"] == 1).astype(int) +
    (df["has_traversal"] == 1).astype(int)
)

features = [
    "has_sql_kw",
    "has_cmd_kw",
    "has_traversal",
    "req_len",
    "body_len",
    "cookie_len",
    "ua_len",
    "combo_attack"
]

print("\nUsing features:")
for f in features:
    print("  -", f)

# --- Compute drift proxy score ---
df["drift_score"] = (
    2.5 * df["has_sql_kw"] +
    2.2 * df["has_cmd_kw"] +
    2.0 * df["has_traversal"] +
    0.006 * df["req_len"] +
    0.008 * df["body_len"] +
    0.003 * df["cookie_len"] +
    0.002 * df["ua_len"] +
    4.0 * df["combo_attack"]
)

# --- Normalize ---
mean = df["drift_score"].mean()
std = df["drift_score"].std() + 1e-6
df["drift_z"] = (df["drift_score"] - mean) / std

# --- Balanced Threshold Search ---
best_threshold = None
best_score = -999
best_tpr = 0
best_fpr = 1

thresholds = np.linspace(
    np.percentile(df["drift_z"], 5),
    np.percentile(df["drift_z"], 95),
    300
)

for t in thresholds:
    preds = (df["drift_z"] >= t).astype(int)

    TP = ((preds == 1) & (df["Target"] == 1)).sum()
    FP = ((preds == 1) & (df["Target"] == 0)).sum()
    FN = ((preds == 0) & (df["Target"] == 1)).sum()
    TN = ((preds == 0) & (df["Target"] == 0)).sum()

    TPR = TP / (TP + FN + 1e-6)
    FPR = FP / (FP + TN + 1e-6)

    # Balanced objective: prioritize TPR but punish FPR
    score = (2.8 * TPR) - (1.2 * FPR)

    if TPR >= 0.95 and FPR <= 0.35 and score > best_score:
        best_score = score
        best_threshold = t
        best_tpr = TPR
        best_fpr = FPR

# --- Fallback: strict percentile ---
if best_threshold is None:
    print("\n⚠ No clean threshold found — using percentile fallback")
    best_threshold = np.percentile(df["drift_z"], 85)

    preds = (df["drift_z"] >= best_threshold).astype(int)
    TP = ((preds == 1) & (df["Target"] == 1)).sum()
    FP = ((preds == 1) & (df["Target"] == 0)).sum()
    FN = ((preds == 0) & (df["Target"] == 1)).sum()
    TN = ((preds == 0) & (df["Target"] == 0)).sum()

    best_tpr = TP / (TP + FN + 1e-6)
    best_fpr = FP / (FP + TN + 1e-6)

print("\n==============================")
print("BALANCED IDS MODE (SRBH-20)")
print("==============================")
print(f"Threshold: {best_threshold:.4f}")
print(f"TPR (Detection Rate): {best_tpr:.4f}")
print(f"FPR (False Positive Rate): {best_fpr:.4f}")

trained = {
    "warning": float(best_threshold * 0.8),
    "alert": float(best_threshold),
    "detection_rate": float(best_tpr),
    "false_positive_rate": float(best_fpr),
    "used_features": features,
    "mode": "balanced_srbh20"
}

with open(OUTPUT, "w") as f:
    json.dump(trained, f, indent=2)

print("\nSaved trained thresholds to:", OUTPUT)
