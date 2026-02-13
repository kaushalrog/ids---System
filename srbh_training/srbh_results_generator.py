#!/usr/bin/env python3
"""
SRBH-20 IDS Result Generator
Produces tables and graphs
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = "../srbh_training/srbh_processed.csv"
THRESHOLDS = "../srbh_training/trained_thresholds.json"
OUT_CSV = "srbh_results.csv"
OUT_PLOT = "srbh_results_plot.png"

print("=" * 70)
print("SRBH-20 IDS RESULT GENERATOR")
print("=" * 70)

# ---------- Load Data ----------

df = pd.read_csv(INPUT)

with open(THRESHOLDS) as f:
    thr = json.load(f)

warning_thr = thr["warning"]
alert_thr = thr["alert"]

print(f"\nUsing thresholds:")
print(f"  WARNING: {warning_thr}")
print(f"  ALERT:   {alert_thr}")

# ---------- Feature Engineering ----------

df["combo_attack"] = (
    df["has_sql_kw"] +
    df["has_cmd_kw"] +
    df["has_traversal"]
)

df["req_len_n"] = np.log1p(df["req_len"])
df["body_len_n"] = np.log1p(df["body_len"])
df["cookie_len_n"] = np.log1p(df["cookie_len"])
df["ua_len_n"] = np.log1p(df["ua_len"])

df["weak_signal"] = (df["combo_attack"] == 1).astype(int)

df["drift_score"] = (
    2.8 * df["has_sql_kw"] +
    2.4 * df["has_cmd_kw"] +
    2.2 * df["has_traversal"] +
    0.25 * df["req_len_n"] +
    0.35 * df["body_len_n"] +
    0.15 * df["cookie_len_n"] +
    0.10 * df["ua_len_n"] +
    1.8 * (df["combo_attack"] >= 2).astype(int) -
    1.2 * df["weak_signal"]
)

# ---------- Predictions ----------

df["Predicted"] = 0
df.loc[df["drift_score"] >= warning_thr, "Predicted"] = 1
df.loc[df["drift_score"] >= alert_thr, "Predicted"] = 2

df["Pred_Label"] = df["Predicted"].map({
    0: "NORMAL",
    1: "WARNING",
    2: "ALERT"
})

df["True_Label"] = df["Target"].map({
    0: "NORMAL",
    1: "ATTACK"
})

df.to_csv(OUT_CSV, index=False)

# ---------- Confusion Matrix ----------

tp = np.sum((df["Predicted"] >= 1) & (df["Target"] == 1))
fn = np.sum((df["Predicted"] == 0) & (df["Target"] == 1))
fp = np.sum((df["Predicted"] >= 1) & (df["Target"] == 0))
tn = np.sum((df["Predicted"] == 0) & (df["Target"] == 0))

accuracy = (tp + tn) / len(df)
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TN: {tn}")

print("\n==============================")
print("PERFORMANCE METRICS")
print("==============================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# ---------- Plots ----------

plt.figure(figsize=(12, 5))

# Drift Distribution
plt.subplot(1, 2, 1)
sns.histplot(df["drift_score"], bins=50, color="blue")
plt.axvline(warning_thr, color="orange", linestyle="--", label="WARNING")
plt.axvline(alert_thr, color="red", linestyle="--", label="ALERT")
plt.title("Drift Score Distribution")
plt.legend()

# Prediction Counts
plt.subplot(1, 2, 2)
sns.countplot(x="Pred_Label", data=df, palette="viridis")
plt.title("Prediction Breakdown")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300)

print("\nSaved:")
print("  CSV:", OUT_CSV)
print("  Plot:", OUT_PLOT)
print("\nDone.")
