import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

INPUT = "../srbh_training/srbh_processed.csv"
THRESHOLD_FILE = "../srbh_training/trained_thresholds.json"

def main():
    print("=" * 70)
    print("SRBH-20 IDS RESULT GENERATOR")
    print("=" * 70)

    # Load threshold safely
    with open(THRESHOLD_FILE) as f:
        thresholds = json.load(f)

    threshold = (
        thresholds.get("threshold") or
        thresholds.get("best_threshold") or
        thresholds.get("optimal_threshold")
    )

    if threshold is None:
        raise ValueError("No valid threshold key found in trained_thresholds.json")

    print(f"Using threshold: {threshold:.4f}")

    # Load dataset
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} samples")
    print(f"Attack: {df['Target'].sum()} | Normal: {(df['Target'] == 0).sum()}")

    # Compute drift score
    df["drift_score"] = (
        2.5 * df["has_sql_kw"] +
        2.0 * df["has_cmd_kw"] +
        1.8 * df["has_traversal"] +
        0.02 * df["req_len"] +
        0.015 * df["body_len"] +
        0.01 * df["cookie_len"] +
        0.01 * df["ua_len"]
    )

    # Prediction
    df["Predicted"] = (df["drift_score"] >= threshold).astype(int)

    # Confusion matrix
    TP = int(((df["Predicted"] == 1) & (df["Target"] == 1)).sum())
    TN = int(((df["Predicted"] == 0) & (df["Target"] == 0)).sum())
    FP = int(((df["Predicted"] == 1) & (df["Target"] == 0)).sum())
    FN = int(((df["Predicted"] == 0) & (df["Target"] == 1)).sum())

    accuracy = (TP + TN) / len(df)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\nConfusion Matrix:")
    print(f"TP: {TP}  FP: {FP}")
    print(f"FN: {FN}  TN: {TN}")

    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Save tables
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Value": [accuracy, precision, recall, f1]
    }).to_csv("metrics_table.csv", index=False)

    pd.DataFrame({
        "TP": [TP], "FP": [FP], "FN": [FN], "TN": [TN]
    }).to_csv("confusion_matrix.csv", index=False)

    df[["drift_score", "Target", "Predicted"]].to_csv("scored_dataset.csv", index=False)

    # ROC Curve
    fpr, tpr, _ = roc_curve(df["Target"], df["drift_score"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — SRBH-20 IDS")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # Drift score distribution
    plt.figure()
    df[df["Target"] == 0]["drift_score"].hist(alpha=0.5, bins=50, label="Normal")
    df[df["Target"] == 1]["drift_score"].hist(alpha=0.5, bins=50, label="Attack")
    plt.xlabel("Drift Score")
    plt.ylabel("Frequency")
    plt.title("Drift Score Distribution")
    plt.legend()
    plt.savefig("drift_distribution.png")
    plt.close()

    # Attack vs Normal bar plot
    counts = df.groupby(["Target", "Predicted"]).size().unstack().fillna(0)
    counts.plot(kind="bar", stacked=True)
    plt.title("Attack vs Normal Predictions")
    plt.xlabel("True Label (0=Normal, 1=Attack)")
    plt.ylabel("Count")
    plt.savefig("attack_vs_normal.png")
    plt.close()

    print("\nSaved:")
    print(" - metrics_table.csv")
    print(" - confusion_matrix.csv")
    print(" - scored_dataset.csv")
    print(" - roc_curve.png")
    print(" - drift_distribution.png")
    print(" - attack_vs_normal.png")

    print("\nDONE — SRBH-20 results generated successfully.")

if __name__ == "__main__":
    main()
