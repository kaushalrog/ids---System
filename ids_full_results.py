#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    precision_recall_curve
)

INPUT = "drift_log.csv"

def main():
    print("=" * 70)
    print("RESEARCH-GRADE IDS RESULT ANALYZER (ALL-IN-ONE)")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    # -------------------------------------------------
    # 1. Ground Truth Assumption (Adjust if needed)
    # -------------------------------------------------
    df["Actual"] = df["endpoint"].apply(
        lambda x: 0 if x in ["/health"] else 1
    )
    df["Predicted"] = df["alert_level"].apply(
        lambda x: 1 if x in ["WARNING", "ALERT"] else 0
    )

    # -------------------------------------------------
    # 2. METRICS
    # -------------------------------------------------
    cm = confusion_matrix(df["Actual"], df["Predicted"])
    acc = accuracy_score(df["Actual"], df["Predicted"])
    prec = precision_score(df["Actual"], df["Predicted"], zero_division=0)
    rec = recall_score(df["Actual"], df["Predicted"])
    f1 = f1_score(df["Actual"], df["Predicted"])

    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print("\n======================")
    print("CONFUSION MATRIX")
    print("======================")
    print(pd.DataFrame(cm, index=["Actual Normal", "Actual Attack"],
                        columns=["Pred Normal", "Pred Attack"]))

    print("\n======================")
    print("PERFORMANCE METRICS")
    print("======================")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall (TPR)", "F1-Score", "FPR"],
        "Value": [acc, prec, rec, f1, fpr]
    })
    print(metrics_df)

    # -------------------------------------------------
    # 3. ROC CURVE
    # -------------------------------------------------
    fpr_vals, tpr_vals, _ = roc_curve(df["Actual"], df["drift_score"])
    roc_auc = auc(fpr_vals, tpr_vals)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_vals, tpr_vals, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("IDS ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 4. DRIFT SCORE TIMELINE
    # -------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(df["drift_score"], label="Drift Score")
    plt.axhline(1.5, linestyle="--", color="orange", label="Warning Threshold")
    plt.axhline(2.5, linestyle="--", color="red", label="Alert Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Drift Score")
    plt.title("Drift Score Timeline")
    plt.legend()
    plt.savefig("drift_timeline.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 5. ALERT TIMELINE
    # -------------------------------------------------
    plt.figure(figsize=(12, 4))
    colors = df["alert_level"].map(
        {"NORMAL": "green", "WARNING": "orange", "ALERT": "red"}
    )
    plt.scatter(df.index, df["drift_score"], c=colors)
    plt.xlabel("Sample Index")
    plt.ylabel("Drift Score")
    plt.title("Alert Level Timeline")
    plt.savefig("alert_timeline.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 6. DRIFT DISTRIBUTION
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["drift_score"], bins=30, kde=True)
    plt.xlabel("Drift Score")
    plt.title("Drift Score Distribution")
    plt.savefig("drift_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 7. CONFUSION MATRIX HEATMAP
    # -------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 8. PRECISION–RECALL CURVE
    # -------------------------------------------------
    precision_vals, recall_vals, _ = precision_recall_curve(
        df["Actual"], df["drift_score"]
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.savefig("precision_recall.png", dpi=300)
    plt.close()

    # -------------------------------------------------
    # 9. SAVE METRICS TABLE
    # -------------------------------------------------
    metrics_df.to_csv("ids_metrics_summary.csv", index=False)

    print("\n======================")
    print("FILES GENERATED")
    print("======================")
    print("roc_curve.png")
    print("drift_timeline.png")
    print("alert_timeline.png")
    print("drift_distribution.png")
    print("confusion_matrix.png")
    print("precision_recall.png")
    print("ids_metrics_summary.csv")

    print("\nANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()
