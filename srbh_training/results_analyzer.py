import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

INPUT = "drift_log.csv"
WARNING_T = 1.5
ALERT_T = 2.5

print("="*70)
print("RESEARCH-GRADE IDS RESULT ANALYZER")
print("="*70)

df = pd.read_csv(INPUT)

# Fix column names just in case
df.columns = [c.strip() for c in df.columns]

# Create ground truth from endpoint ordering
df["Actual"] = 0
df.loc[df.index >= 10, "Actual"] = 1
df.loc[df.index >= 25, "Actual"] = 0

df["Predicted"] = df["alert_level"].apply(lambda x: 1 if x in ["WARNING", "ALERT"] else 0)

TP = ((df["Actual"] == 1) & (df["Predicted"] == 1)).sum()
TN = ((df["Actual"] == 0) & (df["Predicted"] == 0)).sum()
FP = ((df["Actual"] == 0) & (df["Predicted"] == 1)).sum()
FN = ((df["Actual"] == 1) & (df["Predicted"] == 0)).sum()

accuracy = (TP + TN) / len(df)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
fpr = FP / (FP + TN + 1e-9)

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "TPR", "FPR"],
    "Value": [accuracy, precision, recall, f1, recall, fpr]
})

summary.to_csv("table_summary.csv", index=False)

cm = pd.DataFrame(
    [[TN, FP],[FN, TP]],
    columns=["Pred Normal","Pred Attack"],
    index=["Actual Normal","Actual Attack"]
)
cm.to_csv("table_confusion_matrix.csv")

stats = df.groupby("Actual")["drift_score"].agg(["mean","std","min","max"])
stats.to_csv("table_drift_stats.csv")

plt.figure(figsize=(10,5))
plt.plot(df["drift_score"], label="Drift Score")
plt.axhline(WARNING_T, linestyle="--", color="orange", label="WARNING")
plt.axhline(ALERT_T, linestyle="--", color="red", label="ALERT")
plt.legend()
plt.title("Drift Score Timeline")
plt.savefig("graph_timeline.png")

plt.figure(figsize=(8,5))
df[df["Actual"]==0]["drift_score"].hist(alpha=0.6, label="Normal")
df[df["Actual"]==1]["drift_score"].hist(alpha=0.6, label="Attack")
plt.legend()
plt.title("Drift Distribution")
plt.savefig("graph_distribution.png")

fpr_vals, tpr_vals, _ = roc_curve(df["Actual"], df["drift_score"])
roc_auc = auc(fpr_vals, tpr_vals)

plt.figure()
plt.plot(fpr_vals, tpr_vals, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.title("ROC Curve")
plt.legend()
plt.savefig("graph_roc.png")

precision_vals, recall_vals, _ = precision_recall_curve(df["Actual"], df["drift_score"])

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.title("Precision-Recall Curve")
plt.savefig("graph_pr.png")

plt.figure()
df.boxplot(column="drift_score", by="Actual")
plt.title("Drift Score Boxplot")
plt.savefig("graph_boxplot.png")

print("\nFiles Generated:")
print(" - table_summary.csv")
print(" - table_confusion_matrix.csv")
print(" - table_drift_stats.csv")
print(" - graph_timeline.png")
print(" - graph_distribution.png")
print(" - graph_roc.png")
print(" - graph_pr.png")
print(" - graph_boxplot.png")

print("\nAccuracy:", round(accuracy*100,2), "%")
print("Detection Rate (TPR):", round(recall*100,2), "%")
print("False Positive Rate (FPR):", round(fpr*100,2), "%")
