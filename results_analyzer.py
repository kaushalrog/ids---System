#!/usr/bin/env python3
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


INPUT = "drift_log.csv"


def main():
    print("=" * 70)
    print("RESEARCH-GRADE IDS RESULT ANALYZER")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    if "alert_level" not in df.columns:
        print("ERROR: drift_log.csv missing alert_level column")
        return

    df["Predicted"] = df["alert_level"].apply(
        lambda x: 1 if x in ["WARNING", "ALERT"] else 0
    )

    # Ground truth labeling by endpoint phase
    df["Actual"] = 0

    df.loc[(df["endpoint"] == "/login") & (df.index >= 10) & (df.index < 15), "Actual"] = 1
    df.loc[(df["endpoint"] == "/ping") & (df.index >= 20) & (df.index < 25), "Actual"] = 1
    df.loc[(df["endpoint"] == "/download") & (df.index >= 25) & (df.index < 30), "Actual"] = 1

    cm = confusion_matrix(df["Actual"], df["Predicted"])
    report = classification_report(df["Actual"], df["Predicted"], digits=4)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    df.to_csv("final_results_labeled.csv", index=False)
    print("\nSaved labeled results to final_results_labeled.csv")


if __name__ == "__main__":
    main()
