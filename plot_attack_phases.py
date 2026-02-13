import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("drift_log.csv")
df["sample_num"] = range(len(df))

attack_df = df[df["alert_level"].isin(["WARNING", "ALERT"])]

plt.figure(figsize=(14,6))
plt.plot(df["sample_num"], df["drift_score"], alpha=0.3, label="All Traffic")
plt.scatter(attack_df["sample_num"], attack_df["drift_score"], 
            color="red", label="Detected Attacks", s=40)

plt.axhline(2.5, linestyle="--", label="WARNING")
plt.axhline(4.0, linestyle="--", label="ALERT")

plt.xlabel("Request Number")
plt.ylabel("Drift Score")
plt.title("Attack Detection Timeline")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("attack_timeline.png", dpi=300)
plt.show()
