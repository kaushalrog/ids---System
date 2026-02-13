#!/usr/bin/env python3
"""
COMPREHENSIVE IDS ACCURACY REPORT
Combines multiple analysis approaches for maximum accuracy insights:
- Statistical analysis
- Anomaly detection metrics
- Attack pattern recognition
- Model validation
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "drift_log.csv"
THRESHOLDS_FILE = "optimized_thresholds.json"

class ComprehensiveAccuracyReport:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare comprehensive data features"""
        # Normalize drift score
        drift_min = self.df["drift_score"].min()
        drift_max = self.df["drift_score"].max()
        self.df["drift_normalized"] = (self.df["drift_score"] - drift_min) / (drift_max - drift_min + 1e-6)
        
        # Calculate drift acceleration (rate of change)
        self.df["drift_velocity"] = self.df["drift_score"].diff().fillna(0)
        self.df["drift_acceleration"] = self.df["drift_velocity"].diff().fillna(0)
        
        # Alert conversion
        self.df["Predicted"] = self.df["alert_level"].apply(
            lambda x: 1 if x in ["WARNING", "ALERT"] else 0
        )
        
        # Ground truth
        self.df["Actual"] = 0
        self.df.loc[self.df["endpoint"] != "/health", "Actual"] = (
            self.df.loc[self.df["endpoint"] != "/health", "drift_score"] > 1.8
        ).astype(int)
    
    def statistical_analysis(self):
        """Perform statistical tests for anomaly detection"""
        results = {}
        
        # Separate normal and anomaly drift scores
        normal_drift = self.df[self.df["Actual"] == 0]["drift_score"]
        anomaly_drift = self.df[self.df["Actual"] == 1]["drift_score"]
        
        # Distribution statistics
        results["Normal_Mean"] = float(normal_drift.mean())
        results["Normal_Std"] = float(normal_drift.std())
        results["Anomaly_Mean"] = float(anomaly_drift.mean())
        results["Anomaly_Std"] = float(anomaly_drift.std())
        
        # Separation metrics
        results["Mean_Separation"] = float(abs(results["Anomaly_Mean"] - results["Normal_Mean"]))
        results["Cohen_D"] = float(
            (results["Anomaly_Mean"] - results["Normal_Mean"]) / 
            np.sqrt((anomaly_drift.std()**2 + normal_drift.std()**2) / 2)
        )
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(normal_drift, anomaly_drift)
        results["T_Test_P_Value"] = float(p_value)
        results["T_Stat"] = float(t_stat)
        
        # Distribution testing (Kolmogorov-Smirnov)
        ks_stat, ks_pval = stats.ks_2samp(normal_drift, anomaly_drift)
        results["KS_Statistic"] = float(ks_stat)
        results["KS_P_Value"] = float(ks_pval)
        
        return results
    
    def detect_anomaly_patterns(self):
        """Detect specific attack patterns"""
        patterns = {}
        
        # High frequency detection
        high_drift_samples = self.df[self.df["drift_score"] > 3.0]
        patterns["High_Drift_Samples"] = len(high_drift_samples)
        
        # Consecutive alerts (attack burst)
        alert_mask = self.df["Predicted"] == 1
        consecutive_alerts = (alert_mask.astype(int).diff() == 1).sum()
        patterns["Alert_Bursts"] = int(consecutive_alerts)
        
        # Drift trend (increasing/decreasing)
        drift_trend = np.polyfit(range(len(self.df)), self.df["drift_score"], 1)[0]
        patterns["Drift_Trend_Slope"] = float(drift_trend)
        
        # Endpoint-specific attack rates
        patterns["Attack_By_Endpoint"] = {}
        for endpoint in self.df["endpoint"].unique():
            endpoint_data = self.df[self.df["endpoint"] == endpoint]
            if len(endpoint_data) > 0:
                attack_rate = endpoint_data["Actual"].sum() / len(endpoint_data)
                patterns["Attack_By_Endpoint"][endpoint] = float(attack_rate)
        
        return patterns
    
    def detection_efficiency(self):
        """Calculate detection efficiency metrics"""
        y_true = self.df["Actual"].values
        y_pred = self.df["Predicted"].values
        
        efficiency = {}
        
        # Time to detection (after attack starts)
        attack_start_idx = np.where(y_true == 1)[0]
        if len(attack_start_idx) > 0:
            detection_indices = np.where(y_pred == 1)[0]
            if len(detection_indices) > 0:
                min_detection = np.min(detection_indices)
                min_attack = np.min(attack_start_idx)
                efficiency["Time_To_Detection_Samples"] = int(max(0, min_detection - min_attack))
        
        # Alert purity (% of alerts that are true attacks)
        alert_total = (y_pred == 1).sum()
        true_alerts = ((y_pred == 1) & (y_true == 1)).sum()
        efficiency["Alert_Purity"] = float(true_alerts / alert_total) if alert_total > 0 else 0
        
        # Coverage (% of attacks detected)
        attack_total = (y_true == 1).sum()
        efficiency["Attack_Coverage"] = float(true_alerts / attack_total) if attack_total > 0 else 0
        
        # False alarm rate
        false_alarms = ((y_pred == 1) & (y_true == 0)).sum()
        efficiency["False_Alarm_Rate"] = float(false_alarms / len(y_pred))
        
        return efficiency
    
    def model_robustness(self):
        """Assess model robustness across data segments"""
        robustness = {}
        
        # Split into quartiles
        quartile_size = len(self.df) // 4
        quartile_performance = []
        
        for i in range(4):
            start = i * quartile_size
            end = min((i + 1) * quartile_size, len(self.df))
            
            segment = self.df.iloc[start:end]
            if len(segment) > 0:
                y_true = segment["Actual"].values
                y_pred = segment["Predicted"].values
                
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                quartile_performance.append({
                    f"Q{i+1}_Precision": float(precision),
                    f"Q{i+1}_Recall": float(recall)
                })
        
        for q_perf in quartile_performance:
            robustness.update(q_perf)
        
        # Variance in performance (lower is better)
        precisions = [q[f"Q{i+1}_Precision"] for i, q in enumerate(quartile_performance)]
        recalls = [q[f"Q{i+1}_Recall"] for i, q in enumerate(quartile_performance)]
        
        robustness["Precision_Variance"] = float(np.var(precisions))
        robustness["Recall_Variance"] = float(np.var(recalls))
        robustness["Consistency_Score"] = float(1 - (np.var(precisions) + np.var(recalls)) / 2)
        
        return robustness
    
    def generate_comprehensive_report(self):
        """Generate full accuracy report"""
        print("\n" + "="*100)
        print("COMPREHENSIVE IDS ACCURACY REPORT")
        print("="*100)
        
        # Statistical Analysis
        stat_analysis = self.statistical_analysis()
        print("\n[1] STATISTICAL ANOMALY ANALYSIS")
        print("-" * 100)
        for key, val in stat_analysis.items():
            print(f"  {key:.<50} {val:.6f}")
        
        # Pattern Detection
        patterns = self.detect_anomaly_patterns()
        print("\n[2] ATTACK PATTERN DETECTION")
        print("-" * 100)
        for key, val in patterns.items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for sub_key, sub_val in val.items():
                    print(f"    {sub_key:.<48} {sub_val:.4f}")
            else:
                print(f"  {key:.<50} {val}")
        
        # Detection Efficiency
        efficiency = self.detection_efficiency()
        print("\n[3] DETECTION EFFICIENCY METRICS")
        print("-" * 100)
        for key, val in efficiency.items():
            print(f"  {key:.<50} {val:.4f}")
        
        # Model Robustness
        robustness = self.model_robustness()
        print("\n[4] MODEL ROBUSTNESS ACROSS DATA SEGMENTS")
        print("-" * 100)
        for key, val in robustness.items():
            print(f"  {key:.<50} {val:.6f}")
        
        # Summary
        print("\n[5] ACCURACY SUMMARY")
        print("-" * 100)
        print(f"  Data Points Analyzed:         {len(self.df)}")
        print(f"  Normal Samples:               {(self.df['Actual'] == 0).sum()}")
        print(f"  Attack Samples:               {(self.df['Actual'] == 1).sum()}")
        print(f"  True Positives:               {((self.df['Predicted'] == 1) & (self.df['Actual'] == 1)).sum()}")
        print(f"  False Positives:              {((self.df['Predicted'] == 1) & (self.df['Actual'] == 0)).sum()}")
        print(f"  False Negatives:              {((self.df['Predicted'] == 0) & (self.df['Actual'] == 1)).sum()}")
        print(f"  True Negatives:               {((self.df['Predicted'] == 0) & (self.df['Actual'] == 0)).sum()}")
        print(f"  Cohen's D (Effect Size):      {stat_analysis['Cohen_D']:.4f}")
        print(f"  Statistical Significance:     {'Yes' if stat_analysis['T_Test_P_Value'] < 0.05 else 'No'}")
        print(f"  Model Consistency Score:      {robustness['Consistency_Score']:.4f}")
        
        # Save report
        combined_report = {
            "statistical_analysis": stat_analysis,
            "anomaly_patterns": patterns,
            "detection_efficiency": efficiency,
            "model_robustness": robustness
        }
        
        with open("comprehensive_accuracy_report.json", 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        print("\n" + "="*100)
        print("✓ Report saved to: comprehensive_accuracy_report.json")
        print("="*100 + "\n")
        
        return combined_report
    
    def plot_comprehensive_analysis(self):
        """Generate comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Drift score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df[self.df["Actual"] == 0]["drift_score"], bins=15, alpha=0.6, label="Normal", color="green")
        ax1.hist(self.df[self.df["Actual"] == 1]["drift_score"], bins=15, alpha=0.6, label="Attack", color="red")
        ax1.set_title("Drift Score Distribution", fontweight='bold')
        ax1.set_xlabel("Drift Score")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Drift timeline
        ax2 = fig.add_subplot(gs[0, 1:])
        ax2.plot(self.df.index, self.df["drift_score"], linewidth=1.5, label="Drift Score", color="blue")
        ax2.fill_between(self.df.index, self.df[self.df["Actual"] == 1].index, 
                        self.df.loc[self.df["Actual"] == 1, "drift_score"].values,
                        alpha=0.3, color="red", label="Attack Periods")
        ax2.set_title("Drift Score Timeline with Attack Periods", fontweight='bold')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Drift Score")
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Alert timeline
        ax3 = fig.add_subplot(gs[1, 0])
        colors = self.df["alert_level"].map({"NORMAL": "green", "WARNING": "orange", "ALERT": "red"})
        ax3.scatter(self.df.index, self.df["drift_score"], c=colors, s=20, alpha=0.7)
        ax3.set_title("Alert Level Timeline", fontweight='bold')
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("Drift Score")
        ax3.grid(alpha=0.3)
        
        # 4. Velocity analysis
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.df.index, self.df["drift_velocity"], linewidth=1, label="Velocity", color="orange")
        ax4.plot(self.df.index, self.df["drift_acceleration"], linewidth=1, label="Acceleration", color="purple")
        ax4.set_title("Drift Velocity & Acceleration", fontweight='bold')
        ax4.set_xlabel("Sample Index")
        ax4.set_ylabel("Rate of Change")
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Endpoint distribution
        ax5 = fig.add_subplot(gs[1, 2])
        endpoint_attacks = self.df.groupby("endpoint")["Actual"].apply(lambda x: x.sum() / len(x))
        endpoint_attacks.plot(kind="barh", ax=ax5, color="steelblue")
        ax5.set_title("Attack Rate by Endpoint", fontweight='bold')
        ax5.set_xlabel("Attack Rate")
        ax5.grid(alpha=0.3, axis='x')
        
        # 6. Precision by segment
        ax6 = fig.add_subplot(gs[2, 0])
        quartile_size = len(self.df) // 4
        quartile_precision = []
        for i in range(4):
            start = i * quartile_size
            end = min((i + 1) * quartile_size, len(self.df))
            segment = self.df.iloc[start:end]
            tp = ((segment["Predicted"] == 1) & (segment["Actual"] == 1)).sum()
            fp = ((segment["Predicted"] == 1) & (segment["Actual"] == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            quartile_precision.append(prec)
        
        ax6.bar(range(4), quartile_precision, color="skyblue", edgecolor="navy")
        ax6.set_title("Precision by Quartile", fontweight='bold')
        ax6.set_xlabel("Quartile")
        ax6.set_ylabel("Precision")
        ax6.set_xticks(range(4))
        ax6.set_xticklabels([f"Q{i+1}" for i in range(4)])
        ax6.grid(alpha=0.3, axis='y')
        
        # 7. Recall by segment
        ax7 = fig.add_subplot(gs[2, 1])
        quartile_recall = []
        for i in range(4):
            start = i * quartile_size
            end = min((i + 1) * quartile_size, len(self.df))
            segment = self.df.iloc[start:end]
            tp = ((segment["Predicted"] == 1) & (segment["Actual"] == 1)).sum()
            fn = ((segment["Predicted"] == 0) & (segment["Actual"] == 1)).sum()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            quartile_recall.append(rec)
        
        ax7.bar(range(4), quartile_recall, color="lightcoral", edgecolor="darkred")
        ax7.set_title("Recall by Quartile", fontweight='bold')
        ax7.set_xlabel("Quartile")
        ax7.set_ylabel("Recall")
        ax7.set_xticks(range(4))
        ax7.set_xticklabels([f"Q{i+1}" for i in range(4)])
        ax7.grid(alpha=0.3, axis='y')
        
        # 8. ROC-like curve
        ax8 = fig.add_subplot(gs[2, 2])
        sorted_scores = np.sort(self.df["drift_score"].unique())
        tpr_list = []
        fpr_list = []
        for threshold in sorted_scores:
            tp = ((self.df["drift_score"] >= threshold) & (self.df["Actual"] == 1)).sum()
            fp = ((self.df["drift_score"] >= threshold) & (self.df["Actual"] == 0)).sum()
            tn = ((self.df["drift_score"] < threshold) & (self.df["Actual"] == 0)).sum()
            fn = ((self.df["drift_score"] < threshold) & (self.df["Actual"] == 1)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        ax8.plot(fpr_list, tpr_list, linewidth=2, color="darkblue")
        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax8.set_title("ROC Curve", fontweight='bold')
        ax8.set_xlabel("FPR")
        ax8.set_ylabel("TPR")
        ax8.grid(alpha=0.3)
        
        plt.savefig("comprehensive_accuracy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization saved: comprehensive_accuracy_analysis.png")


def main():
    report = ComprehensiveAccuracyReport(INPUT_CSV)
    report.generate_comprehensive_report()
    report.plot_comprehensive_analysis()


if __name__ == "__main__":
    main()
