#!/usr/bin/env python3
"""
ADVANCED THRESHOLD OPTIMIZER
Optimizes alert thresholds using multiple strategies:
- Grid search with multi-objective optimization
- ROC-based optimization
- Youden index maximization
- Cost-sensitive learning
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_curve, auc, balanced_accuracy_score
)
import matplotlib.pyplot as plt

INPUT_CSV = "drift_log.csv"
OUTPUT_FILE = "optimized_thresholds.json"

class AdvancedThresholdOptimizer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data with ground truth labels"""
        # Create normalized prediction probability
        drift_min = self.df["drift_score"].min()
        drift_max = self.df["drift_score"].max()
        self.df["pred_proba"] = (self.df["drift_score"] - drift_min) / (drift_max - drift_min + 1e-6)
        
        # Create ground truth labels
        self.df["Actual"] = 0
        self.df.loc[self.df["endpoint"] != "/health", "Actual"] = (
            self.df.loc[self.df["endpoint"] != "/health", "drift_score"] > 1.8
        ).astype(int)
    
    def optimize_f1_threshold(self):
        """Find threshold that maximizes F1-score"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.linspace(0, 1, 201):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def optimize_youden_index(self):
        """Maximize Youden index (TPR - FPR)"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        
        return thresholds[best_idx], youden[best_idx]
    
    def optimize_balanced_accuracy(self):
        """Find threshold maximizing balanced accuracy"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        best_threshold = 0.5
        best_acc = 0
        
        for threshold in np.linspace(0, 1, 201):
            y_pred = (y_proba >= threshold).astype(int)
            acc = balanced_accuracy_score(y_true, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return best_threshold, best_acc
    
    def optimize_cost_sensitive(self, fn_cost=2.0, fp_cost=1.0):
        """Minimize cost: fn_cost * FN + fp_cost * FP"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        best_threshold = 0.5
        best_cost = float('inf')
        
        for threshold in np.linspace(0, 1, 201):
            y_pred = (y_proba >= threshold).astype(int)
            
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            cost = fn_cost * fn + fp_cost * fp
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        return best_threshold, best_cost
    
    def get_metrics_at_threshold(self, threshold):
        """Calculate all metrics at a given threshold"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        y_pred = (y_proba >= threshold).astype(int)
        
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred)
        }
    
    def find_roc_optimal(self):
        """Find point on ROC curve closest to (0, 1)"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Distance to (0, 1)
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        best_idx = np.argmin(distances)
        
        return thresholds[best_idx], fpr[best_idx], tpr[best_idx]
    
    def generate_threshold_curve(self):
        """Generate F1 curve across all thresholds"""
        y_true = self.df["Actual"].values
        y_proba = self.df["pred_proba"].values
        
        thresholds = np.linspace(0, 1, 201)
        f1_scores = []
        accuracies = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
        return thresholds, f1_scores, accuracies, precisions, recalls
    
    def run_optimization(self):
        """Run all optimization strategies"""
        print("\n" + "="*80)
        print("ADVANCED THRESHOLD OPTIMIZATION")
        print("="*80)
        
        # Strategy 1: F1 Optimization
        f1_threshold, f1_score_val = self.optimize_f1_threshold()
        print(f"\n[Strategy 1] F1-Score Optimization")
        print(f"  Optimal Threshold: {f1_threshold:.4f}")
        print(f"  F1-Score: {f1_score_val:.4f}")
        print(f"  Metrics: {self.get_metrics_at_threshold(f1_threshold)}")
        
        # Strategy 2: Youden Index
        youden_threshold, youden_val = self.optimize_youden_index()
        print(f"\n[Strategy 2] Youden Index Optimization")
        print(f"  Optimal Threshold: {youden_threshold:.4f}")
        print(f"  Youden Index: {youden_val:.4f}")
        print(f"  Metrics: {self.get_metrics_at_threshold(youden_threshold)}")
        
        # Strategy 3: Balanced Accuracy
        ba_threshold, ba_val = self.optimize_balanced_accuracy()
        print(f"\n[Strategy 3] Balanced Accuracy Optimization")
        print(f"  Optimal Threshold: {ba_threshold:.4f}")
        print(f"  Balanced Accuracy: {ba_val:.4f}")
        print(f"  Metrics: {self.get_metrics_at_threshold(ba_threshold)}")
        
        # Strategy 4: Cost Sensitive
        cost_threshold, cost_val = self.optimize_cost_sensitive(fn_cost=2.0, fp_cost=1.0)
        print(f"\n[Strategy 4] Cost-Sensitive Learning (FN_cost=2, FP_cost=1)")
        print(f"  Optimal Threshold: {cost_threshold:.4f}")
        print(f"  Total Cost: {cost_val:.4f}")
        print(f"  Metrics: {self.get_metrics_at_threshold(cost_threshold)}")
        
        # Strategy 5: ROC Optimal
        roc_threshold, roc_fpr, roc_tpr = self.find_roc_optimal()
        print(f"\n[Strategy 5] ROC Optimal Point")
        print(f"  Optimal Threshold: {roc_threshold:.4f}")
        print(f"  FPR: {roc_fpr:.4f}, TPR: {roc_tpr:.4f}")
        print(f"  Metrics: {self.get_metrics_at_threshold(roc_threshold)}")
        
        # Generate visualization
        self.plot_threshold_curves()
        
        # Generate recommendations
        recommendations = {
            "f1_optimized": {"threshold": float(f1_threshold), **self.get_metrics_at_threshold(f1_threshold)},
            "youden_optimized": {"threshold": float(youden_threshold), **self.get_metrics_at_threshold(youden_threshold)},
            "balanced_accuracy": {"threshold": float(ba_threshold), **self.get_metrics_at_threshold(ba_threshold)},
            "cost_sensitive": {"threshold": float(cost_threshold), **self.get_metrics_at_threshold(cost_threshold)},
            "roc_optimal": {"threshold": float(roc_threshold), **self.get_metrics_at_threshold(roc_threshold)},
            "ensemble_recommendation": float((f1_threshold + ba_threshold + roc_threshold) / 3)
        }
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n✓ Optimized thresholds saved to {OUTPUT_FILE}")
        
        return recommendations
    
    def plot_threshold_curves(self):
        """Plot all metric curves across thresholds"""
        thresholds, f1_scores, accuracies, precisions, recalls = self.generate_threshold_curve()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1 Score
        axes[0, 0].plot(thresholds, f1_scores, linewidth=2, color='blue')
        axes[0, 0].set_title("F1-Score vs Threshold", fontweight='bold')
        axes[0, 0].set_xlabel("Threshold")
        axes[0, 0].set_ylabel("F1-Score")
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(thresholds, accuracies, linewidth=2, color='green')
        axes[0, 1].set_title("Accuracy vs Threshold", fontweight='bold')
        axes[0, 1].set_xlabel("Threshold")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].grid(alpha=0.3)
        
        # Precision & Recall
        axes[1, 0].plot(thresholds, precisions, linewidth=2, color='orange', label='Precision')
        axes[1, 0].plot(thresholds, recalls, linewidth=2, color='red', label='Recall')
        axes[1, 0].set_title("Precision & Recall vs Threshold", fontweight='bold')
        axes[1, 0].set_xlabel("Threshold")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # All metrics
        axes[1, 1].plot(thresholds, f1_scores, linewidth=2, label='F1')
        axes[1, 1].plot(thresholds, accuracies, linewidth=2, label='Accuracy')
        axes[1, 1].plot(thresholds, precisions, linewidth=2, label='Precision')
        axes[1, 1].plot(thresholds, recalls, linewidth=2, label='Recall')
        axes[1, 1].set_title("All Metrics vs Threshold", fontweight='bold')
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("threshold_optimization_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualization saved: threshold_optimization_curves.png")


def main():
    optimizer = AdvancedThresholdOptimizer(INPUT_CSV)
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
