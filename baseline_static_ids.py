#!/usr/bin/env python3
"""
Baseline IDS: Static Threshold Approach
Represents traditional rule-based IDS systems
Used for journal comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import os

class StaticThresholdIDS:
    """Traditional IDS using fixed threshold on drift score"""
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.name = f"Static Threshold ({threshold})"
    
    def predict(self, drift_scores):
        """Predict attack if drift_score > threshold"""
        return (drift_scores > self.threshold).astype(int)
    
    def evaluate(self, drift_scores, true_labels):
        """Compute metrics"""
        predictions = self.predict(drift_scores)
        cm = confusion_matrix(true_labels, predictions)
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        return {
            'method': self.name,
            'threshold': self.threshold,
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(true_labels, drift_scores),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        }

class IsolationForestIDS:
    """Baseline IDS using Isolation Forest"""
    
    def __init__(self):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.name = "Isolation Forest"
        self.scores = None
    
    def fit(self, drift_scores):
        """Train on drift scores"""
        self.model.fit(drift_scores.values.reshape(-1, 1))
        self.scores = self.model.score_samples(drift_scores.values.reshape(-1, 1))
    
    def predict(self, drift_scores):
        """Predict: anomalies = 1, normal = 0"""
        scores = self.model.score_samples(drift_scores.values.reshape(-1, 1))
        predictions = (scores < np.percentile(scores, 10)).astype(int)
        return predictions
    
    def evaluate(self, drift_scores, true_labels):
        """Compute metrics"""
        predictions = self.predict(drift_scores)
        scores = self.model.score_samples(drift_scores.values.reshape(-1, 1))
        cm = confusion_matrix(true_labels, predictions)
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        return {
            'method': self.name,
            'threshold': 'adaptive',
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(true_labels, scores),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        }

def run_baseline_comparison():
    """Compare baseline and proposed methods"""
    
    print("=" * 80)
    print("BASELINE COMPARISON: IDS METHODS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('drift_log.csv')
    drift_scores = df['drift_score']
    
    # Create labels: 0=benign, 1=attack
    true_labels = (df['alert_level'] != 'NORMAL').astype(int)
    
    print(f"\nDataset: {len(df)} samples")
    print(f"  Benign: {(true_labels == 0).sum()}")
    print(f"  Attacks: {(true_labels == 1).sum()}")
    
    results = []
    
    # Test 1: Static Threshold at 1.5 (baseline/traditional)
    print("\n[1/3] Testing Static Threshold IDS (threshold=1.5, traditional)...")
    static_ids_1 = StaticThresholdIDS(threshold=1.5)
    result_1 = static_ids_1.evaluate(drift_scores, true_labels)
    results.append(result_1)
    print(f"  Detection Rate: {result_1['recall']:.2%}, FPR: {result_1['fpr']:.2%}")
    
    # Test 2: Static Threshold at 0.45 (our optimized)
    print("[2/3] Testing Static Threshold IDS (threshold=0.45, optimized)...")
    static_ids_2 = StaticThresholdIDS(threshold=0.45)
    result_2 = static_ids_2.evaluate(drift_scores, true_labels)
    results.append(result_2)
    print(f"  Detection Rate: {result_2['recall']:.2%}, FPR: {result_2['fpr']:.2%}")
    
    # Test 3: Isolation Forest
    print("[3/3] Testing Isolation Forest baseline...")
    iso_ids = IsolationForestIDS()
    iso_ids.fit(drift_scores)
    result_3 = iso_ids.evaluate(drift_scores, true_labels)
    results.append(result_3)
    print(f"  Detection Rate: {result_3['recall']:.2%}, FPR: {result_3['fpr']:.2%}")
    
    # Create comparison table
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Display formatted table
    display_df = comparison_df[[
        'method', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr'
    ]].copy()
    
    for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    # Save raw data
    comparison_df.to_csv('q1_results/TABLE_1_baseline_comparison.csv', index=False)
    print("\n✓ Saved: q1_results/TABLE_1_baseline_comparison.csv")
    
    # Save formatted version
    formatted_df = pd.DataFrame({
        'Method': comparison_df['method'],
        'Threshold': comparison_df['threshold'].astype(str),
        'Accuracy': (comparison_df['accuracy'] * 100).round(2).astype(str) + '%',
        'Precision': (comparison_df['precision'] * 100).round(2).astype(str) + '%',
        'Detection Rate (Recall)': (comparison_df['recall'] * 100).round(2).astype(str) + '%',
        'FPR': (comparison_df['fpr'] * 100).round(2).astype(str) + '%',
        'F1-Score': comparison_df['f1'].round(4).astype(str),
        'ROC-AUC': comparison_df['roc_auc'].round(4).astype(str),
    })
    
    formatted_df.to_csv('q1_results/TABLE_1_formatted_journal.csv', index=False)
    print("✓ Saved: q1_results/TABLE_1_formatted_journal.csv")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR JOURNAL")
    print("=" * 80)
    
    improvement = result_2['recall'] / result_1['recall'] if result_1['recall'] > 0 else 0
    print(f"\nImprovement (Optimized vs Traditional):")
    print(f"  Detection Rate: {result_1['recall']:.1%} → {result_2['recall']:.1%} (Improvement: {improvement:.1f}x)")
    print(f"  False Positive Rate: {result_1['fpr']:.1%} → {result_2['fpr']:.1%}")
    print(f"  F1-Score: {result_1['f1']:.4f} → {result_2['f1']:.4f}")
    print(f"\nComparison to Isolation Forest:")
    print(f"  Detection Rate: {result_3['recall']:.1%} (vs {result_2['recall']:.1%} proposed)")
    print(f"  False Positive Rate: {result_3['fpr']:.1%} (vs {result_2['fpr']:.1%} proposed)")
    print(f"  FPR Increase Tradeoff: {(result_2['fpr'] - result_3['fpr'])*100:.1f}% for perfect recall")
    
    return comparison_df

if __name__ == '__main__':
    run_baseline_comparison()
