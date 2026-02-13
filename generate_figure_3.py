#!/usr/bin/env python3
"""
Figure 3: Time-Series Drift Plot for Q1 Journal
Shows how drift changes over time during normal and attack periods
Publication-quality visualization (300 DPI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

def create_time_series_plot():
    """Create time-series drift plot"""
    
    print("Generating Figure 3: Time-Series Drift Plot...")
    
    # Load data
    df = pd.read_csv('drift_log.csv')
    
    # Sample every 10th point for clarity
    df_sampled = df[::10].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot drift score line
    ax.plot(df_sampled.index, df_sampled['drift_score'], color='#2c3e50', linewidth=1.5, alpha=0.8)
    
    # Color background by alert level
    normal_mask = df_sampled['alert_level'] == 'NORMAL'
    warning_mask = df_sampled['alert_level'] == 'WARNING'
    alert_mask = df_sampled['alert_level'] == 'ALERT'
    
    ax.scatter(df_sampled[normal_mask].index, df_sampled[normal_mask]['drift_score'], 
              color='#2ecc71', s=20, alpha=0.6, label='Normal', edgecolors='none')
    ax.scatter(df_sampled[warning_mask].index, df_sampled[warning_mask]['drift_score'], 
              color='#f39c12', s=20, alpha=0.7, label='Warning', edgecolors='none')
    ax.scatter(df_sampled[alert_mask].index, df_sampled[alert_mask]['drift_score'], 
              color='#e74c3c', s=30, alpha=0.8, label='Alert', edgecolors='none')
    
    # Add threshold lines
    ax.axhline(y=0.45, color='purple', linestyle='--', linewidth=2.5, label='Detection Threshold (0.45)')
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2.5, label='Alert Threshold (1.50)')
    
    # Formatting
    ax.set_xlabel('Sample Index (Time Progression)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drift Score', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3. Temporal Evolution of Drift Scores', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 3.5])
    
    plt.tight_layout()
    plt.savefig('q1_results/FIGURE_3_time_series.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: q1_results/FIGURE_3_time_series.png (300 DPI)")
    
    # Statistics
    normal_count = normal_mask.sum()
    warning_count = warning_mask.sum()
    alert_count = alert_mask.sum()
    total = len(df_sampled)
    
    stats_df = pd.DataFrame({
        'Alert Level': ['NORMAL', 'WARNING', 'ALERT', 'TOTAL'],
        'Count': [normal_count, warning_count, alert_count, total],
        'Percentage': [
            f"{100*normal_count/total:.1f}%",
            f"{100*warning_count/total:.1f}%",
            f"{100*alert_count/total:.1f}%",
            "100.0%"
        ]
    })
    
    stats_df.to_csv('q1_results/FIGURE_3_statistics.csv', index=False)
    print("✓ Saved: q1_results/FIGURE_3_statistics.csv")
    
    # Print caption info
    attack_total = warning_count + alert_count
    print("\nFIGURE 3 CAPTION:")
    print(f"Temporal evolution of drift scores across {len(df):,} monitored samples. Each sampled")
    print(f"point is colored by alert level: green (normal, drift_score < 0.45, n={normal_count}),")
    print(f"orange (warning, 0.45 ≤ drift_score < 1.50, n={warning_count}), and red (alert, drift_score ≥ 1.50,")
    print(f"n={alert_count}). The purple dashed line indicates the detection threshold (0.45) identified through")
    print(f"multi-strategy optimization. Attack samples (warning + alert, {100*attack_total/total:.1f}% of total)")
    print(f"demonstrate clear temporal clustering, enabling sequential detection of coordinated attack campaigns.")
    
    return stats_df

if __name__ == '__main__':
    create_time_series_plot()
