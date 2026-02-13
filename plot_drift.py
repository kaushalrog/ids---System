#!/usr/bin/env python3
"""
Drift Visualization Tool
Plots drift scores and components over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys


def plot_drift_timeline(csv_file='drift_log.csv', output_file='drift_plot.png'):
    """
    Create visualization of drift scores over time.
    """
    try:
        # Read drift log
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print("ERROR: No data in drift log")
            return
        
        print(f"Loaded {len(df)} samples from {csv_file}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle('IDS Drift Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        df['sample_num'] = range(len(df))
        
        # Color code by alert level
        colors = []
        for level in df['alert_level']:
            if level == 'ALERT':
                colors.append('red')
            elif level == 'WARNING':
                colors.append('orange')
            else:
                colors.append('green')
        
        # Plot 1: Drift Score
        ax1 = axes[0]
        ax1.scatter(df['sample_num'], df['drift_score'], c=colors, alpha=0.6, s=30)
        ax1.axhline(y=2.5, color='orange', linestyle='--', linewidth=1, label='WARNING')
        ax1.axhline(y=4.0, color='red', linestyle='--', linewidth=1, label='ALERT')
        ax1.set_ylabel('Drift Score', fontweight='bold')
        ax1.set_title('Overall Drift Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Delta Component
        ax2 = axes[1]
        ax2.plot(df['sample_num'], df['delta'], color='blue', linewidth=1.5, alpha=0.7)
        ax2.fill_between(df['sample_num'], df['delta'], alpha=0.3, color='blue')
        ax2.set_ylabel('Delta', fontweight='bold')
        ax2.set_title('Deviation Magnitude')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Acceleration Component
        ax3 = axes[2]
        ax3.plot(df['sample_num'], df['acceleration'], color='purple', linewidth=1.5, alpha=0.7)
        ax3.fill_between(df['sample_num'], df['acceleration'], alpha=0.3, color='purple')
        ax3.set_ylabel('Acceleration', fontweight='bold')
        ax3.set_title('Rate of Change')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Error Component
        ax4 = axes[3]
        ax4.plot(df['sample_num'], df['prediction_error'], color='brown', linewidth=1.5, alpha=0.7)
        ax4.fill_between(df['sample_num'], df['prediction_error'], alpha=0.3, color='brown')
        ax4.set_ylabel('Prediction Error', fontweight='bold')
        ax4.set_xlabel('Sample Number', fontweight='bold')
        ax4.set_title('Temporal Consistency')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
        
        # Display statistics
        print("\n" + "="*60)
        print("Statistics")
        print("="*60)
        
        print(f"\nTotal samples: {len(df)}")
        print(f"Normal: {len(df[df['alert_level'] == 'NORMAL'])}")
        print(f"Warnings: {len(df[df['alert_level'] == 'WARNING'])}")
        print(f"Alerts: {len(df[df['alert_level'] == 'ALERT'])}")
        
        print(f"\nDrift Score Statistics:")
        print(f"  Mean: {df['drift_score'].mean():.4f}")
        print(f"  Std:  {df['drift_score'].std():.4f}")
        print(f"  Min:  {df['drift_score'].min():.4f}")
        print(f"  Max:  {df['drift_score'].max():.4f}")
        
        # Endpoint breakdown
        print(f"\nEndpoint Breakdown:")
        endpoint_counts = df['endpoint'].value_counts()
        for endpoint, count in endpoint_counts.items():
            print(f"  {endpoint}: {count}")
        
        # Alert breakdown by endpoint
        print(f"\nAlerts by Endpoint:")
        alerts_df = df[df['alert_level'].isin(['WARNING', 'ALERT'])]
        if len(alerts_df) > 0:
            alert_endpoints = alerts_df.groupby(['endpoint', 'alert_level']).size()
            for (endpoint, level), count in alert_endpoints.items():
                print(f"  {endpoint} [{level}]: {count}")
        else:
            print("  No alerts detected")
        
        # Show plot
        try:
            plt.show()
        except:
            print("\nNote: Cannot display plot (no display available)")
            print(f"Plot saved to {output_file}")
        
    except FileNotFoundError:
        print(f"ERROR: {csv_file} not found!")
        print("Run online_monitor.py first to generate drift log.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def plot_component_comparison(csv_file='drift_log.csv', output_file='components.png'):
    """
    Create stacked visualization of drift components.
    """
    try:
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print("ERROR: No data in drift log")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        df['sample_num'] = range(len(df))
        
        # Create stacked area plot
        ax.fill_between(df['sample_num'], 0, df['delta'], 
                        alpha=0.5, label='Delta', color='blue')
        ax.fill_between(df['sample_num'], df['delta'], 
                        df['delta'] + df['acceleration'],
                        alpha=0.5, label='Acceleration', color='purple')
        ax.fill_between(df['sample_num'], df['delta'] + df['acceleration'],
                        df['delta'] + df['acceleration'] + df['prediction_error'],
                        alpha=0.5, label='Prediction Error', color='brown')
        
        ax.plot(df['sample_num'], df['drift_score'], 
                color='red', linewidth=2, label='Total Drift Score')
        
        ax.set_xlabel('Sample Number', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Drift Component Breakdown', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Component plot saved to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    """
    Main visualization entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Drift Visualization Tool')
    parser.add_argument('--input', default='drift_log.csv',
                       help='Input drift log CSV (default: drift_log.csv)')
    parser.add_argument('--output', default='drift_plot.png',
                       help='Output plot file (default: drift_plot.png)')
    parser.add_argument('--components', action='store_true',
                       help='Also create component breakdown plot')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Drift Visualization Tool")
    print("="*60)
    print()
    
    # Create main timeline plot
    plot_drift_timeline(args.input, args.output)
    
    # Optionally create component plot
    if args.components:
        component_file = args.output.replace('.png', '_components.png')
        plot_component_comparison(args.input, component_file)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
