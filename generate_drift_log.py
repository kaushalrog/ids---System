#!/usr/bin/env python3
"""
Generate realistic drift_log.csv from SRBH dataset
This simulates real IDS detection data without needing live Flask/attack setup
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("="*80)
print("Generating realistic drift_log.csv from SRBH dataset")
print("="*80)

# Load SRBH dataset
print("\nLoading SRBH preprocessed data...")
try:
    srbh_df = pd.read_csv('srbh_training/srbh_processed.csv')
    print(f"Loaded {len(srbh_df)} SRBH records")
except Exception as e:
    print(f"Error loading SRBH data: {e}")
    exit(1)

# Create realistic drift scores based on SRBH features
print("\nGenerating drift scores from SRBH features...")
drift_scores = []

for idx, row in srbh_df.iterrows():
    # Combine features to create drift score
    score = (
        2.5 * row.get('has_sql_kw', 0) +
        2.2 * row.get('has_cmd_kw', 0) +
        2.0 * row.get('has_traversal', 0) +
        0.006 * row.get('req_len', 0) +
        0.008 * row.get('body_len', 0) +
        0.003 * row.get('cookie_len', 0) +
        0.002 * row.get('ua_len', 0) +
        np.random.normal(1.0, 0.3)  # Normal background noise
    )
    drift_scores.append(max(0.1, score))

# Normalize scores
drift_scores = np.array(drift_scores)
drift_scores = (drift_scores - drift_scores.min()) / (drift_scores.max() - drift_scores.min() + 1e-6) * 4.0

# Generate alert levels based on thresholds
alert_levels = []
for score in drift_scores:
    if score > 2.5:
        alert_levels.append('ALERT')
    elif score > 1.5:
        alert_levels.append('WARNING')
    else:
        alert_levels.append('NORMAL')

# Generate timestamps
timestamps = []
base_time = datetime.now() - timedelta(hours=2)
for i in range(len(srbh_df)):
    timestamps.append(base_time + timedelta(seconds=i*2))

# Map endpoints based on features
endpoints = []
for idx, row in srbh_df.iterrows():
    if row.get('has_sql_kw', 0):
        endpoints.append('/login')
    elif row.get('has_cmd_kw', 0):
        endpoints.append('/ping')
    elif row.get('has_traversal', 0):
        endpoints.append('/download')
    else:
        endpoints.append('/api/data')

# Create DataFrame
data = {
    'timestamp': timestamps,
    'endpoint': endpoints,
    'drift_score': drift_scores,
    'alert_level': alert_levels,
    'cpu_percent': np.random.uniform(5, 80, len(srbh_df)),
    'memory_mb': np.random.uniform(100, 500, len(srbh_df)),
    'disk_read_mb': np.random.uniform(0, 50, len(srbh_df)),
    'disk_write_mb': np.random.uniform(0, 50, len(srbh_df))
}

drift_log_df = pd.DataFrame(data)

# Save to CSV
output_file = 'drift_log.csv'
drift_log_df.to_csv(output_file, index=False)

print(f"\nGenerated {output_file}:")
print(f"  - Total records: {len(drift_log_df)}")
print(f"  - Columns: {', '.join(drift_log_df.columns)}")

# Statistics
print(f"\nDrift Score Statistics:")
print(f"  - Min: {drift_log_df['drift_score'].min():.4f}")
print(f"  - Max: {drift_log_df['drift_score'].max():.4f}")
print(f"  - Mean: {drift_log_df['drift_score'].mean():.4f}")
print(f"  - Std: {drift_log_df['drift_score'].std():.4f}")

print(f"\nAlert Level Distribution:")
alert_counts = drift_log_df['alert_level'].value_counts()
for alert, count in alert_counts.items():
    pct = count / len(drift_log_df) * 100
    print(f"  - {alert:10}: {count:>5} ({pct:>5.1f}%)")

print(f"\nEndpoint Distribution:")
endpoint_counts = drift_log_df['endpoint'].value_counts()
for endpoint, count in endpoint_counts.items():
    print(f"  - {endpoint:15}: {count:>5}")

print(f"\nSuccess! drift_log.csv is ready for analysis.")
print("\n" + "="*80 + "\n")
