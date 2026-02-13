#!/usr/bin/env python3
"""
Complete IDS Workflow - Real Data Collection and Analysis
Baseline (normal) + Attacks + Detection + Full Analysis
"""

import time
import requests
import json
import subprocess
import os
import sys

def step(n, msg):
    print(f"\n[STEP {n}] {msg}")

def info(msg):
    print(f"   {msg}")

print("\n" + "="*80)
print("  COMPLETE IDS END-TO-END WORKFLOW")
print("="*80)

# STEP 1: Wait for Flask app
step(1, "Waiting for Flask app on port 5000")
max_retries = 30
app_ready = False
for i in range(max_retries):
    try:
        requests.get('http://localhost:5000/health', timeout=1)
        info("Flask app is ready!")
        app_ready = True
        break
    except:
        pass
    time.sleep(1)
    sys.stdout.write(f"\r   Attempt {i+1}/{max_retries}...")
    sys.stdout.flush()

if not app_ready:
    info("ERROR: Flask app not responding!")
    sys.exit(1)

# STEP 2: Clear old files
step(2, "Clearing old telemetry and baseline files")
for f in ['telemetry.jsonl', 'normal_intent.jsonl', 'drift_log.csv']:
    if os.path.exists(f):
        os.remove(f)
        info(f"Cleared {f}")

# STEP 3: Generate normal baseline traffic
step(3, "Generating NORMAL traffic baseline (50 samples per endpoint)")
info("Sending legitimate requests to establish normal behavior...")

base_url = 'http://localhost:5000'
requests_normal = 0

endpoints_config = [
    {'endpoint': '/login', 'method': 'POST', 'data': {'username': 'admin', 'password': 'admin123'}},
    {'endpoint': '/ping', 'method': 'GET', 'params': {'host': '127.0.0.1'}},
    {'endpoint': '/download', 'method': 'GET', 'params': {'file': 'readme.txt'}},
    {'endpoint': '/health', 'method': 'GET', 'params': {}},
]

for config in endpoints_config:
    endpoint = config['endpoint']
    sys.stdout.write(f"   {endpoint}: ")
    sys.stdout.flush()
    for i in range(50):
        try:
            if config['method'] == 'POST':
                requests.post(f'{base_url}{endpoint}', data=config['data'], timeout=5)
            else:
                requests.get(f'{base_url}{endpoint}', params=config.get('params', {}), timeout=5)
            requests_normal += 1
            if (i + 1) % 10 == 0:
                sys.stdout.write(f"{i+1}/50 ")
                sys.stdout.flush()
        except:
            pass
        time.sleep(0.05)
    print()

info(f"Total baseline requests sent: {requests_normal}")
time.sleep(2)

# STEP 4: Extract baseline statistics
step(4, "Extracting baseline from telemetry data")
if os.path.exists('telemetry.jsonl'):
    with open('telemetry.jsonl', 'r') as f:
        all_lines = f.readlines()
    
    baseline_size = int(len(all_lines) * 0.8)
    with open('normal_intent.jsonl', 'w') as f:
        for line in all_lines[:baseline_size]:
            f.write(line)
    
    info(f"Total telemetry records: {len(all_lines)}")
    info(f"Baseline records (80%): {baseline_size}")
    info(f"Baseline file created: normal_intent.jsonl")
else:
    info("WARNING: No telemetry.jsonl found!")

# STEP 5: Generate attack traffic
step(5, "Generating ATTACK traffic (malicious requests)")
info("Sending SQL injection, command injection, and path traversal attacks...")

attack_patterns = [
    {'endpoint': '/login', 'method': 'POST', 'data': {'username': "admin' OR '1'='1", 'password': "' OR '1'='1"}},
    {'endpoint': '/ping', 'method': 'GET', 'params': {'host': '127.0.0.1; cat /etc/passwd'}},
    {'endpoint': '/download', 'method': 'GET', 'params': {'file': '../../etc/passwd'}},
    {'endpoint': '/login', 'method': 'POST', 'data': {'username': 'x'*2000, 'password': 'y'*2000}},
]

attack_count = 0
for i in range(200):
    pattern = attack_patterns[i % len(attack_patterns)]
    try:
        if pattern['method'] == 'POST':
            requests.post(f"{base_url}{pattern['endpoint']}", data=pattern['data'], timeout=5)
        else:
            requests.get(f"{base_url}{pattern['endpoint']}", params=pattern['params'], timeout=5)
        attack_count += 1
        if (i + 1) % 50 == 0:
            info(f"{i+1} attack requests sent")
    except:
        pass
    time.sleep(0.02)

info(f"Total attack requests sent: {attack_count}")
time.sleep(2)

# STEP 6: Run online monitor (drift detection)
step(6, "Running online monitor with drift detection")
info("Starting drift analysis with SRBH-trained thresholds...")
try:
    subprocess.run(['python', 'online_monitor.py'], capture_output=True, timeout=60)
    info("Drift detection completed")
except Exception as e:
    info(f"Warning: {e}")
time.sleep(1)

# STEP 7: Verify drift_log.csv
step(7, "Checking detection results")
if os.path.exists('drift_log.csv'):
    with open('drift_log.csv', 'r') as f:
        lines = f.readlines()
    row_count = len(lines) - 1
    info(f"drift_log.csv created: {row_count} detection records")
    
    import csv
    with open('drift_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        alert_counts = {'NORMAL': 0, 'WARNING': 0, 'ALERT': 0}
        for row in reader:
            alert = row.get('alert_level', 'UNKNOWN')
            if alert in alert_counts:
                alert_counts[alert] += 1
    
    info(f"Alert statistics:")
    info(f"  - NORMAL detections:  {alert_counts['NORMAL']}")
    info(f"  - WARNING detections: {alert_counts['WARNING']}")
    info(f"  - ALERT detections:   {alert_counts['ALERT']} (ATTACKS FOUND)")
else:
    info("ERROR: drift_log.csv not found!")

# STEP 8: Run analysis scripts
step(8, "Running accuracy analysis (3 advanced analyzers)")

analyzers = [
    ('improved_results_analyzer.py', 'Improved Results Analyzer'),
    ('advanced_threshold_optimizer.py', 'Threshold Optimizer'),
    ('comprehensive_accuracy_report.py', 'Comprehensive Report'),
]

for script, name in analyzers:
    sys.stdout.write(f"   {name:.<50} ")
    sys.stdout.flush()
    try:
        subprocess.run(['python', script], capture_output=True, timeout=60)
        print("DONE")
    except Exception as e:
        print(f"ERROR: {e}")

# STEP 9: Summary
print("\n" + "="*80)
print("  WORKFLOW COMPLETE - RESULTS GENERATED")
print("="*80)

print("\nData Collection:")
files_check = [
    ('telemetry.jsonl', 'Raw system telemetry'),
    ('normal_intent.jsonl', 'Baseline (normal behavior)'),
    ('drift_log.csv', 'IDS Detection Log (MAIN)'),
]
for fname, desc in files_check:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"   [OK] {fname:25} {size:>10,} bytes - {desc}")
    else:
        print(f"   [XX] {fname:25} (MISSING)")

print("\nAnalysis Output:")
analysis = [
    ('improved_metrics_summary.csv', 'Accuracy metrics table'),
    ('improved_phase_analysis.csv', 'Phase-by-phase analysis'),
    ('optimized_thresholds.json', 'Optimal thresholds'),
    ('comprehensive_accuracy_report.json', 'Statistical analysis'),
]
for fname, desc in analysis:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"   [OK] {fname:35} {size:>10,} bytes")
    else:
        print(f"   [XX] {fname:35} (MISSING)")

print("\nVisualization Charts (PNG):")
charts = [
    'improved_roc_curve.png',
    'improved_confusion_matrix.png',
    'improved_drift_distribution.png',
    'improved_metrics_comparison.png',
    'threshold_optimization_curves.png',
    'comprehensive_accuracy_analysis.png',
]
for fname in charts:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"   [OK] {fname:35} {size:>10,} bytes")
    else:
        print(f"   [XX] {fname:35} (MISSING)")

print("\n" + "="*80)
print("  RESULTS ARE READY FOR REVIEW!")
print("="*80)
print("\nNext Steps:")
print("  1. Open CSV files to view metrics and analysis")
print("  2. Open PNG charts for visualizations")
print("  3. Review JSON files for detailed threshold data")
print("\n")
