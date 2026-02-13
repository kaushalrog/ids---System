#!/usr/bin/env python3
"""
Baseline Generator
Generates normal traffic baseline by exercising all legitimate endpoints.
"""

import time
import requests
import json
import sys
import os


def clear_telemetry():
    """
    Clear existing telemetry log.
    """
    if os.path.exists('telemetry.jsonl'):
        os.remove('telemetry.jsonl')
        print("Cleared existing telemetry log")


def wait_for_app():
    """
    Wait for Flask app to be ready.
    """
    print("Waiting for Flask app to be ready...")
    max_retries = 30
    
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:5000/health', timeout=1)
            if response.status_code == 200:
                print("Flask app is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        sys.stdout.write(f"\rAttempt {i+1}/{max_retries}...")
        sys.stdout.flush()
    
    print("\nERROR: Flask app not responding!")
    print("Please start app.py in another terminal first.")
    return False


def generate_normal_traffic(num_samples=50):
    """
    Generate normal traffic by exercising all endpoints.
    
    Args:
        num_samples: Number of samples per endpoint
    """
    print(f"\nGenerating {num_samples} samples per endpoint...")
    
    base_url = 'http://localhost:5000'
    
    # Valid login attempts
    print("[1/3] Generating normal login traffic...")
    for i in range(num_samples):
        try:
            response = requests.post(
                f'{base_url}/login',
                data={'username': 'admin', 'password': 'admin123'},
                timeout=5
            )
            time.sleep(0.1)  # Small delay between requests
            
            if (i + 1) % 10 == 0:
                print(f"  Login: {i+1}/{num_samples}")
        except Exception as e:
            print(f"  Warning: Login request failed: {e}")
    
    # Normal ping requests
    print("[2/3] Generating normal ping traffic...")
    for i in range(num_samples):
        try:
            response = requests.get(
                f'{base_url}/ping?host=127.0.0.1',
                timeout=5
            )
            time.sleep(0.1)
            
            if (i + 1) % 10 == 0:
                print(f"  Ping: {i+1}/{num_samples}")
        except Exception as e:
            print(f"  Warning: Ping request failed: {e}")
    
    # Normal download requests
    print("[3/3] Generating normal download traffic...")
    for i in range(num_samples):
        try:
            response = requests.get(
                f'{base_url}/download?file=readme.txt',
                timeout=5
            )
            time.sleep(0.1)
            
            if (i + 1) % 10 == 0:
                print(f"  Download: {i+1}/{num_samples}")
        except Exception as e:
            print(f"  Warning: Download request failed: {e}")
    
    print("\nNormal traffic generation complete!")


def extract_baseline():
    """
    Extract baseline from telemetry log.
    """
    print("\nExtracting baseline statistics...")
    
    if not os.path.exists('telemetry.jsonl'):
        print("ERROR: telemetry.jsonl not found!")
        return False
    
    # Copy telemetry to baseline
    samples = []
    with open('telemetry.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if len(samples) == 0:
        print("ERROR: No samples in telemetry.jsonl!")
        return False
    
    # Write baseline
    with open('normal_intent.jsonl', 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Baseline created: {len(samples)} samples")
    print("Saved to: normal_intent.jsonl")
    
    # Display baseline statistics
    print("\nBaseline Statistics:")
    
    # Sample feature
    if samples:
        features = [k for k in samples[0].keys() 
                   if k not in ['timestamp', 'endpoint', 'status']]
        
        print(f"Features: {len(features)}")
        
        # Show stats for a few key features
        import numpy as np
        
        key_features = ['flask_cpu', 'flask_memory_mb', 'mysql_cpu', 'mysql_memory_mb']
        for feature in key_features:
            if feature in samples[0]:
                values = [s[feature] for s in samples]
                print(f"  {feature}:")
                print(f"    Mean: {np.mean(values):.4f}")
                print(f"    Std:  {np.std(values):.4f}")
                print(f"    Min:  {np.min(values):.4f}")
                print(f"    Max:  {np.max(values):.4f}")
    
    return True


def main():
    """
    Main baseline generation workflow.
    """
    print("="*60)
    print("Baseline Generator")
    print("="*60)
    print()
    print("This script will:")
    print("1. Clear existing telemetry")
    print("2. Wait for Flask app to be ready")
    print("3. Generate normal traffic")
    print("4. Extract baseline statistics")
    print()
    
    # Confirm Flask app is running
    if not wait_for_app():
        return
    
    # Clear previous telemetry
    clear_telemetry()
    
    # Generate traffic
    generate_normal_traffic(num_samples=50)
    
    # Wait for telemetry to be written
    print("\nWaiting for telemetry to be written...")
    time.sleep(2)
    
    # Extract baseline
    if extract_baseline():
        print("\n" + "="*60)
        print("Baseline generation complete!")
        print("="*60)
        print("\nYou can now:")
        print("1. Start the monitor: python3 online_monitor.py")
        print("2. Run attack simulations: bash attack_simulation.sh")
    else:
        print("\nERROR: Failed to extract baseline")
        return


if __name__ == '__main__':
    main()
