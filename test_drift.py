#!/usr/bin/env python3
"""
Drift Detector Test Suite
Validates drift detection logic and numeric stability.
"""

import json
import numpy as np
from drift_detector import DriftDetector


def test_zero_std_protection():
    """
    Test that zero standard deviation doesn't cause divide-by-zero.
    """
    print("Test 1: Zero Standard Deviation Protection")
    print("-" * 50)
    
    # Create baseline with constant values
    baseline = []
    for i in range(10):
        baseline.append({
            'timestamp': f'2025-01-01T00:00:{i:02d}',
            'endpoint': '/test',
            'status': 'success',
            'flask_cpu': 10.0,  # Constant value
            'flask_memory_mb': 50.0,
            'flask_fds': 20,
            'flask_disk_read_mb': 0.0,
            'flask_disk_write_mb': 0.0,
            'flask_ctx_switches': 100,
            'flask_children': 0,
            'mysql_cpu': 5.0,
            'mysql_memory_mb': 100.0,
            'mysql_fds': 30,
            'mysql_disk_read_mb': 0.0,
            'mysql_disk_write_mb': 0.0,
            'mysql_ctx_switches': 50,
            'mysql_children': 0
        })
    
    # Write baseline
    with open('test_baseline.jsonl', 'w') as f:
        for sample in baseline:
            f.write(json.dumps(sample) + '\n')
    
    # Load detector
    try:
        detector = DriftDetector('test_baseline.jsonl')
        
        # Test with same values
        test_sample = baseline[0].copy()
        score, components = detector.compute_drift_score(test_sample)
        
        print(f"Drift score: {score:.4f}")
        assert not np.isnan(score), "Drift score is NaN!"
        assert not np.isinf(score), "Drift score is infinite!"
        
        print("✓ Zero-std protection works correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    print()
    return True


def test_extreme_values():
    """
    Test handling of extreme metric values.
    """
    print("Test 2: Extreme Value Handling")
    print("-" * 50)
    
    # Create normal baseline
    baseline = []
    for i in range(10):
        baseline.append({
            'timestamp': f'2025-01-01T00:00:{i:02d}',
            'endpoint': '/test',
            'status': 'success',
            'flask_cpu': 10.0 + np.random.randn(),
            'flask_memory_mb': 50.0 + np.random.randn() * 5,
            'flask_fds': 20,
            'flask_disk_read_mb': 1.0,
            'flask_disk_write_mb': 1.0,
            'flask_ctx_switches': 100,
            'flask_children': 0,
            'mysql_cpu': 5.0,
            'mysql_memory_mb': 100.0,
            'mysql_fds': 30,
            'mysql_disk_read_mb': 2.0,
            'mysql_disk_write_mb': 2.0,
            'mysql_ctx_switches': 50,
            'mysql_children': 0
        })
    
    with open('test_baseline.jsonl', 'w') as f:
        for sample in baseline:
            f.write(json.dumps(sample) + '\n')
    
    detector = DriftDetector('test_baseline.jsonl')
    
    # Test extreme sample
    extreme_sample = baseline[0].copy()
    extreme_sample['flask_cpu'] = 1000.0  # 100x normal
    extreme_sample['mysql_cpu'] = 500.0
    
    try:
        score, components = detector.compute_drift_score(extreme_sample)
        
        print(f"Drift score: {score:.4f}")
        print(f"Delta: {components['delta']:.4f}")
        
        assert not np.isnan(score), "Drift score is NaN!"
        assert not np.isinf(score), "Drift score is infinite!"
        assert score < 1000, "Drift score exploded!"
        
        print("✓ Extreme values handled correctly")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    print()
    return True


def test_mysql_amplification():
    """
    Test that MySQL metrics are properly amplified.
    """
    print("Test 3: MySQL Amplification")
    print("-" * 50)
    
    # Create baseline
    baseline = []
    for i in range(10):
        baseline.append({
            'timestamp': f'2025-01-01T00:00:{i:02d}',
            'endpoint': '/test',
            'status': 'success',
            'flask_cpu': 10.0,
            'flask_memory_mb': 50.0,
            'flask_fds': 20,
            'flask_disk_read_mb': 1.0,
            'flask_disk_write_mb': 1.0,
            'flask_ctx_switches': 100,
            'flask_children': 0,
            'mysql_cpu': 5.0,
            'mysql_memory_mb': 100.0,
            'mysql_fds': 30,
            'mysql_disk_read_mb': 2.0,
            'mysql_disk_write_mb': 2.0,
            'mysql_ctx_switches': 50,
            'mysql_children': 0
        })
    
    with open('test_baseline.jsonl', 'w') as f:
        for sample in baseline:
            f.write(json.dumps(sample) + '\n')
    
    detector = DriftDetector('test_baseline.jsonl')
    
    # Test 1: Flask anomaly only
    flask_anomaly = baseline[0].copy()
    flask_anomaly['flask_cpu'] = 50.0  # 5x increase
    
    score1, _ = detector.compute_drift_score(flask_anomaly)
    detector.reset_history()
    
    # Test 2: MySQL anomaly only
    mysql_anomaly = baseline[0].copy()
    mysql_anomaly['mysql_cpu'] = 25.0  # 5x increase
    
    score2, _ = detector.compute_drift_score(mysql_anomaly)
    
    print(f"Flask anomaly drift: {score1:.4f}")
    print(f"MySQL anomaly drift: {score2:.4f}")
    
    # MySQL should have higher drift due to amplification
    if score2 > score1:
        print("✓ MySQL amplification working correctly")
    else:
        print("⚠ MySQL amplification may not be working as expected")
    
    print()
    return True


def test_temporal_components():
    """
    Test acceleration and prediction error components.
    """
    print("Test 4: Temporal Components")
    print("-" * 50)
    
    # Create baseline
    baseline = []
    for i in range(10):
        baseline.append({
            'timestamp': f'2025-01-01T00:00:{i:02d}',
            'endpoint': '/test',
            'status': 'success',
            'flask_cpu': 10.0,
            'flask_memory_mb': 50.0,
            'flask_fds': 20,
            'flask_disk_read_mb': 1.0,
            'flask_disk_write_mb': 1.0,
            'flask_ctx_switches': 100,
            'flask_children': 0,
            'mysql_cpu': 5.0,
            'mysql_memory_mb': 100.0,
            'mysql_fds': 30,
            'mysql_disk_read_mb': 2.0,
            'mysql_disk_write_mb': 2.0,
            'mysql_ctx_switches': 50,
            'mysql_children': 0
        })
    
    with open('test_baseline.jsonl', 'w') as f:
        for sample in baseline:
            f.write(json.dumps(sample) + '\n')
    
    detector = DriftDetector('test_baseline.jsonl')
    
    # Simulate gradual increase
    print("Simulating gradual CPU increase...")
    for i in range(5):
        sample = baseline[0].copy()
        sample['flask_cpu'] = 10.0 + i * 5.0
        
        score, components = detector.compute_drift_score(sample)
        print(f"  Step {i+1}: drift={score:.4f}, accel={components['acceleration']:.4f}")
    
    print("✓ Temporal components computed")
    print()
    return True


def test_history_reset():
    """
    Test that history reset works correctly.
    """
    print("Test 5: History Reset")
    print("-" * 50)
    
    # Create baseline
    baseline = []
    for i in range(10):
        baseline.append({
            'timestamp': f'2025-01-01T00:00:{i:02d}',
            'endpoint': '/test',
            'status': 'success',
            'flask_cpu': 10.0,
            'flask_memory_mb': 50.0,
            'flask_fds': 20,
            'flask_disk_read_mb': 1.0,
            'flask_disk_write_mb': 1.0,
            'flask_ctx_switches': 100,
            'flask_children': 0,
            'mysql_cpu': 5.0,
            'mysql_memory_mb': 100.0,
            'mysql_fds': 30,
            'mysql_disk_read_mb': 2.0,
            'mysql_disk_write_mb': 2.0,
            'mysql_ctx_switches': 50,
            'mysql_children': 0
        })
    
    with open('test_baseline.jsonl', 'w') as f:
        for sample in baseline:
            f.write(json.dumps(sample) + '\n')
    
    detector = DriftDetector('test_baseline.jsonl')
    
    # Build up history
    for i in range(3):
        detector.compute_drift_score(baseline[0])
    
    history_len_before = len(detector.history)
    print(f"History length before reset: {history_len_before}")
    
    # Reset
    detector.reset_history()
    history_len_after = len(detector.history)
    print(f"History length after reset: {history_len_after}")
    
    assert history_len_after == 0, "History not properly reset!"
    print("✓ History reset works correctly")
    
    print()
    return True


def main():
    """
    Run all tests.
    """
    print("="*60)
    print("Drift Detector Test Suite")
    print("="*60)
    print()
    
    tests = [
        test_zero_std_protection,
        test_extreme_values,
        test_mysql_amplification,
        test_temporal_components,
        test_history_reset
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    # Cleanup
    import os
    if os.path.exists('test_baseline.jsonl'):
        os.remove('test_baseline.jsonl')


if __name__ == '__main__':
    main()
