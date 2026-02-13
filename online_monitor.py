#!/usr/bin/env python3
"""
Real-Time IDS Monitor (SRBH-20 Integrated)
Streams telemetry and detects attacks using drift scoring
and SRBH-trained thresholds.

Features:
- True tail mode (no rereading)
- Batch analysis mode
- Two-tier alerting (WARNING / ALERT)
- Cooldown protection
- Drift logging to CSV
- Uses SRBH-trained thresholds if provided
"""

import time
import json
import csv
import os
from datetime import datetime
from drift_detector import DriftDetector


class OnlineMonitor:
    def __init__(
        self,
        telemetry_file="telemetry.jsonl",
        baseline_file="normal_intent.jsonl",
        drift_log="drift_log.csv",
        trained_threshold_file="srbh_training/trained_thresholds.json",
        warning_threshold=1.5,
        alert_threshold=2.5,
        cooldown_seconds=3
    ):
        self.telemetry_file = telemetry_file
        self.drift_log = drift_log
        self.cooldown_seconds = cooldown_seconds

        # Load SRBH-trained thresholds if available
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold

        if trained_threshold_file and os.path.exists(trained_threshold_file):
            try:
                with open(trained_threshold_file, "r") as f:
                    trained = json.load(f)

                if "warning" in trained and "alert" in trained:
                    self.warning_threshold = trained["warning"]
                    self.alert_threshold = trained["alert"]

                    print("\n==============================")
                    print("IDS MODE: SRBH20_TRAINED")
                    print("==============================")
                else:
                    print("\nWARNING: trained_thresholds.json missing keys")
                    print("Falling back to manual thresholds")

            except Exception as e:
                print("\nWARNING: Could not load trained thresholds")
                print("Reason:", e)
                print("Falling back to manual thresholds")
        else:
            print("\nINFO: No trained thresholds found")
            print("Using manual thresholds")

        print(f"WARNING Threshold: {self.warning_threshold:.4f}")
        print(f"ALERT Threshold:   {self.alert_threshold:.4f}")
        print("==============================\n")

        # Initialize drift detector
        self.detector = DriftDetector(baseline_file)

        # State
        self.last_alert_time = 0
        self.sample_count = 0

        # Initialize CSV drift log
        self._init_drift_log()

    def _init_drift_log(self):
        if not os.path.exists(self.drift_log):
            with open(self.drift_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "endpoint",
                    "drift_score",
                    "delta",
                    "acceleration",
                    "prediction_error",
                    "alert_level"
                ])

    def _log_drift(self, sample, drift_score, components, alert_level):
        try:
            with open(self.drift_log, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    sample.get("timestamp", datetime.utcnow().isoformat()),
                    sample.get("endpoint", "unknown"),
                    f"{drift_score:.4f}",
                    f"{components['delta']:.4f}",
                    f"{components['acceleration']:.4f}",
                    f"{components['prediction_error']:.4f}",
                    alert_level
                ])
        except Exception:
            pass

    def _check_cooldown(self):
        return (time.time() - self.last_alert_time) < self.cooldown_seconds

    def _process_sample(self, sample):
        self.sample_count += 1

        drift_score, components = self.detector.compute_drift_score(sample)

        alert_level = "NORMAL"
        if drift_score >= self.alert_threshold:
            alert_level = "ALERT"
        elif drift_score >= self.warning_threshold:
            alert_level = "WARNING"

        self._log_drift(sample, drift_score, components, alert_level)

        if alert_level in ["WARNING", "ALERT"]:
            if not self._check_cooldown():
                self._emit_alert(sample, drift_score, components, alert_level)

                if alert_level == "ALERT":
                    self.detector.reset_history()
                    self.last_alert_time = time.time()

    def _emit_alert(self, sample, drift_score, components, alert_level):
        timestamp = sample.get("timestamp", "unknown")
        endpoint = sample.get("endpoint", "unknown")

        print("\n" + "=" * 60)
        print(f"[{alert_level}] Anomaly Detected!")
        print("=" * 60)
        print(f"Timestamp: {timestamp}")
        print(f"Endpoint: {endpoint}")
        print(f"Drift Score: {drift_score:.4f}")
        print(f"  - Delta: {components['delta']:.4f}")
        print(f"  - Acceleration: {components['acceleration']:.4f}")
        print(f"  - Prediction Error: {components['prediction_error']:.4f}")

        contributions = self.detector.get_feature_contributions(sample)
        print("\nTop Feature Contributions:")
        for i, (feature, contrib) in enumerate(list(contributions.items())[:5]):
            print(f"  {i+1}. {feature}: {contrib:.4f}")

        print("=" * 60 + "\n")

    def tail_file(self):
        print("Starting Online IDS Monitor...")
        print("Monitoring:", self.telemetry_file)
        print("Press Ctrl+C to stop\n")

        try:
            with open(self.telemetry_file, "r") as f:
                f.seek(0, 2)  # move to EOF

                while True:
                    line = f.readline()

                    if not line:
                        time.sleep(0.1)
                        continue

                    try:
                        sample = json.loads(line.strip())
                        self._process_sample(sample)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print("Warning: Error processing sample:", e)
                        continue

        except FileNotFoundError:
            print("ERROR:", self.telemetry_file, "not found!")
        except KeyboardInterrupt:
            print("\nMonitor stopped")
            print("Samples processed:", self.sample_count)

    def batch_analyze(self):
        print("Running batch analysis...")

        try:
            with open(self.telemetry_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        sample = json.loads(line)
                        self._process_sample(sample)
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue

            print("Batch complete:", self.sample_count, "samples")

        except FileNotFoundError:
            print("ERROR:", self.telemetry_file, "not found!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time IDS Monitor")

    parser.add_argument(
        "--mode",
        choices=["stream", "batch"],
        default="stream",
        help="Run mode: stream (live) or batch (offline)"
    )

    parser.add_argument(
        "--telemetry",
        default="telemetry.jsonl",
        help="Telemetry JSONL file"
    )

    parser.add_argument(
        "--baseline",
        default="normal_intent.jsonl",
        help="Baseline file"
    )

    parser.add_argument(
        "--trained",
        default="srbh_training/trained_thresholds.json",
        help="Path to SRBH-trained thresholds JSON"
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=3,
        help="Cooldown period in seconds after alert"
    )

    args = parser.parse_args()

    monitor = OnlineMonitor(
        telemetry_file=args.telemetry,
        baseline_file=args.baseline,
        trained_threshold_file=args.trained,
        cooldown_seconds=args.cooldown
    )

    if args.mode == "stream":
        monitor.tail_file()
    else:
        monitor.batch_analyze()


if __name__ == "__main__":
    main()
