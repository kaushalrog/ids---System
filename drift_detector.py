#!/usr/bin/env python3
import json
import numpy as np
import sys


class DriftDetector:
    def __init__(self, baseline_file="normal_intent.jsonl"):
        self.baseline_mean = {}
        self.baseline_std = {}
        self.feature_names = []
        self.history = []
        self.max_history = 3
        self.eps = 1e-6

        # Adaptive weighting based on feature importance
        self.weight_delta = 0.6
        self.weight_accel = 0.2
        self.weight_error = 0.2
        self.mysql_amplifier = 2.0  # Reduced from 3.0 to lower false positives

        # Tracking for validation
        self.validation_mode = False
        self.min_baseline_samples = 10
        
        self._load_baseline(baseline_file)

    def _load_baseline(self, baseline_file):
        samples = []

        try:
            with open(baseline_file, "r") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except FileNotFoundError:
            print("ERROR: Baseline file not found:", baseline_file)
            sys.exit(1)

        if not samples:
            print("ERROR: Baseline file empty!")
            sys.exit(1)

        # Validate minimum baseline samples for statistical reliability
        if len(samples) < self.min_baseline_samples:
            print(f"WARNING: Only {len(samples)} baseline samples (recommend {self.min_baseline_samples}+)")

        self.feature_names = [
            k for k in samples[0].keys()
            if k not in ["timestamp", "endpoint", "status"]
        ]

        if not self.feature_names:
            print("ERROR: No features found in baseline data!")
            sys.exit(1)

        for feature in self.feature_names:
            values = [s.get(feature, 0.0) for s in samples]
            
            # Filter out invalid values
            valid_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            
            if not valid_values:
                print(f"WARNING: Feature '{feature}' has no valid values, using default")
                self.baseline_mean[feature] = 0.0
                self.baseline_std[feature] = self.eps
                continue

            mean = np.mean(valid_values)
            std = np.std(valid_values)

            if std < self.eps:
                std = self.eps

            self.baseline_mean[feature] = mean
            self.baseline_std[feature] = std

        print(f"✓ Baseline loaded: {len(samples)} samples, {len(self.feature_names)} features")

    def _zscore(self, value, mean, std):
        z = (value - mean) / max(std, self.eps)
        return np.clip(z, -12.0, 12.0)

    def _delta_component(self, sample):
        deviations = []

        for feature in self.feature_names:
            value = sample.get(feature, 0.0)
            mean = self.baseline_mean[feature]
            std = self.baseline_std[feature]

            z = self._zscore(value, mean, std)

            if "mysql" in feature:
                z *= self.mysql_amplifier

            deviations.append(abs(z))

        return np.sqrt(np.mean(np.array(deviations) ** 2))

    def _acceleration_component(self, delta):
        if len(self.history) < 2:
            return 0.0

        velocities = [self.history[i] - self.history[i - 1]
                      for i in range(1, len(self.history))]

        mean_vel = np.mean(velocities)
        accel = abs((delta - self.history[-1]) - mean_vel)

        return accel

    def _prediction_error(self, delta):
        if not self.history:
            return 0.0

        predicted = np.mean(self.history)
        return abs(delta - predicted)

    def compute_drift_score(self, sample):
        """Calculate drift score with improved numerical stability"""
        # Validate input
        if not sample or not isinstance(sample, dict):
            return 0.0, {"error": "invalid_sample"}

        delta = self._delta_component(sample)
        accel = self._acceleration_component(delta)
        pred_error = self._prediction_error(delta)

        # Ensure all components are valid numbers
        if any(np.isnan([delta, accel, pred_error])):
            return 0.0, {"error": "nan_component"}

        drift_score = (
            self.weight_delta * delta +
            self.weight_accel * accel +
            self.weight_error * pred_error
        )

        # Bound drift score to prevent extreme values
        drift_score = float(np.clip(drift_score, 0.0, 10.0))

        self.history.append(delta)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        components = {
            "delta": float(delta),
            "acceleration": float(accel),
            "prediction_error": float(pred_error),
            "drift_score": drift_score
        }

        return drift_score, components

    def reset_history(self):
        self.history = []

    def get_feature_contributions(self, sample):
        contributions = {}

        for feature in self.feature_names:
            value = sample.get(feature, 0.0)
            mean = self.baseline_mean[feature]
            std = self.baseline_std[feature]

            z = self._zscore(value, mean, std)

            if "mysql" in feature:
                z *= self.mysql_amplifier

            contributions[feature] = abs(z)

        return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
