<div align="center">

<br/>

# 🛡️ IDS — Behavioral Intrusion Detection System

### Detect attacks not by what adversaries *send* — but by what they *cause* at the OS level.

<br/>

[![Python](https://img.shields.io/badge/Python-95.9%25-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Shell](https://img.shields.io/badge/Shell-4.1%25-4EAA25?style=flat-square&logo=gnubash&logoColor=white)](https://gnu.org/software/bash)
[![Platform](https://img.shields.io/badge/Ubuntu%20%2F%20Raspberry%20Pi%205-supported-E95420?style=flat-square&logo=ubuntu&logoColor=white)](https://ubuntu.com)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-2ECC71?style=flat-square)](https://github.com/kaushalrog/ids---System)
[![Recall](https://img.shields.io/badge/Recall-100%25-gold?style=flat-square)](https://github.com/kaushalrog/ids---System)
[![ROC AUC](https://img.shields.io/badge/ROC--AUC-1.0-gold?style=flat-square)](https://github.com/kaushalrog/ids---System)

<br/>

</div>

---

## What Makes This Different

Traditional intrusion detection systems scan HTTP payloads for known attack signatures — a perpetual cat-and-mouse game that sophisticated attackers bypass with obfuscation. This project takes a different path.

**When an attacker hammers your `/login` endpoint, the operating system reacts.** CPU usage spikes, memory access patterns shift, I/O behavior changes. This system learns what "normal" looks like at the OS level, then raises an alert the moment behavior drifts — regardless of how the request was crafted or encoded.

No signatures. No payloads. Just behavioral truth.

---

## Results at a Glance

> Validated on **24,990 records** across a ~2-hour simulated window. Three independent analyzers produced consistent results.

| Metric | Score |
|---|---|
| Recall (Attack Coverage) | **100.00%** — zero missed attacks |
| Balanced Accuracy | **91.68%** |
| Overall Accuracy | **85.03%** |
| ROC-AUC | **1.0** — perfect class separability |
| Consistency Score | **99.93%** across all time windows |
| False Alarm Rate | **11.5%** — ~1 in 9 alerts is benign |

The KS-test returns a separation statistic of 1.0 with p < 0.001 — normal and attack traffic are **statistically non-overlapping** at the OS level. Cohen's D effect size is 3.44, which is classified as exceptionally strong.

---

## Architecture

```
Web Server (Flask)
       │
       ▼
OS Telemetry Collector        ← CPU, memory, I/O, syscall rates
       │
       ▼
Drift Detector                ← compares live metrics to learned baseline
       │
       ├── drift < 0.40  →  ✅  NORMAL
       ├── drift 0.40–0.45  →  ⚠️  WARNING   (investigate)
       ├── drift ≥ 0.45  →  🚨  ALERT      (block / log)
       └── drift ≥ 0.60  →  🔴  CRITICAL   (immediate action)
```

Five independent threshold optimization strategies (F1, Youden Index, Balanced Accuracy, Cost-Sensitive, ROC Optimal) all converged on the same optimal decision boundary: **0.45**.

---

## Project Structure

```
ids---System/
├── app.py                            # Flask web server (monitored target)
├── online_monitor.py                 # Real-time OS telemetry + alerting
├── drift_detector.py                 # Core drift detection engine
├── baseline_static_ids.py            # Rule-based baseline for comparison
├── generate_baseline.py              # Normal-behavior profiler
├── advanced_threshold_optimizer.py   # 5-strategy threshold tuner
├── attack_simulation.sh              # Controlled attack traffic generator
├── setup.sh                          # One-command environment setup
├── run_workflow.py                   # End-to-end analysis runner
│
├── optimized_thresholds.json         # ← deploy with these values
├── drift_log.csv                     # 24,990-record detection log
├── FINAL_RESULTS_REPORT.md           # Full technical report
├── RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt
└── UBUNTU_EXECUTION_GUIDE.txt
```

---

## Getting Started

**Prerequisites:** Python 3.10+, Ubuntu 22.04+ or Raspberry Pi 5

```bash
# 1. Clone and set up
git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
chmod +x setup.sh && ./setup.sh

# 2. Install dependencies
pip install flask psutil scikit-learn pandas numpy matplotlib scipy

# 3. Build a normal-behavior baseline
python app.py &
python generate_baseline.py
kill %1

# 4. Start the IDS
python app.py &
python online_monitor.py

# 5. (Optional) Simulate attacks to verify detection
chmod +x attack_simulation.sh && ./attack_simulation.sh

# 6. Generate full analysis and charts
python run_workflow.py
```

---

## Deployment Configuration

Copy these values into `online_monitor.py` before going to production:

```python
THRESHOLDS = {
    "WARNING":  0.40,   # Elevated — watch closely
    "ALERT":    0.45,   # Optimal threshold — act now
    "CRITICAL": 0.60,   # Severe — trigger automated response
}
```

**Before deploying, verify:**
- [ ] Baseline generated from your own production traffic
- [ ] ROC curve shows clean separation on your dataset
- [ ] Logging enabled for all WARNING-level and above events
- [ ] Alerting pipeline (email / Slack / PagerDuty) connected

---

## Performance Deep Dive

### Temporal Stability

The model was tested across four sequential time windows. Recall stayed at **100%** throughout; accuracy fluctuated naturally as attack density varied.

| Phase | Samples | Accuracy | Recall |
|---|---|---|---|
| Phase 1 | 6,248 | 92.16% | 100% |
| Phase 2 | 6,248 | 83.74% | 100% |
| Phase 3 | 6,248 | 76.09% | 100% |
| Phase 4 | 6,246 | 88.12% | 100% |

### Attack Distribution by Endpoint

| Endpoint | Attack Rate |
|---|---|
| `/login` | **99.70%** of all attacks |
| `/api/data` | 9.94% |
| `/ping`, `/download` | < 1% |

The `/login` endpoint is overwhelmingly the primary attack surface — brute force, credential stuffing, and injection attempts dominate. Rate limiting here is strongly recommended.

### Confusion Matrix

|  | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 18,734 ✅ | 3,742 (false alarms) |
| **Actual Attack** | 0 ✅ | 2,514 ✅ |

Zero false negatives. Every attack in the dataset was caught.

---

## Raspberry Pi 5

This system is lightweight enough to run as a dedicated network sensor on a Raspberry Pi 5.

```bash
sudo apt update && sudo apt install python3-pip python3-venv -y
pip3 install flask psutil scikit-learn pandas numpy
git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
```

See [`RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt`](./RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt) for systemd service setup and network interface configuration.

---

## Roadmap

- [x] OS-level behavioral drift detection
- [x] Multi-strategy threshold optimization
- [x] Raspberry Pi 5 support
- [x] Temporal consistency validation
- [ ] Per-endpoint adaptive thresholds
- [ ] Docker deployment
- [ ] Grafana + Prometheus real-time dashboard
- [ ] SIEM integration (Splunk / Elastic)
- [ ] Automated quarterly retraining pipeline

---

<div align="center">

Made with 🛡️ by [kaushalrog](https://github.com/kaushalrog)

</div>
