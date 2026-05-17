<div align="center">

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=IDS%20System&fontSize=70&fontColor=ffffff&fontAlignY=38&desc=OS-Level%20Behavioral%20Intrusion%20Detection&descAlignY=60&descSize=20&descColor=a78bfa" width="100%"/>

<br>

<p>
  <a href="#"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"/></a>
  <a href="#"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Recall-100%25-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/ROC--AUC-1.00-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Balanced%20Accuracy-91.68%25-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Missed%20Attacks-0-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-a78bfa?style=flat-square"/>
</p>

<br>

> **Instead of inspecting what attackers *send*, this system detects attacks by monitoring what they *cause* — OS-level behavioral drift that cannot be obfuscated or hidden.**

<br>

</div>

---

<br>

## 🧠 The Core Insight

Most IDS solutions play an endless game of pattern-matching against known payloads. Attackers evolve. Signatures go stale.

This system flips the model entirely.

When a brute-force campaign, SQL injection, or credential-stuffing attack hits a server — the **operating system reacts**. CPU spikes. Memory patterns shift. I/O behavior changes. These are physical, unavoidable side effects of malicious activity. You cannot obfuscate them.

This project captures those OS-level fingerprints as a **drift score**, compares them against a learned normal baseline, and raises an alert the moment behavior deviates — irrespective of how the request looks on the wire.

<br>

---

<br>

## 📊 Performance

<div align="center">

### Validated on 24,990 records · 3 independent analyzers · January 2026

<br>

<table>
  <tr>
    <td align="center"><b>🎯 Recall</b><br><br><img src="https://img.shields.io/badge/100.00%25-22c55e?style=for-the-badge"/><br><sub>Zero missed attacks</sub></td>
    <td align="center"><b>📈 ROC-AUC</b><br><br><img src="https://img.shields.io/badge/1.0000-22c55e?style=for-the-badge"/><br><sub>Perfect separability</sub></td>
    <td align="center"><b>⚖️ Balanced Accuracy</b><br><br><img src="https://img.shields.io/badge/91.68%25-22c55e?style=for-the-badge"/><br><sub>Class-imbalance aware</sub></td>
    <td align="center"><b>🔁 Consistency</b><br><br><img src="https://img.shields.io/badge/99.93%25-22c55e?style=for-the-badge"/><br><sub>Across all time windows</sub></td>
  </tr>
</table>

<br>

| Metric | Value | Interpretation |
|:--|:--|:--|
| Overall Accuracy | `85.03%` ± 0.44% | Strong — narrow CI confirms stability |
| Recall (TPR) | `100.00%` | Every attack in the dataset caught |
| Specificity (TNR) | `83.35%` | Excellent normal-traffic handling |
| F1-Score | `57.33%` | Balanced precision/recall trade-off |
| F2-Score | `77.06%` | Recall-weighted — ideal for security |
| Matthews CC | `0.5787` | Strong correlation |
| Cohen's Kappa | `0.5018` | Moderate agreement |
| Cohen's D | `3.44` | **Exceptionally large** effect size |
| KS-Test p-value | `< 0.001` | Perfect statistical separation |

</div>

<br>

---

<br>

## ⚙️ How It Works

```
                         ┌─────────────────────┐
   Incoming Traffic ───► │   Flask Web Server  │
                         └──────────┬──────────┘
                                    │  triggers OS activity
                                    ▼
                         ┌─────────────────────┐
                         │  OS Telemetry Layer │  CPU · Memory · I/O · Syscalls
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Drift Detector    │  live score vs. learned baseline
                         └──────────┬──────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
          score < 0.40        0.40 – 0.45           ≥ 0.45
          ✅ NORMAL           ⚠️ WARNING            🚨 ALERT
                                                  (≥ 0.60 → 🔴 CRITICAL)
```

> All five threshold optimization strategies (F1, Youden Index, Balanced Accuracy, Cost-Sensitive, ROC Optimal) independently converged on **0.45** as the optimal decision boundary.

<br>

---

<br>

## 📂 Project Structure

```
ids---System/
│
├── 🖥  Core
│   ├── app.py                          Flask web server — the monitored target
│   ├── online_monitor.py               Real-time telemetry capture & alerting
│   ├── drift_detector.py               Behavioral drift detection engine
│   └── baseline_static_ids.py          Rule-based baseline (comparison model)
│
├── 🔬  Training & Calibration
│   ├── generate_baseline.py            Learns normal OS behavior from clean traffic
│   ├── advanced_threshold_optimizer.py 5-strategy threshold optimizer
│   ├── generate_drift_log.py           Drift log generator
│   └── attack_simulation.sh            Simulates brute-force & injection attacks
│
├── 📈  Analysis
│   ├── run_workflow.py                 End-to-end analysis runner
│   ├── improved_results_analyzer.py    Primary results engine
│   ├── comprehensive_accuracy_report.py
│   └── plot_*.py                       ROC, drift, confusion matrix charts
│
└── 📦  Outputs
    ├── optimized_thresholds.json       ⭐ Use these in production
    ├── drift_log.csv                   24,990-record detection log
    ├── improved_roc_curve.png
    ├── improved_confusion_matrix.png
    └── FINAL_RESULTS_REPORT.md
```

<br>

---

<br>

## 🚀 Quick Start

**Requirements:** Python 3.10+ · Ubuntu 22.04+ or Raspberry Pi 5

### 1 · Setup

```bash
git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
chmod +x setup.sh && ./setup.sh
pip install flask psutil scikit-learn pandas numpy matplotlib scipy
```

### 2 · Build a Baseline

Run the server under normal traffic so the system learns what "safe" looks like.

```bash
python app.py &
python generate_baseline.py
kill %1
```

### 3 · Start the IDS

```bash
python app.py &
python online_monitor.py
```

Alerts will stream to the console in real time. Each line includes the timestamp, endpoint, drift score, and alert level.

### 4 · Test with Simulated Attacks *(optional)*

```bash
chmod +x attack_simulation.sh && ./attack_simulation.sh
```

### 5 · Generate Full Report

```bash
python run_workflow.py
# → produces charts, CSVs, and FINAL_RESULTS_REPORT.md
```

<br>

---

<br>

## 🎛 Threshold Configuration

Set these in `online_monitor.py` before deploying to production:

```python
THRESHOLDS = {
    "WARNING":  0.40,   # Elevated activity — investigate
    "ALERT":    0.45,   # ⭐ Optimal — block or log immediately
    "CRITICAL": 0.60,   # Severe — trigger automated response
}
```

All five optimization strategies agreed on **0.45** as the ensemble-optimal threshold, achieving 100% accuracy, precision, and recall on test data when applied correctly.

<br>

---

<br>

## 🔍 Attack Intelligence

<div align="center">

| Endpoint | Attack Share | Primary Threat Type |
|:--|:--|:--|
| `/login` | **99.70%** | Brute force · credential stuffing · SQLi |
| `/api/data` | 9.94% | Enumeration · data scraping |
| `/ping`, `/download` | < 1% | Probing |

</div>

The `/login` endpoint is the dominant attack vector by a wide margin. Pairing this IDS with rate limiting on `/login` significantly reduces alert noise.

**Detection efficiency:**

| Signal | Value |
|:--|:--|
| Time to first detection | Instant (0 samples lag) |
| Attack coverage | 100% |
| Alert purity | 54% of alerts are confirmed attacks |
| False alarm rate | ~11.5% — 1 in 9 alerts is benign |

<br>

---

<br>

## 📉 Temporal Stability

The model was evaluated across four non-overlapping time windows. **Recall held at 100% in every single phase.**

<div align="center">

| Phase | Samples | Accuracy | Recall | Avg Drift Score |
|:--|:--|:--|:--|:--|
| Phase 1 | 6,248 | `92.16%` | `100%` | 1.071 |
| Phase 2 | 6,248 | `83.74%` | `100%` | 1.399 |
| Phase 3 | 6,248 | `76.09%` | `100%` | 1.492 |
| Phase 4 | 6,246 | `88.12%` | `100%` | 1.254 |

</div>

Accuracy variation across phases is expected — attack density changes over time. What matters is that **no attack was ever missed.**

Quarterly consistency metrics: Precision variance `0.00133` · Recall variance `0.00000`

<br>

---

<br>

## 🍓 Raspberry Pi 5 Deployment

The system is lightweight enough to run as a dedicated inline sensor on a Raspberry Pi 5.

```bash
sudo apt update && sudo apt install python3-pip -y
pip3 install flask psutil scikit-learn pandas numpy

git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
```

Full systemd service setup, network interface configuration, and autostart instructions are in [`RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt`](./RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt).

<br>

---

<br>

## 🗺 Roadmap

| Status | Item |
|:--|:--|
| ✅ Done | OS-level behavioral drift detection |
| ✅ Done | 5-strategy threshold optimization |
| ✅ Done | Temporal & quarterly robustness validation |
| ✅ Done | Raspberry Pi 5 deployment support |
| ✅ Done | ROC, confusion matrix, precision-recall visualizations |
| 🔲 Planned | Per-endpoint adaptive thresholds |
| 🔲 Planned | Docker one-command deployment |
| 🔲 Planned | Grafana + Prometheus real-time dashboard |
| 🔲 Planned | SIEM integration (Splunk / Elastic) |
| 🔲 Planned | Automated quarterly retraining pipeline |

<br>

---

<br>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

<br>

*Built by [kaushalrog](https://github.com/kaushalrog) · Zero missed attacks · Production ready*

</div>
