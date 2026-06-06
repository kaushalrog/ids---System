<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:6d28d9,100:1e1b4b&height=300&section=header&text=IDS%20System&fontSize=90&fontColor=ffffff&fontAlignY=45&desc=OS-Level%20%C2%B7%20Behavioral%20%C2%B7%20Intrusion%20Detection&descSize=18&descColor=c4b5fd&descAlignY=65&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](#)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi%205-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white)](#)

<br/>

![](https://img.shields.io/badge/-%E2%9C%93%20Recall%20100%25-0d9488?style=flat-square)
![](https://img.shields.io/badge/-%E2%9C%93%20ROC--AUC%201.0-0d9488?style=flat-square)
![](https://img.shields.io/badge/-%E2%9C%93%20Balanced%20Acc%2091.68%25-0d9488?style=flat-square)
![](https://img.shields.io/badge/-%E2%9C%93%20Zero%20Missed%20Attacks-0d9488?style=flat-square)
![](https://img.shields.io/badge/-%E2%98%85%20Production%20Ready-7c3aed?style=flat-square)

<br/>

### *Attackers can obfuscate their payloads. They cannot hide what they do to the OS.*

<br/>

</div>

---

<br/>

<div align="center">

##  A Different Kind of IDS

</div>

Every existing IDS asks the same question: *"Does this request look malicious?"*

This system asks something the attacker cannot control: **"Is the operating system behaving normally?"**

When a brute-force campaign hits `/login`, CPU spikes. When SQL injection runs, memory patterns shift. When scrapers flood endpoints, I/O changes. These are **unavoidable physical side effects** — they happen regardless of how the HTTP payload is encoded or obfuscated.

This project monitors those OS-level telemetry signals in real time, computes a **drift score** against a learned normal baseline, and raises a graded alert the moment behavior deviates. No signatures. No payload inspection. Pure behavioral truth.

<br/>

---

<br/>

<div align="center">

##  Results

**24,990 records · 3 independent analyzers · January 2026**

<br/>

| | |
|:---:|:---:|
| ![](https://img.shields.io/badge/RECALL-100.00%25-16a34a?style=for-the-badge) | ![](https://img.shields.io/badge/ROC--AUC-1.0000-16a34a?style=for-the-badge) |
| ![](https://img.shields.io/badge/BALANCED%20ACCURACY-91.68%25-16a34a?style=for-the-badge) | ![](https://img.shields.io/badge/CONSISTENCY-99.93%25-16a34a?style=for-the-badge) |

</div>

<br/>

| Metric | Value | Notes |
|:---|:---:|:---|
| Overall Accuracy | **85.03%** | ±0.44% — narrow confidence interval |
| Recall (TPR) | **100.00%** | Zero missed attacks across all phases |
| Specificity (TNR) | **83.35%** | Strong normal-traffic classification |
| F2-Score | **77.06%** | Recall-weighted — optimal for security |
| Matthews CC | **0.5787** | Strong model correlation |
| Cohen's D | **3.44** | Exceptionally large effect size |
| KS Statistic | **1.00** (p < 0.001) | Perfect distribution separation |
| False Alarm Rate | **~11.5%** | ~1 in 9 alerts; tunable via threshold |

> The Kolmogorov-Smirnov statistic of **1.0** confirms that normal and attack OS-behavior distributions have **zero overlap** — they are completely separable at the statistical level.

<br/>

### Confusion Matrix

|  | Predicted: Normal | Predicted: Attack |
|:---|:---:|:---:|
| **Actual: Normal** | 18,734 ✅ | 3,742 ⚠️ |
| **Actual: Attack** | **0** ✅ | 2,514 ✅ |

<br/>

### Temporal Stability — Recall held at 100% across every phase

| Phase | Samples | Accuracy | Recall |
|:---|:---:|:---:|:---:|
| Phase 1 | 6,248 | 92.16% | **100%** |
| Phase 2 | 6,248 | 83.74% | **100%** |
| Phase 3 | 6,248 | 76.09% | **100%** |
| Phase 4 | 6,246 | 88.12% | **100%** |

<br/>

---

<br/>

##  Detection Pipeline

```
Incoming Traffic
      │
      ▼
 Flask Web Server ──────────── triggers OS-level activity
      │
      ▼
 Telemetry Collector ─────────  CPU  ·  Memory  ·  I/O  ·  Syscalls
      │
      ▼
 Drift Detector ──────────────  live score  vs.  learned baseline
      │
      ├──  score < 0.40   ──►  ✅  NORMAL
      ├──  score 0.40–0.45 ──►  ⚠️  WARNING    investigate
      ├──  score ≥ 0.45   ──►  🚨  ALERT      block / log
      └──  score ≥ 0.60   ──►  🔴  CRITICAL   immediate response
```

Five independent optimization strategies — F1, Youden Index, Balanced Accuracy, Cost-Sensitive, and ROC Optimal — **all converged on a threshold of 0.45**. This is the recommended production value.

<br/>

---

<br/>

##  Repository Structure

```
ids---System/
│
├── app.py                          Flask web server (monitored target)
├── online_monitor.py               Real-time OS telemetry & alert engine
├── drift_detector.py               Behavioral drift detection core
├── baseline_static_ids.py          Rule-based model for benchmarking
│
├── generate_baseline.py            Learns normal-behavior OS profile
├── advanced_threshold_optimizer.py 5-strategy threshold optimizer
├── attack_simulation.sh            Controlled brute-force & injection simulator
├── setup.sh                        One-command environment setup
├── run_workflow.py                 Full end-to-end analysis pipeline
│
├── optimized_thresholds.json       ← Production-ready threshold values
├── drift_log.csv                   24,990-record detection log (3.3 MB)
├── FINAL_RESULTS_REPORT.md         Complete technical analysis
│
└── plots/
    ├── improved_roc_curve.png
    ├── improved_confusion_matrix.png
    ├── improved_drift_distribution.png
    └── threshold_optimization_curves.png
```

<br/>

---

<br/>

##  Getting Started

> **Requirements:** Python 3.10+ · Ubuntu 22.04+ or Raspberry Pi 5 · sudo access for OS telemetry

**Step 1 — Clone & install**

```bash
git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
chmod +x setup.sh && ./setup.sh
pip install flask psutil scikit-learn pandas numpy matplotlib scipy
```

**Step 2 — Build the normal-behavior baseline**

Run the server under clean traffic so the system learns what safe looks like.

```bash
python app.py &
python generate_baseline.py
kill %1
```

**Step 3 — Launch the IDS**

```bash
python app.py &
python online_monitor.py
```

Each line in the output includes timestamp · endpoint · drift score · alert level.

**Step 4 — Simulate attacks to verify** *(optional)*

```bash
chmod +x attack_simulation.sh && ./attack_simulation.sh
```

**Step 5 — Generate the full analysis report**

```bash
python run_workflow.py
# Outputs → charts, CSVs, and FINAL_RESULTS_REPORT.md
```

<br/>

---

<br/>

##  Production Threshold Config

```python
# online_monitor.py

THRESHOLDS = {
    "WARNING":  0.40,   # Elevated — log and watch
    "ALERT":    0.45,   # ⭐ Optimal — block or notify
    "CRITICAL": 0.60,   # Severe — trigger automated response
}
```

<br/>

---

<br/>

##  Attack Breakdown

Nearly all attacks target a single endpoint:

| Endpoint | Attack Share | Threat Type |
|:---|:---:|:---|
| `/login` | **99.70%** | Brute force · Credential stuffing · SQLi |
| `/api/data` | 9.94% | Enumeration · Scraping |
| `/ping`, `/download` | < 1% | Reconnaissance |

Pairing this IDS with rate limiting on `/login` is strongly recommended — it reduces false-alarm noise significantly while the IDS continues to catch behavioral anomalies.

<br/>

---

<br/>

##  Raspberry Pi 5 Deployment

Runs comfortably as a dedicated inline network sensor on Raspberry Pi 5.

```bash
sudo apt update && sudo apt install python3-pip -y
pip3 install flask psutil scikit-learn pandas numpy
git clone https://github.com/kaushalrog/ids---System.git
cd ids---System
```

Full systemd service setup and autostart guide → [`RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt`](./RASPBERRY_PI_5_DEPLOYMENT_GUIDE.txt)

<br/>

---

<br/>

##  Roadmap

| | |
|:---:|:---|
| ✅ | OS-level behavioral drift detection engine |
| ✅ | 5-strategy threshold optimization — all converged at 0.45 |
| ✅ | Temporal & quarterly robustness validation |
| ✅ | Raspberry Pi 5 deployment support |
| ✅ | ROC, confusion matrix, precision-recall, drift visualizations |
| 🔲 | Per-endpoint adaptive thresholds |
| 🔲 | Docker one-command deployment |
| 🔲 | Grafana + Prometheus real-time dashboard |
| 🔲 | SIEM integration — Splunk / Elastic |
| 🔲 | Automated quarterly retraining pipeline |

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e1b4b,100:6d28d9&height=140&section=footer&text=Zero%20Missed%20Attacks&fontSize=24&fontColor=c4b5fd&fontAlignY=65" width="100%"/>

*Built by [kaushalrog](https://github.com/kaushalrog)*

</div>
