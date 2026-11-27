# Data Pipeline Failure Prediction System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8-017CEE.svg)](https://airflow.apache.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C.svg)](https://spark.apache.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-pipeline-failure-prediction-system-6pzkxs3w9kjrhqqkyuiz3o.streamlit.app/)
[![Academic Report](https://img.shields.io/badge/Technical%20Report-20%20Pages%20PDF-success)](docs/Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Pre-Failure Early Warning & Root Cause Attribution for Distributed Data Pipelines using Dual-Head Temporal Attention Gated Recurrent Units (AttentionGRU).**

---

### Project Links & Deliverables

* **Live Interactive Web Application:** [https://data-pipeline-failure-prediction-system-6pzkxs3w9kjrhqqkyuiz3o.streamlit.app/](https://data-pipeline-failure-prediction-system-6pzkxs3w9kjrhqqkyuiz3o.streamlit.app/)
* **Academic Technical Report (20-Page PDF):** [`docs/Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf`](docs/Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf)
* **ReportLab Generator Script:** [`docs/generate_full_technical_report.py`](docs/generate_full_technical_report.py)
* **Course:** Advanced Neural Network Architectures (Course Code: `2307411T` — Track 2: Sequential Models)
* **Institution:** Department of Electronics and Telecommunication Engineering, MIT Academy of Engineering, Pune (MITAOE)
* **Guide:** Dr. Smita Kulkarni
* **Group 12 Members:**
  1. **Ayush Nagras** (PRN: 202301070001)
  2. **Siddhesh Bhivare** (PRN: 202301070005)
  3. **Om Nikam** (PRN: 202301070006)
  4. **Vaibhav Tajane** (PRN: 202301070008)

---

## 1. Executive Summary

In enterprise data platforms, automated Extract-Transform-Load (ETL) pipelines orchestrate thousands of scheduled DAGs across **Apache Airflow**, **Apache Spark**, **Amazon S3**, and **PostgreSQL**. 

### The Problem: Reactive vs. Proactive Observability
Conventional monitoring solutions (AWS CloudWatch, Datadog, Prometheus) rely on **static univariate threshold alerts** (e.g., `RAM > 85%`) or post-mortem stack trace analysis. They alert on-call data engineers only **after** a mission-critical pipeline crashes, causing broken analytical dashboards, stale machine learning feature stores, and SLA violation penalties.

### The Solution
The **Data Pipeline Failure Prediction System** shifts reliability left by analyzing multi-day sequential telemetry sliding windows $[t-3, t-2, t-1]$ **prior to DAG execution**. The system concurrently:
1. Forecasts upcoming next-run **failure risk probability** ($0.0\%$ to $100.0\%$).
2. Localizes the probable **root cause subsystem** across 5 pipeline stages.
3. Provides **Explainable AI (XAI)** temporal attention weights $[\alpha_{t-3}, \alpha_{t-2}, \alpha_{t-1}]$ identifying which historical run drove the alert.
4. Generates **dynamic engineering remediation guidance** with actionable code and terminal commands.

---

## 2. 4-Stage Enterprise Pipeline Architecture

The system is grounded in an enterprise batch data processing architecture orchestrated by Apache Airflow:

```text
[ Stage 1: REST API Ingestion ]
           │  producer_users_daily.py (HTTP GET, Rate Limiting, Retries)
           ▼
[ Stage 2: Amazon S3 Data Lake ]
           │  s3a://datalake/users_daily/ Partition Vault (S3KeySensor)
           ▼
[ Stage 3: PySpark Distributed Audit ]
           │  consumer_users_daily.py (StructType Schema Validation & Null Checks)
           ▼
[ Stage 4: PostgreSQL Warehouse Load ]
              users_daily Table (:5432) with ACID DELETE-and-INSERT Idempotency
```

### Stage Details & Failure Modes
* **Stage 1 (REST API Ingestion):** Ingests raw customer user payloads. Detected failure modes: `HTTP 500 Internal Server Error`, `HTTP 429 Rate Limit`, socket `ReadTimeout`.
* **Stage 2 (Amazon S3 Data Lake):** Persists JSON payloads into date-partitioned vaults. Detected failure modes: missing partition keys, `IAM 403 Access Denied`, 0-byte corrupt payloads.
* **Stage 3 (PySpark Quality Audit):** Executes distributed schema enforcement and data quality checks. Detected failure modes: `java.lang.OutOfMemoryError: Java heap space`, schema drift, null primary key violations.
* **Stage 4 (PostgreSQL Warehouse Load):** Loads verified records into enterprise analytical marts. Detected failure modes: connection pool exhaustion, database port refused, table deadlocks.
  * **ACID Transaction Idempotency:** Implements a transactional `DELETE FROM users_daily WHERE dt = 'today'` before insert. If an execution terminates abruptly or retries 10 times, the warehouse table remains 100% consistent with zero duplicate rows.

---

## 3. Dataset & Preprocessing

The system operates on an empirical dataset spanning **450 continuous Airflow pipeline executions across 15 calendar months**:

* **Baseline Failure Rate:** ~10.2% (46 failures out of 450 runs), accurately reflecting production enterprise data environments where catastrophic failures are sparse but critical.
* **Relational Schema Integration (`log_id`):**
  * `pipeline_telemetry.csv` (Quantitative Modality): 450 rows capturing 7 numerical metrics: `duration` (s), `records_processed`, `retry_count`, `memory_mb`, `cpu_usage_pct`, `upstream_latency_ms`, and `error_rate`.
  * `pipeline_logs.json` (Qualitative Modality): 450 rich Airflow execution logs keyed by `log_id`, storing executor environment variables, stdout/stderr streams, and full Java/Python stack traces across 14 failure classes.

### Zero-Leakage Preprocessing Guarantee
* **Sliding Window Horizon ($T = 3$):** Formats telemetry into 3-day sequential observation tensors: $[x_{t-3}, x_{t-2}, x_{t-1}] \in \mathbb{R}^{3 \times 4}$.
* **Chronological Partitions:** Split into 70% train (runs 1–315), 15% validation (runs 316–382), and 15% test (runs 383–450) without random shuffling.
* **Feature Normalization:** `StandardScaler` parameters ($\mu, \sigma$) are fitted **strictly on the training partition**, guaranteeing zero forward-looking bias.

---

## 4. Deep Learning Model Architecture: AttentionGRU

The core neural model is an **AttentionGRU** combining recurrent gating, additive temporal attention, and multi-task prediction heads:

```text
Input Sequence: [x_(t-3), x_(t-2), x_(t-1)]  ∈ R^(3 x 4)
                        │
                        ▼
         Gated Recurrent Unit (GRU, D=24)
         Hidden states: [h_1, h_2, h_3]
                        │
                        ▼
    Bahdanau Additive Temporal Attention Layer
    Alignment energies: e_t = v^T tanh(W_a h_t + b_a)
    Attention weights:  [α_(t-3), α_(t-2), α_(t-1)] (sum = 1.0)
    Context vector:     c = Σ α_t h_t  ∈ R^24
                        │
                        ▼
         Feature Fusion: f = Dropout(ReLU(W_f [c; h_3] + b_f))
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [ Head 1: Binary Risk ]       [ Head 2: Subsystem Attribution ]
  P(Fail) = σ(W_risk f + b)     P(Stage) = Softmax(W_stage f + b)
  Trained with BCEWithLogits    Trained with CrossEntropy (5 classes)
```

### Architectural Advantages
* **Parameter Efficiency:** AttentionGRU uses 4,385 parameters compared to 5,825 for AttentionLSTM (a 24.7% reduction), preventing overfitting on short sequences ($T=3$).
* **Ultra-Low Latency:** Inference evaluates in **1.2 ms on a single CPU core** (17.5 KB model size), making it suitable for live Airflow task scheduler callbacks.
* **Composite Multi-Task Loss:** Optimized end-to-end via:
  $$\mathcal{L}_{\text{total}} = 0.5 \, \mathcal{L}_{\text{BCE}}(y_{\text{risk}}, \hat{y}_{\text{risk}}) + 0.5 \, \mathcal{L}_{\text{CE}}(y_{\text{stage}}, \hat{y}_{\text{stage}})$$

---

## 5. Experimental Results & Benchmarks

Evaluated on the unseen chronological test partition (runs 383–450):

| Model Architecture | Parameters | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | CPU Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AttentionGRU (Proposed)** | **4,385** | **0.942** | **0.887** | **89.5%** | **86.4%** | **0.879** | **1.2 ms** |
| AttentionLSTM (Ablation) | 5,825 | 0.931 | 0.874 | 88.1% | 85.2% | 0.866 | 1.8 ms |
| Vanilla GRU (No Attention) | 4,193 | 0.894 | 0.825 | 84.0% | 80.1% | 0.820 | 1.1 ms |
| Random Forest (100 Trees) | N/A | 0.852 | 0.764 | 78.3% | 74.2% | 0.762 | 4.5 ms |
| Static Rule Thresholds | 0 | 0.684 | 0.412 | 52.1% | 48.5% | 0.502 | 0.1 ms |

### Operational Risk Thresholds
In a production platform with a ~10% baseline failure rate, an output probability of **52.4%** represents a **5.2x odds ratio increase**, placing the run in the 99.8th percentile of historical severity. Calibrated alert tiers:
* **Nominal Risk (< 20.0%):** Green badge — normal scheduling proceeds.
* **Elevated Risk (20.0% – 39.9%):** Amber badge — resource utilization and upstream latency flagged.
* **Critical High Risk ($\ge$ 40.0%):** Red badge — actionable engineering remediation triggered.

---

## 6. Live Web Dashboard Modules (Streamlit)

The application is deployed at: [Streamlit Community Cloud](https://data-pipeline-failure-prediction-system-6pzkxs3w9kjrhqqkyuiz3o.streamlit.app/)

1. **Tab 1: Live Risk Monitor (Historical Playback)**
   * Run selector allows operators to replay any historical run.
   * Displays real-time risk gauges, alert badges, and attributed subsystem stage.
   * Renders **Explainable AI Attention Weight charts** across $[t-3, t-2, t-1]$.
   * Renders **Dynamic Engineering Remediation cards** with impact analysis and terminal fix commands.

2. **Tab 2: What-If Simulator, Batch CSV Evaluation & Log Diagnostics**
   * **Reactive Telemetry Sliders:** Simulate failure profiles (`Gradual Memory Leak (Spark OOM)`, `Upstream API Timeout Cascade`, `Nominal Healthy Execution`).
   * **Batch CSV File Evaluation:** Upload custom telemetry files or evaluate built-in samples (`sample_high_risk_failure.csv`).
   * **Airflow Log Text Diagnostics Parser:** 14 regular expression patterns parse raw pasted stack traces to extract root cause, failing stage, and recommended terminal commands.

3. **Tab 3: Airflow Pipeline Inspector (Animated Architecture)**
   * Animated architectural topology showing the 4 pipeline nodes connected by conduits with CSS keyframe glowing data packets (`@keyframes packet-slide`).
   * Interactive fault injection scenarios: API 500 Crash, PySpark Quality Null Breach, PostgreSQL Pool Exhaustion.

---

## 7. Repository Structure

```text
data-pipeline-failure-prediction-system/
├── app.py                             # Streamlit Web Application (3 Production Tabs)
├── requirements.txt                   # Production dependencies (torch, streamlit, scikit-learn)
├── dags/                              # Apache Airflow Production Pipeline DAGs
│   ├── producer_users_daily.py        # Stage 1 REST API -> Stage 2 S3 Lake
│   └── consumer_users_daily.py        # Stage 3 PySpark Audit -> Stage 4 PostgreSQL
├── data/
│   ├── raw/
│   │   ├── pipeline_telemetry.csv     # 450 historical Airflow runs (7 metrics + log_id)
│   │   └── pipeline_logs.json         # 450 raw task execution traces
│   └── samples/                       # Calibrated CSV datasets for examiner evaluation
├── models/
│   └── best_attention_model.pth       # Trained AttentionGRU PyTorch checkpoint
├── src/
│   ├── data/log_analyzer.py           # Natural text log diagnostic parser (14 regex signatures)
│   ├── features/sequence_generator.py # Zero-leakage temporal sliding window engine
│   └── models/attention_model.py      # AttentionGRU & AttentionLSTM dual-head PyTorch models
├── docs/
│   ├── Data_Pipeline_Failure_Prediction_System_Technical_Report.pdf  # 20-Page Academic Report
│   ├── generate_full_technical_report.py                             # ReportLab Generator Script
│   └── figures/                                                      # Publication Figures & Images
├── run_week2.py                       # Standalone reproduction runner for Week 2 (CO3)
├── run_week3.py                       # Standalone reproduction runner for Week 3 (CO4)
├── run_week4.py                       # Standalone reproduction runner for Week 4 (CO5)
└── tests/                             # Automated test suite (pytest)
```

---

## 8. Quickstart & Local Installation

### Prerequisites
* Python 3.10+ (tested on Python 3.11 and 3.12)
* Git

### Installation
```bash
# Clone the repository
git clone https://github.com/AyushNagras-07/data-pipeline-failure-prediction-system.git
cd data-pipeline-failure-prediction-system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```

### Run Standalone Reproduction Scripts
```bash
# Verify sliding window generation & AttentionGRU forward pass (CO3)
python run_week2.py

# Verify multi-task training & hyperparameter ablation sweeps (CO4)
python run_week3.py

# Verify Streamlit components, batch CSV evaluator & metrics (CO5)
python run_week4.py

# Run automated tests
pytest
```

### Recompile the 20-Page Academic Technical Report PDF
```bash
python docs/generate_full_technical_report.py
```

---

## 9. Academic Course Outcome Mapping

This project fulfills all requirements for **Course Code: 2307411T (Advanced Neural Network Architectures — ANNA)**:

| Course Outcome | Description | Project Implementation |
|:---|:---|:---|
| **CO3** | Design and analyze sequential neural networks (RNN, LSTM, GRU). | Custom PyTorch `AttentionGRU` trained from scratch over 60 epochs on 450 sliding window sequences. |
| **CO4** | Implement attention mechanisms for sequential Explainable AI (XAI). | Bahdanau Additive Temporal Attention layer generating dynamic scalar weights $[\alpha_{t-3}, \alpha_{t-2}, \alpha_{t-1}]$. |
| **CO5** | Deploy, evaluate, and operationalize deep learning systems. | Streamlit Community Cloud web app, What-If simulator, batch CSV evaluator, and dynamic engineering remediation engine. |

---

## 10. License

This project is licensed under the [MIT License](LICENSE).
