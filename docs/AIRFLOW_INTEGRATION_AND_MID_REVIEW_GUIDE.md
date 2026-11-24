# Data Pipeline Failure Prediction System — Simple Integration & Mid-Review Guide

## Project Title
**Data Pipeline Failure Prediction System**  
*Course:* Advanced Neural Network Architectures (ANNA — Course Code: 2307411T)  
*Track:* Track 2 — Sequential Models / Time-Series (RNN, LSTM, GRU)  
*Existing Pipeline Repository:* [airflow-producer-consumer-etl](https://github.com/AyushNagras-07/airflow-producer-consumer-etl)

---

## 1. Simplified Architecture

The system connects your existing, working Apache Airflow ETL pipeline to sequential deep learning models through one simple, transparent telemetry dataset:

```text
┌────────────────────────────────────────────────────────┐
│             EXISTING AIRFLOW ETL PIPELINE              │
│                                                        │
│  [API] ──► [Producer DAG] ──► [S3 Raw JSON]           │
│                                     │                  │
│                                     ▼                  │
│  [PostgreSQL] ◄── [Parquet S3] ◄── [Consumer PySpark] │
└──────────────────────────┬─────────────────────────────┘
                           │
                           │ Records 1 row per pipeline run
                           ▼
┌────────────────────────────────────────────────────────┐
│           SINGLE TELEMETRY CSV (data/raw/)             │
│                                                        │
│  pipeline_telemetry.csv                                │
│  (run_id, timestamp, duration, records, retries,       │
│   memory_mb, scenario, status)                         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           │ Sliding window of past T runs (e.g., T = 3 or 5)
                           ▼
┌────────────────────────────────────────────────────────┐
│         SEQUENTIAL NEURAL MODELS (src/models/)         │
│                                                        │
│       [Past Runs t-3, t-2, t-1] ──► RNN / LSTM / GRU   │
│                                           │            │
│                                           ▼            │
│                 Failure Probability for Next Run (t)   │
│                 (e.g., Risk: 82% HIGH)                 │
└────────────────────────────────────────────────────────┘
```

---

## 2. Minimal Telemetry Schema (`pipeline_telemetry.csv`)

Instead of multiple complex tables, we store all execution history in **one readable CSV file** in `data/raw/pipeline_telemetry.csv`.

Each row represents **one complete execution run** of the pipeline.

### Schema Fields & Exact Origin

| Column Name | Type | Example | Where Does It Come From? (Exact Origin) |
| :--- | :--- | :--- | :--- |
| `run_id` | String | `run_042` | Airflow DAG run ID (`context['dag_run'].run_id`). |
| `timestamp` | String | `2026-11-01 10:00:00` | The start time of the pipeline execution run. |
| `duration` | Float | `45.2` | Elapsed time in seconds (`end_time - start_time`) for the run. |
| `records_processed` | Integer | `10` | Records successfully processed. Extracted from `len(data)` in Producer or `df.count()` in Consumer Spark task (0 if failed early). |
| `retry_count` | Integer | `0` | Sum of task retry attempts during this run (`ti.try_number - 1`). |
| `memory_mb` | Float | `142.5` | Peak process memory (RSS) in megabytes measured via Python's standard `psutil` library (`psutil.Process().memory_info().rss / 1e6`). |
| `failure_scenario` | String | `NONE` | Identifier of test condition: `NONE`, `API_TIMEOUT`, `MISSING_S3_FILE`, or `DATA_QUALITY_NULL`. |
| **`status`** | **Integer** | **`0`** | **Ground-truth outcome label:** `0 = SUCCESS`, `1 = FAILED`. |

### Why This Is Beginner-Friendly & Defensible
- **No complex monitoring agents:** Only 8 intuitive columns.
- **Transparent:** You can open `pipeline_telemetry.csv` in Excel or VS Code and inspect every single run.
- **Directly linked to code:** Every number corresponds to an actual operation in your existing DAGs.

---

## 3. The Three Controlled Failure Scenarios

To train sequential models, our dataset needs realistic failure events alongside normal successful runs. We choose **3 realistic failure scenarios** mapped directly to existing lines of code in your DAGs:

### Scenario 1: API Request Timeout
- **Where in code:** `producer_users_daily.py` (Line 20: `requests.get("https://jsonplaceholder.typicode.com/users")`).
- **Real-world meaning:** The external data provider is temporarily down or experiencing severe network latency.
- **Safe reproduction:** In test mode, pass a short timeout or an unreachable mock endpoint.
- **Telemetry produced:**
  - `duration`: Spikes to the timeout limit.
  - `retry_count`: Increments as Airflow retries the task up to 5 times.
  - `records_processed`: `0`.
  - `status`: `1` (Failure).
  - `failure_scenario`: `"API_TIMEOUT"`.

### Scenario 2: Missing S3 Input File
- **Where in code:** `consumer_users_daily.py` (Line 142: `S3KeySensor(bucket_key="data/sample/{{ ds }}.json")`).
- **Real-world meaning:** Upstream producer failed to stage the file, or there was a file naming/partition mismatch.
- **Safe reproduction:** In test mode, point the sensor to a non-existent test date partition (or test with a short sensor timeout).
- **Telemetry produced:**
  - `duration`: Polling duration reaches sensor timeout.
  - `records_processed`: `0` (Consumer PySpark transformation is never reached).
  - `status`: `1` (Failure).
  - `failure_scenario`: `"MISSING_S3_FILE"`.

### Scenario 3: Data-Quality Validation Failure (Null Values)
- **Where in code:** `consumer_users_daily.py` (Line 51–56: `null_count = df.filter(col("id").isNull() | col("email").isNull()).count()`).
- **Real-world meaning:** Upstream data schema corrupted or incomplete user records received from source.
- **Safe reproduction:** In test mode, inject a single mock record with `email: null`.
- **Telemetry produced:**
  - PySpark immediately raises `ValueError("Data quality check failed. X rows contain null values.")`.
  - `records_processed`: Ingested 10, but 0 loaded to PostgreSQL.
  - `status`: `1` (Failure).
  - `failure_scenario`: `"DATA_QUALITY_NULL"`.

> **Preservation Rule:** The baseline production DAGs ([`producer_users_daily.py`](file:///home/ayush/airflow/dags/producer_users_daily.py) and [`consumer_users_daily.py`](file:///home/ayush/airflow/dags/consumer_users_daily.py)) remain clean and unmodified. Failure experiments run through a parameter flag or a dedicated experiment script.

---

## 4. The Machine-Learning Problem Explained Simply

### What Does One Row Represent?
One row in `pipeline_telemetry.csv` represents **one complete run** of the pipeline:
`[run_010, 2026-11-01 12:00, duration=42.1s, records=10, retries=0, memory=140MB, scenario=NONE, status=0]`

### What Does One Sequence Represent?
One sequence is a sliding window of the **previous $T$ completed pipeline runs** (e.g., $T = 3$ or $T = 5$):
- **Input sequence for predicting Run 4:** `[Run 1, Run 2, Run 3]`
- **Input sequence for predicting Run 5:** `[Run 2, Run 3, Run 4]`
- **Input sequence for predicting Run 6:** `[Run 3, Run 4, Run 5]`

### What Does the Model Predict?
The model outputs a single decimal number between `0.0` and `1.0`:
- **Failure Probability** of the upcoming run:
  - Output `< 0.30` $\to$ **Risk: LOW**
  - Output `0.30 – 0.70` $\to$ **Risk: MEDIUM**
  - Output `> 0.70` $\to$ **Risk: HIGH**

### How Do RNN, LSTM, and GRU Differ at a Basic Level?
- **Vanilla RNN (Recurrent Neural Network):**
  - Uses a single hidden state passed from one step to the next.
  - *Strength:* Very fast and simple.
  - *Weakness:* Suffers from the "vanishing gradient" problem—it easily forgets older runs in the sequence.
- **LSTM (Long Short-Term Memory):**
  - Introduces a separate "cell state" (long-term memory) regulated by three gates: **Forget Gate**, **Input Gate**, and **Output Gate**.
  - *Strength:* Can remember degradation patterns across many past runs without forgetting.
  - *Weakness:* More mathematical operations and parameters.
- **GRU (Gated Recurrent Unit):**
  - A streamlined, modern variation of LSTM with only two gates: **Reset Gate** and **Update Gate**.
  - *Strength:* Retains the memory advantages of LSTM, but trains faster and has fewer parameters—ideal for small to medium telemetry datasets.

### How We Avoid Data Leakage
- **Strict Pre-Execution Boundary:** When predicting Run $t$, the model **only** receives data from completed runs $[t-T \dots t-1]$. It **never** receives Run $t$'s own duration, error logs, or status, because at prediction time, Run $t$ hasn't finished yet.
- **Chronological Split:** We divide data in time order:
  - **Train:** First 70% of runs.
  - **Validation:** Next 15% of runs.
  - **Test:** Final 15% of runs.
  - We *never* use random shuffle/k-fold splitting, which would leak future temporal trends into the past.

---

## 5. Minimal Files to Create or Modify

We avoid bloated tooling, external databases, or complex plugins. Only a few focused files are required:

```text
data-pipeline-failure-prediction-system/
│
├── data/
│   └── raw/
│       └── pipeline_telemetry.csv         <-- Single CSV file storing telemetry
│
├── src/
│   ├── data/
│   │   ├── telemetry_logger.py            <-- Simple helper to log run metrics to CSV
│   │   └── make_dataset.py                <-- Clean & load CSV into training arrays
│   ├── features/
│   │   └── sequence_generator.py          <-- Builds sliding windows (T past runs -> next run)
│   ├── models/
│   │   ├── rnn_model.py                   <-- Simple PyTorch RNN
│   │   ├── lstm_model.py                  <-- Simple PyTorch LSTM
│   │   ├── gru_model.py                   <-- Simple PyTorch GRU
│   │   └── train.py                       <-- Unified training loop & comparison
│   └── evaluation/
│       └── evaluate.py                    <-- Computes Accuracy, Recall, ROC-AUC, Confusion Matrix
│
└── airflow/
    └── dags/
        └── experiment_etl_pipeline.py     <-- Clean experiment DAG to run normal & 3 failure tests
```

---

## 6. Mid-Review Presentation Guide (For Professor & Teammates)

When presenting next week, explain the project using this 5-point script:

### Slide 1: The Practical Problem
> "Modern data engineering pipelines fail due to fluctuating data volumes, network drops, and corrupted records. Current tools like Airflow only alert engineers *after* a pipeline crashes. Our project builds a proactive failure prediction system that forecasts failure risk *before* or *during* execution."

### Slide 2: Real-World Architecture (Not Fake Data)
> "Instead of using synthetic Kaggle numbers, our project is built around an actual, working Apache Airflow Producer–Consumer ETL pipeline. The pipeline ingests user data from an API, stages raw JSON in AWS S3, cleans and validates schema in Apache PySpark, and loads records idempotently into PostgreSQL."

### Slide 3: Telemetry Collection & Controlled Failure Modes
> "Each pipeline execution logs 8 key metrics to a clean CSV: duration, record count, retries, memory, scenario, and status. To ensure our model learns realistic failure patterns, we test 3 controlled production failure scenarios:
> 1. API request timeouts
> 2. Missing S3 input files
> 3. Data-quality null value violations in PySpark."

### Slide 4: Track 2 — Sequential Modeling
> "Why sequential models? Because pipeline health is a sequence over time. A single run using 140MB memory looks fine, but if the previous runs used 60MB, 90MB, and 120MB, the temporal trend reveals a compounding problem. We use a sliding window of past runs $[t-3 \dots t-1]$ to predict run $t$ without data leakage."

### Slide 5: Model Comparison & Metric Priority
> "We compare three sequential architectures: Vanilla RNN (baseline), LSTM (gated long-term memory), and GRU (efficient gating). In pipeline monitoring, a False Negative (a pipeline crashes silently without warning) is much worse than a False Positive. Therefore, our evaluation prioritizes **Recall** and **PR-AUC** over raw accuracy."
