# Raw Data Directory (`data/raw/`)

## 1. What This Folder Is
The `data/raw/` directory is reserved for raw, unmodified pipeline execution telemetry records gathered directly from pipeline runs.

## 2. Why This Folder Exists
In data engineering and machine learning workflows, raw data must be treated as an immutable source of truth. Having an isolated raw data repository ensures that:
- Raw telemetry can be reprocessed at any time without losing original values.
- Preprocessing pipelines can be tested against fixed, reproducible source data.
- Downstream transformations never inadvertently alter the original observations.

## 3. What Will Eventually Be Stored Here
Future telemetry exports will be stored here in structured tabular formats (such as CSV or JSONL). Each record will represent an individual pipeline or task execution instance.

### Example Candidate Telemetry Schema
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `execution_id` | String / UUID | Unique identifier for the specific pipeline execution run |
| `timestamp` | ISO 8601 Datetime | Scheduled or actual execution start timestamp |
| `pipeline_name` | String | Name or identifier of the executed pipeline/DAG |
| `duration` | Float (seconds) | Total execution duration from start to finish |
| `records_processed` | Integer | Total volume of records ingested or transformed |
| `failed_records` | Integer | Count of bad or dropped records during execution |
| `retry_count` | Integer | Number of retry attempts triggered during the run |
| `cpu_usage` | Float (percentage) | Average or peak CPU utilization measured during the run |
| `memory_usage` | Float (MB / %) | Average or peak memory consumed during the run |
| `queue_time` | Float (seconds) | Delay between task scheduling and worker pickup |
| `status` | String / Integer | Execution outcome: `SUCCESS` (0) or `FAILED` (1) |

## 4. How It Connects to the Overall Project
The data placed here will serve as the raw input to the data cleaning and validation scripts located in `src/data/`. Once cleaned and verified, outputs will be written to `data/processed/`.

## 5. What We Are NOT Doing Yet
- We are **not** creating fake datasets, mock CSVs, or simulated telemetry records yet.
- Actual telemetry collection and curation will be handled systematically in **Phase 2**.
