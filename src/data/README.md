# Data Processing Modules (`src/data/`)

## 1. What This Folder Is
This subpackage will contain the Python scripts responsible for loading, validating, cleaning, and preprocessing pipeline execution telemetry.

## 2. Why This Folder Exists
Telemetry collected from execution environments (such as Airflow metadata databases or worker log collectors) cannot be fed directly into deep learning models. It requires systematic extraction, parsing, type-casting, and quality validation. Centralizing this logic within `src/data/` ensures repeatable preprocessing pipelines.

## 3. What Will Eventually Be Stored Here
When implemented, this folder will contain modules such as:
- **`loader.py`**: Utilities to ingest raw CSV, JSONL, or database telemetry exports from `data/raw/`.
- **`validator.py`**: Data validation checks to verify expected schemas, identify corrupted records, and ensure valid ranges for critical metrics (e.g., non-negative durations, valid status codes).
- **`preprocessor.py`**: Transformation logic to parse timestamps, handle missing values, resolve duplicate execution IDs, and standardize columns.
- **`make_dataset.py`**: An executable CLI or pipeline script that coordinates loading, cleaning, and saving outputs directly into `data/processed/`.

## 4. How It Connects to the Overall Project
This subpackage bridges the gap between raw data storage (`data/raw/`) and structured datasets (`data/processed/`). It acts as the initial data-engineering barrier, ensuring that bad telemetry records or unhandled null values never propagate to the feature engineering stage (`src/features/`).

## 5. What We Are NOT Doing Yet
- We are **not** implementing data validation or cleaning scripts at this stage.
- Code in this folder will be written during **Phase 3** (Data Validation & Preprocessing).
