# Processed Data Directory (`data/processed/`)

## 1. What This Folder Is
The `data/processed/` directory is reserved for clean, normalized, and preprocessed datasets produced by our data pipeline scripts.

## 2. Why This Folder Exists
Raw telemetry often contains missing metrics, unparsed datetime strings, varied scales across features, or occasional duplicate runs. The `processed/` directory stores cleaned datasets where:
- Inconsistent types are cast and timestamps are parsed into standardized UTC datetime formats.
- Missing values and corrupted records are systematically addressed (e.g., filtered or imputed).
- Telemetry records are sorted chronologically per pipeline.
- Schema integrity is guaranteed so downstream modules can consume tabular data without defensive parsing.

## 3. What Will Eventually Be Stored Here
This folder will store:
- **Cleaned Telemetry Tables:** Intermediate CSV or Parquet files containing verified and sorted execution logs.
- **Aggregated Pipeline Runs:** Consolidated execution histories enriched with standardized metrics ready for temporal sequence windowing.

## 4. How It Connects to the Overall Project
The data in this directory is the direct output of `src/data/` (data validation and preprocessing) and serves as the immediate input to `src/features/` (feature extraction and temporal sequence windowing). The downstream feature engineering scripts read from `data/processed/` to build historical observation tensors $(X, y)$ for sequential model training in `src/models/`.

## 5. What We Are NOT Doing Yet
- We are **not** generating processed data files at this stage.
- Processed datasets will be generated only after raw telemetry is collected and preprocessing logic is implemented in **Phase 3**.
