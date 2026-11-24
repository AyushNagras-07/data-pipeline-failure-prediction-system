# Data Directory (`data/`)

## 1. What This Folder Is
This directory is the centralized storage location for all pipeline telemetry datasets utilized across the lifecycle of the project.

## 2. Why This Folder Exists
A structured data tier separates immutable source records from intermediate transformed files and final training matrices. Maintaining an explicit separation prevents accidental data corruption, guarantees reproducibility of preprocessing experiments, and simplifies data pipeline auditing.

## 3. Directory Breakdown
The `data/` folder is divided into two primary subdirectories:

- **`raw/`**: Holds the original, immutable pipeline execution telemetry records collected from pipeline logs or telemetry extractors. No manual edits or cleaning occur directly in this folder.
- **`processed/`**: Holds cleaned, validated, standardized, and formatted datasets produced by the preprocessing scripts in `src/data/`. These files are structured and ready for feature engineering and sequence generation.

### Raw vs. Processed Data
| Aspect | Raw Data (`data/raw/`) | Processed Data (`data/processed/`) |
| :--- | :--- | :--- |
| **Origin** | Direct exporter logs / pipeline telemetry runs | Output of validation & cleaning pipelines |
| **Immutability** | Strictly read-only / immutable | Generated programmatically; reproducible |
| **Data Quality** | May contain nulls, outliers, or unparsed timestamps | Imputed, validated, filtered, and type-checked |
| **Downstream Usage**| Consumed only by preprocessing scripts (`src/data/`) | Consumed by feature engineering & sequence creation (`src/features/`) |

## 4. How It Connects to the Overall Project
Data flows from external pipeline execution tracking into `data/raw/`. The processing scripts in `src/data/` ingest this raw data, validate schema consistency, handle missing values, and save normalized tabular files to `data/processed/`. The feature extraction modules then consume `data/processed/` to build sequential training windows for the deep learning models (`src/models/`).

## 5. What We Are NOT Doing Yet
- We are **not** generating synthetic datasets or copying mock telemetry at this stage.
- We are **not** creating binary serialization formats, databases, or cache files.
- Dataset creation and collection will take place during **Phase 2**.
