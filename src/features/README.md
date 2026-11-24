# Feature Engineering Modules (`src/features/`)

## 1. What This Folder Is
This subpackage will contain the logic for transforming preprocessed tabular telemetry into model-ready features and temporal sequence tensors.

## 2. Why This Folder Exists
Sequential deep learning architectures (RNN, LSTM, GRU) require 3-dimensional tensor inputs shaped as `(batch_size, time_steps, feature_dim)`. A single execution row from a table cannot be directly ingested. 

This folder provides the essential mathematical and data-engineering bridge between static tabular pipeline logs and sequential neural-network inputs.

## 3. What Will Eventually Be Stored Here
When implemented, this directory will host:
- **`build_features.py`**: Extraction of domain-specific pipeline health indicators, such as:
  - Rolling failure rates over the previous $k$ runs
  - Moving averages and trends of execution duration and memory consumption
  - Record error ratios (`failed_records / records_processed`)
  - Resource saturation indicators (CPU/memory spike flags)
- **`scaler.py`**: Feature normalization and standardization (e.g., `StandardScaler`, `MinMaxScaler`, or `RobustScaler`) fitted strictly on training splits to prevent data leakage.
- **`sequence_generator.py`**: Sliding-window algorithms that bundle historical runs $[t-W, \dots, t-1]$ as an input sequence $X_t$ to predict the target execution outcome $y_t$ at time $t$.

## 4. How It Connects to the Overall Project
- **Input:** Consumes cleaned, sorted tabular data from `data/processed/`.
- **Output:** Produces sequenced feature arrays $(X_{\text{train}}, y_{\text{train}}, X_{\text{val}}, y_{\text{val}}, X_{\text{test}}, y_{\text{test}})$ passed directly into model training scripts in `src/models/`.

## 5. What We Are NOT Doing Yet
- We are **not** calculating rolling metrics, scaling arrays, or generating sliding-window sequences yet.
- Feature engineering code will be implemented during **Phase 4** and **Phase 5**.
